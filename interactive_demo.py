import gradio as gr
import json
import base64
from PIL import Image
import io
import os
from typing import Dict, Any
import uuid
import time
from datetime import datetime
import threading
import queue

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from test import (
    T2IConfig, 
    create_t2i_workflow,
    load_models,
    IntentionAnalyzer,
    ModelSelector,
    execute_model,
    make_gen_image_judge_prompt,
    initialize_llms,
    get_bbox_from_gpt
)

from utils import setup_logging

from models.Grounded_SAM2.test_REF import referring_expression_segmentation
from models.PowerPaint.test import generate_mask_from_bbox, dilate_mask, parse_bbox


# global config variable, used to share between modules
config = None
logger = None

class T2ICopilotAdapter:
    """adapter: adapt the LangGraph workflow in main.py to the Gradio interface"""
    
    def __init__(self):
        global logger
        self.sessions: Dict[str, 'SessionAdapter'] = {}
        
        # initialize the model (gracefully handle missing API key)
        try:
            self.llm, self.llm_json = initialize_llms()
        except Exception as e:
            print("[WARN] Failed to initialize LLMs. Set your OpenAI API key and click 'Apply API Key'.")
            self.llm, self.llm_json = None, None
        load_models()
        print("T2I-Copilot Adapter initialized")
    
    def set_openai_api_key(self, api_key: str):
        """Set/OpenAI API key at runtime and reinitialize LLMs. Propagate to existing sessions."""
        if api_key and api_key.strip():
            os.environ['OPENAI_API_KEY'] = api_key.strip()
            try:
                self.llm, self.llm_json = initialize_llms()
                # propagate to existing sessions
                for _sid, sess in self.sessions.items():
                    sess.llm = self.llm
                    sess.llm_json = self.llm_json
                print("[INFO] OpenAI API key applied and LLMs reinitialized.")
            except Exception as e:
                print(f"[ERROR] Reinitializing LLMs failed: {e}")
    
    def get_or_create_session(self, session_id: str, human_in_loop: bool = False) -> 'SessionAdapter':
        """get or create the session adapter"""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionAdapter(session_id, self.llm, self.llm_json, human_in_loop)
        return self.sessions[session_id]

class SessionAdapter:
    """session adapter: manage the state and interaction of a single session"""
    
    def __init__(self, session_id: str, llm, llm_json, human_in_loop: bool = False):
        global logger, config
        self.session_id = session_id
        self.llm = llm
        self.llm_json = llm_json
        self.config = T2IConfig(human_in_loop=human_in_loop)
        self.workflow = create_t2i_workflow()
        self.messages = []
        self.generated_images = []
        self.current_step = "waiting_for_input"
        self.pending_user_input = None
        self.user_input_queue = queue.Queue()
        self.logger = None
        
        # set the log
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"demo_logs/{session_id}_{session_timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        self.logger = setup_logging(log_dir, filename=f"session.log", console_output=False)
        self.config.logger = self.logger

        self.config.save_dir = log_dir
        self.config.image_index = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # update the global logger and config reference
        logger = self.logger
        config = self.config
    
    def process_user_input(self, message: str) -> Dict[str, Any]:
        """process user input, using the workflow in main.py"""
        global logger
        
        # add user message
        self.messages.append({"role": "user", "content": message})
        
        try:
            # process based on the current step
            if self.current_step == "waiting_for_input":
                return self._handle_new_input(message)
            elif self.current_step == "waiting_for_clarification":
                return self._handle_user_clarification(message)
            elif self.current_step == "feedback_needed":
                return self._handle_user_feedback(message)
            else:
                self.current_step = "waiting_for_input"
                return self._handle_new_input(message)
                
        except Exception as e:
            import traceback
            error_msg = f"Error in workflow execution: {str(e)}"
            stack_trace = traceback.format_exc()
            
            # record detailed error information
            self.logger.error(error_msg)
            self.logger.error(f"Stack trace: {stack_trace}")
            
            # print error information in the console
            print(f"ERROR: {error_msg}")
            print(f"Stack trace: {stack_trace}")
            
            return {
                "response": f"Error occurred during processing: {str(e)}\n\nDetailed error information has been recorded in the log file.",
                "step": "error",
                "images": self.generated_images
            }
    
    def _handle_new_input(self, message: str) -> Dict[str, Any]:
        """process new user input"""
        global logger, config
        # use the IntentionAnalyzer in main.py
        analyzer = IntentionAnalyzer(self.llm, logger, config)
        
        try:     
            analysis = analyzer.analyze_prompt(message, config.prompt_understanding["creativity_level"])
            config.prompt_understanding["prompt_analysis"] = json.dumps(analysis)
            config.prompt_understanding["original_prompt"] = message
            
            questions = analyzer.retrieve_questions(analysis, config.prompt_understanding["creativity_level"])
            
            if questions in ["SUFFICIENT_DETAIL", "AUTOCOMPLETE"]:
                refinement_result = analyzer.refine_prompt_with_analysis(
                    message,
                    analysis,
                    creativity_level=config.prompt_understanding["creativity_level"]
                )
                config.prompt_understanding["refined_prompt"] = refinement_result["refined_prompt"]
                
                return self._handle_model_selection()
            else:
                # need user clarification
                self.current_step = "waiting_for_clarification"
                return {
                    "response": f"I need more information to understand your request: \n\n{questions}",
                    "step": "clarification_needed",
                    "images": self.generated_images
                }
        except Exception as e:
            import traceback
            error_msg = f"Error in _handle_new_input: {str(e)}"
            stack_trace = traceback.format_exc()
            
            logger.error(error_msg)
            logger.error(f"Stack trace: {stack_trace}")
            print(f"ERROR: {error_msg}")
            print(f"Stack trace: {stack_trace}")
            
            raise
    
    def _handle_user_feedback(self, user_input: str) -> Dict[str, Any]:
        """process user feedback after evaluation and trigger regeneration"""
        global logger, config
        # allow user to finish the session explicitly
        if user_input.strip().lower() in {"done", "finish", "end", "完成", "結束", "ok"}:
            self.current_step = "completed"
            return {
                "response": "✅ Session completed. Start a new session to generate another image.",
                "step": "completed",
                "images": self.generated_images
            }

        try:
            # store feedback to current config
            current_config = config.get_current_config()
            current_config["user_feedback"] = user_input

            # create a new regeneration config and carry over previous outputs as reference
            config.add_regeneration_config()
            prev_config = config.get_prev_config()
            new_config = config.get_current_config()

            # if user provided an editing mask previously, carry it over as reference mask
            if prev_config.get("editing_mask"):
                new_config["reference_mask_dir"] = prev_config["editing_mask"]

            # proceed with model selection and regeneration
            return self._handle_model_selection()
        except Exception as e:
            import traceback
            error_msg = f"Error in _handle_user_feedback: {str(e)}"
            stack_trace = traceback.format_exc()
            logger.error(error_msg)
            logger.error(f"Stack trace: {stack_trace}")
            print(f"ERROR: {error_msg}")
            print(f"Stack trace: {stack_trace}")
            raise

    def _handle_user_clarification(self, user_input: str) -> Dict[str, Any]:
        """process user clarification input"""
        global logger
        analyzer = IntentionAnalyzer(self.llm, logger)
        config.prompt_understanding['user_clarification'] = user_input
        
        try:
            analysis = json.loads(config.prompt_understanding["prompt_analysis"])
            refinement_result = analyzer.refine_prompt_with_analysis(
                config.prompt_understanding["original_prompt"],
                analysis,
                user_input,
                config.prompt_understanding["creativity_level"]
            )
            config.prompt_understanding["refined_prompt"] = refinement_result["refined_prompt"]
            
            return self._handle_model_selection()
        except Exception as e:
            import traceback
            error_msg = f"Error in _handle_user_clarification: {str(e)}"
            stack_trace = traceback.format_exc()
            
            logger.error(error_msg)
            logger.error(f"Stack trace: {stack_trace}")
            print(f"ERROR: {error_msg}")
            print(f"Stack trace: {stack_trace}")
            
            raise

    def _handle_model_selection(self) -> Dict[str, Any]:

        global logger, config
        selector = ModelSelector(self.llm, logger, config)
        current_config = config.get_current_config()
        if config.regeneration_count != 0:
            prev_regen_config = config.get_prev_config()
        
        try:
            # Select the most suitable model
            model_selection = selector.select_model()
            logger.debug(f"model_selection: {model_selection}")
            
            # Update current config with model selection
            current_config["selected_model"] = model_selection["selected_model"]
            if config.regeneration_count == 0:
                # if not regen, when creating new config, it would put the prev gen image as the reference content image
                if "reference_content_image" in model_selection:
                    current_config["reference_content_image"] = model_selection["reference_content_image"]
                else:
                    current_config["reference_content_image"] = None
            current_config["generating_prompt"] = model_selection["generating_prompt"]
            current_config["unwanted_object"] = model_selection["unwanted_object"]
            current_config["task_type"] = model_selection["task_type"]
            if "bbox_coordinates" in model_selection:
                current_config["bbox_coordinates"] = model_selection["bbox_coordinates"]
            else:
                current_config["bbox_coordinates"] = None
            current_config["reasoning"] = model_selection["reasoning"]
            current_config["confidence_score"] = model_selection["confidence_score"]

            # Initialize mask_image_path to None
            mask_image_path = None

            # NOTE: if model selection is editing, ask user for given the mask or call open-vocabulary model for mask generation
            if model_selection['selected_model'] == "PowerPaint":
                logger.info(f"Selected model is PowerPaint, determining mask generation approach")
                
                # Handle mask based on task type and human-in-the-loop mode
                if config.regeneration_count == 0:
                    # if config.is_human_in_loop and not current_config.get("reference_mask_dir"):
                    #     # In human-in-loop mode, caller should handle getting mask from user
                    #     # Don't return here - caller will check and request mask if needed
                    #     logger.info(f"Human-in-the-loop mode: mask needed for {current_config['reference_content_image']}")
                    #     self.current_step = "waiting_for_mask"
                    #     return {
                    #         "response": "✏️ Please draw the editing mask on the image, then click 'Generate!'.",
                    #         "step": "mask_needed",
                    #         "images": self.generated_images,
                    #         "reference_image": current_config["reference_content_image"],
                    #     }
                    # else:
                    # For automated mask generation based on task type
                    if not current_config.get("reference_mask_dir") and current_config["task_type"] == "text-guided":
                        if current_config["unwanted_object"] is not None:
                            logger.info("Generating unwanted object mask by RES for text-guided inpainting")
                            try:
                                # Call the referring_expression_segmentation function
                                sam_mask_path = referring_expression_segmentation(
                                    image_path=current_config["reference_content_image"],
                                    text_input=current_config["unwanted_object"],
                                    output_dir=config.save_dir
                                )
                                # expand the mask to make the mask boundary unavailiable
                                print(f"Expanding mask: {sam_mask_path}")
                                sam_mask_path = dilate_mask(sam_mask_path)
                                if sam_mask_path and os.path.exists(sam_mask_path):
                                    current_config["reference_mask_dir"] = sam_mask_path
                                    print(f"Using SAM-generated mask: {current_config["reference_mask_dir"]}")
                                else:
                                    print("Failed to generate mask with SAM.")
                            except Exception as e:
                                print(f"Error generating mask with SAM: {e}")
                        elif current_config["bbox_coordinates"] is not None and not current_config.get("reference_mask_dir"):
                            logger.info(f"Generating mask for text-guided inpainting: '{current_config["generating_prompt"]}'")
                            gpt_mask_path = generate_mask_from_bbox(parse_bbox(current_config["bbox_coordinates"]), current_config["reference_content_image"])
                            if gpt_mask_path:
                                current_config["reference_mask_dir"] = gpt_mask_path
                                logger.info(f"Using bbox-generated mask: {current_config["reference_mask_dir"]}")
                            else:
                                logger.info("Failed to generate mask from bbox.")
                        
                        elif (current_config["bbox_coordinates"] is None or current_config.get("reference_mask_dir") is None) and not current_config.get("reference_mask_dir"):
                            # an exception for bbox is None, given the image and the prompt for gpt to again provide the bbox for the referrring object's coordinates
                            
                            logger.info(f"Generating bounding box coordinates for text-guided inpainting: '{current_config['generating_prompt']}'")
        
                            # Call GPT to provide the bbox for the referring object's coordinates
                            bbox_coords_or_mask_path, using_task_type = get_bbox_from_gpt(
                                image_path=current_config["reference_content_image"],
                                prompt=current_config["generating_prompt"],
                                unwanted_object=current_config["unwanted_object"]
                            )
                            
                            if using_task_type == "bbox":
                                # Update the bbox_coordinates in the current config
                                current_config["bbox_coordinates"] = bbox_coords_or_mask_path
                                logger.info(f"Using GPT-generated bbox coordinates: {bbox_coords_or_mask_path}")
                                # Generate mask from the bbox coordinates
                                mask_image_path = generate_mask_from_bbox(parse_bbox(current_config["bbox_coordinates"]), current_config["reference_content_image"])
                            elif using_task_type == "mask":
                                mask_image_path = bbox_coords_or_mask_path
                                logger.info(f"Using RES-generated mask: {mask_image_path}")

                            mask_image_path = dilate_mask(mask_image_path)
                            current_config["reference_mask_dir"] = mask_image_path
                            
                    elif not current_config.get("reference_mask_dir") and current_config["task_type"] == "object-removal" and referring_expression_segmentation is not None:
                        logger.info(f"Generating mask for object removal inpainting: '{current_config["unwanted_object"]}'")
                        try:
                            # Call the referring_expression_segmentation function
                            sam_mask_path = referring_expression_segmentation(
                                image_path=current_config["reference_content_image"],
                                text_input=current_config["unwanted_object"],
                                output_dir=config.save_dir
                            )
                            # expand the mask to make the mask boundary unavailiable
                            print(f"Expanding mask: {sam_mask_path}")
                            sam_mask_path = dilate_mask(sam_mask_path)
                            if sam_mask_path and os.path.exists(sam_mask_path):
                                current_config["reference_mask_dir"] = sam_mask_path
                                print(f"Using SAM-generated mask: {current_config["reference_mask_dir"]}")
                            else:
                                print("Failed to generate mask with SAM.")
                        except Exception as e:
                            print(f"Error generating mask with SAM: {e}")

                elif prev_regen_config["editing_mask"] is None:
                    # if config.is_human_in_loop and not current_config.get("reference_mask_dir"):
                    #     logger.info(f"Human-in-the-loop mode enabled. Requesting mask for {current_config['reference_content_image']}...")
                    #     self.current_step = "waiting_for_mask"
                    #     return {
                    #         "response": "✏️ Please draw the editing mask on the image, then click 'Generate!'.",
                    #         "step": "mask_needed",
                    #         "images": self.generated_images,
                    #         "reference_image": current_config["reference_content_image"],
                    #     }
                    # else:
                    # For automated mask generation based on task type
                    if not current_config.get("reference_mask_dir") and current_config["task_type"] == "text-guided":
                        if current_config["unwanted_object"] is not None:
                            logger.info("Generating unwanted object mask by RES for text-guided inpainting")
                            try:
                                # Call the referring_expression_segmentation function
                                sam_mask_path = referring_expression_segmentation(
                                    image_path=current_config["reference_content_image"],
                                    text_input=current_config["unwanted_object"],
                                    output_dir=config.save_dir
                                )
                                # expand the mask to make the mask boundary unavailiable
                                print(f"Expanding mask: {sam_mask_path}")
                                sam_mask_path = dilate_mask(sam_mask_path)
                                if sam_mask_path and os.path.exists(sam_mask_path):
                                    current_config["reference_mask_dir"] = sam_mask_path
                                    print(f"Using SAM-generated mask: {current_config["reference_mask_dir"]}")
                                else:
                                    print("Failed to generate mask with SAM.")
                            except Exception as e:
                                print(f"Error generating mask with SAM: {e}")
                        elif current_config["bbox_coordinates"] is not None and not current_config.get("reference_mask_dir"):
                            logger.info(f"Generating mask for text-guided inpainting: '{current_config["generating_prompt"]}'")
                            gpt_mask_path = generate_mask_from_bbox(parse_bbox(current_config["bbox_coordinates"]), current_config["reference_content_image"])
                            if gpt_mask_path:
                                current_config["reference_mask_dir"] = gpt_mask_path
                                logger.info(f"Using bbox-generated mask: {current_config["reference_mask_dir"]}")
                            else:
                                logger.info("Failed to generate mask from bbox.")
                        
                        elif (current_config["bbox_coordinates"] is None or current_config.get("reference_mask_dir") is None) and not current_config.get("reference_mask_dir"):
                            # an exception for bbox is None, given the image and the prompt for gpt to again provide the bbox for the referrring object's coordinates
                            
                            logger.info(f"Generating bounding box coordinates for text-guided inpainting: '{current_config['generating_prompt']}'")
        
                            # Call GPT to provide the bbox for the referring object's coordinates
                            bbox_coords_or_mask_path, using_task_type = get_bbox_from_gpt(
                                image_path=current_config["reference_content_image"],
                                prompt=current_config["generating_prompt"],
                                unwanted_object=current_config["unwanted_object"]
                            )
                            
                            if using_task_type == "bbox":
                                # Update the bbox_coordinates in the current config
                                current_config["bbox_coordinates"] = bbox_coords_or_mask_path
                                logger.info(f"Using GPT-generated bbox coordinates: {bbox_coords_or_mask_path}")
                                # Generate mask from the bbox coordinates
                                mask_image_path = generate_mask_from_bbox(parse_bbox(current_config["bbox_coordinates"]), current_config["reference_content_image"])
                            elif using_task_type == "mask":
                                mask_image_path = bbox_coords_or_mask_path
                                logger.info(f"Using RES-generated mask: {mask_image_path}")

                            mask_image_path = dilate_mask(mask_image_path)
                            current_config["reference_mask_dir"] = mask_image_path

                    elif not current_config.get("reference_mask_dir") and current_config["task_type"] == "object-removal" and referring_expression_segmentation is not None:
                        logger.info(f"Generating mask for object removal inpainting: '{current_config["unwanted_object"]}'")

                        # Call the referring_expression_segmentation function
                        sam_mask_path = referring_expression_segmentation(
                            image_path=current_config["reference_content_image"],
                            text_input=current_config["unwanted_object"],
                            output_dir=config.save_dir
                        )
                        # expand the mask to make the mask boundary unavailiable
                        print(f"Expanding mask: {sam_mask_path}")
                        sam_mask_path = dilate_mask(sam_mask_path)
                        if sam_mask_path and os.path.exists(sam_mask_path):
                            current_config["reference_mask_dir"] = sam_mask_path
                            print(f"Using SAM-generated mask: {current_config["reference_mask_dir"]}")
                        else:
                            print("Failed to generate mask with SAM.")

                else:
                    # Use the mask from the previous regeneration (which user provided when evaluation)
                    current_config["reference_mask_dir"] = config.get_prev_config()["editing_mask"]

            image_path = execute_model(
                model_name=current_config['selected_model'],
                prompt=current_config['generating_prompt'],
                task_type=current_config['task_type'],
                mask_dir=current_config['reference_mask_dir'],
                reference_content_image=current_config['reference_content_image'],
                logger=logger,
                config=config
            )
            
            print("----- image_path -----")
            print("image_path = ", image_path)
            current_config["gen_image_path"] = image_path
            self.generated_images.append(image_path)
            
            return self._handle_evaluation(image_path)
        except Exception as e:

            if config.regeneration_count > 0:
                prev_regen_config = config.get_prev_config()
                prev_image_path = prev_regen_config["gen_image_path"]
                logger.info(f"Returning previous generated image due to error: {prev_image_path}")
                
                current_config["gen_image_path"] = prev_image_path
                self.generated_images.append(prev_image_path)

                return self._handle_evaluation(prev_image_path)

            import traceback
            error_msg = f"Error in _handle_model_selection: {str(e)}"
            stack_trace = traceback.format_exc()
            
            self.logger.error(error_msg)
            self.logger.error(f"Stack trace: {stack_trace}")
            print(f"ERROR: {error_msg}")
            print(f"Stack trace: {stack_trace}")
            
            raise
    
    def _handle_evaluation(self, image_path: str) -> Dict[str, Any]:
        global logger
        current_config = config.get_current_config()
        
        try:
            with open(image_path, "rb") as image_file:
                base64_gen_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            evaluation_prompt = [
                ("system", make_gen_image_judge_prompt(config)),
                ("human", [
                    {
                        "type": "text",
                        "text": f"original prompt: {config.prompt_understanding['original_prompt']}\n Prompt analysis: {config.prompt_understanding['prompt_analysis']}\n Prompt used for generating the image: {current_config['generating_prompt']}\n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_gen_image}"}
                    }
                ])
            ]
            
            evaluation_result = self.llm_json.invoke(evaluation_prompt)
            evaluation_data = json.loads(evaluation_result.content)

            logger.info(f"Evaluation result: {evaluation_data}")
            
            current_config["evaluation_score"] = evaluation_data["overall_score"]
            current_config["improvement_suggestions"] = evaluation_data["improvement_suggestions"]
            
            # if human-in-loop enabled, keep session open for feedback/regeneration
            if config.is_human_in_loop:
                self.current_step = "feedback_needed"
                return {
                    "response": (
                        "🎉 Image generated successfully!\n\n"
                        f"Score: {evaluation_data['overall_score']}/10\n\n"
                        f"Suggestions: {evaluation_data['improvement_suggestions']}\n\n"
                        "Please type your editing instructions (e.g., 'make it brighter', 'remove the tree'). "
                        "Reply 'done' when you're satisfied."
                    ),
                    "step": "feedback_needed",
                    "images": self.generated_images
                }
            else:
                # directly complete, no feedback
                self.current_step = "completed"
                return {
                    "response": f"🎉 Image generated successfully! \n\nScore: {evaluation_data['overall_score']}/10\n\nImprovement suggestions: {evaluation_data['improvement_suggestions']}\n\n✅ Generation completed. Please start a new session to generate another image.",
                    "step": "completed",
                    "images": self.generated_images
                }
        except Exception as e:
            import traceback
            error_msg = f"Error in _handle_evaluation: {str(e)}"
            stack_trace = traceback.format_exc()
            
            self.logger.error(error_msg)
            self.logger.error(f"Stack trace: {stack_trace}")
            print(f"ERROR: {error_msg}")
            print(f"Stack trace: {stack_trace}")
            
            raise
    


# create Gradio interface
def create_demo():
    adapter = T2ICopilotAdapter()
    
    def generate_fn(message, history, session_id, human_in_loop, mask_data=None):
        """specialized generate function, with status prompt"""
            # guard: ensure API key is set and LLMs are initialized
        if adapter.llm is None or not os.getenv('OPENAI_API_KEY'):
            if isinstance(history, list):
                history.append([None, "❗ Please set your OpenAI API key in the 'API Settings' box, then click 'Apply API Key'."])
            return "", history, "❗ Missing OpenAI API key", gr.update(interactive=True), gr.update(visible=False)
        session = adapter.get_or_create_session(session_id, human_in_loop)

        # If we're waiting for a mask and mask_data is provided, save it first
        # if session.current_step == "waiting_for_mask" and mask_data is not None:
        #     # Save the mask
        #     mask_img = None
        #     if isinstance(mask_data, dict):
        #         if isinstance(mask_data.get('layers'), list) and len(mask_data['layers']) > 0:
        #             layer0 = mask_data['layers'][0]
        #             mask_img = layer0.getchannel('A') if layer0.mode in ('RGBA', 'LA') else layer0.convert('L')

        #     if mask_img is not None:
        #         from PIL import Image
        #         pil_mask = mask_img if isinstance(mask_img, Image.Image) else Image.fromarray(mask_img)
        #         pil_mask = pil_mask.convert('L').point(lambda p: 255 if p > 0 else 0)

        #         mask_save_dir = session.config.save_dir
        #         os.makedirs(mask_save_dir, exist_ok=True)
        #         mask_path = os.path.join(mask_save_dir, f"{session.config.image_index}_editing_mask.png")
        #         pil_mask.save(mask_path)

        #         # Save composition preview
        #         composition_path = os.path.join(mask_save_dir, f"{session.config.image_index}_composition_mask.png")
        #         current_config = session.config.get_current_config()
        #         ref_path = current_config.get("reference_content_image")
        #         try:
        #             if ref_path and os.path.exists(ref_path):
        #                 base_img = Image.open(ref_path).convert('RGBA')
        #                 red_overlay = Image.new('RGBA', base_img.size, (255, 0, 0, 128))
        #                 overlay_region = Image.composite(red_overlay, Image.new('RGBA', base_img.size, (0, 0, 0, 0)), pil_mask)
        #                 composed = Image.alpha_composite(base_img, overlay_region)
        #                 composed.save(composition_path)
        #             else:
        #                 pil_mask.convert('RGB').save(composition_path)
        #         except:
        #             pil_mask.convert('RGB').save(composition_path)

        #         # Add mask preview to history
        #         if os.path.exists(composition_path):
        #             history.append([None, gr.Image(composition_path)])
        #         history.append([None, "Mask confirmed. Generating..."])

        #         # Set mask in config
        #         current_config["reference_mask_dir"] = mask_path
        #         current_config["editing_mask"] = mask_path

        #         if hasattr(session, 'logger') and session.logger:
        #             session.logger.info(f"Generate: saved mask to {mask_path}")

        # Allow empty message if we're continuing after mask
        if not message.strip() and session.current_step != "waiting_for_mask":
            # keep input enabled; hide mask UI
            return "", history, "Please input the prompt", gr.update(interactive=True), gr.update(visible=False), gr.update(visible=False)

        # process user input
        result = session.process_user_input(message)
        
        # build response content, including text and image
        response_content = result["response"]

        # if there is a generated image, add it to the response
        if result.get("images") and len(result["images"]) > 0:
            # get the latest image
            latest_image = result["images"][-1]
            # print(f"Debug: latest_image path = {latest_image}")
            # print(f"Debug: file exists = {os.path.exists(latest_image)}")
            if os.path.exists(latest_image):
                history.append([message, gr.Image(latest_image)])
                history.append([None, response_content])
        else:
            # update history
            history.append([message, response_content])
        
        
        # return the status information
        status = f"Step: {result['step']}"
        disable_input = False
        mask_editor_update = gr.update(visible=False)
        if result['step'] == 'completed':
            status = "✅ Image generated! Session completed."
            disable_input = True
        elif result['step'] == 'clarification_needed':
            status = "❓ Clarification needed"
        elif result['step'] == 'feedback_needed':
            status = "🔄 Feedback needed"
        # elif result['step'] == 'mask_needed':
        #     status = "✏️ Please draw the editing mask, then click 'Generate!' to continue"
        #     # show mask editor with reference image
        #     ref_img = result.get('reference_image')
        #     mask_editor_update = gr.update(visible=True, value=ref_img)
        
        # set the input box according to the completion status
        if disable_input:
            msg_update = gr.update(interactive=False, placeholder="Session completed. Please start a new session to continue.")
        else:
            msg_update = gr.update(interactive=True)

        print("[DEBUG] status = ", status)

        return "", history, status, msg_update, mask_editor_update

    def reset_session(session_id):
        """reset session"""
        if session_id in adapter.sessions:
            del adapter.sessions[session_id]
        return []
    
    # create interface
    with gr.Blocks(title="T2I-Copilot Demo") as interface:
        gr.Markdown("# 🎨 T2I-Copilot Demo")
        gr.Markdown("T2I-Copilot: A Training-Free Multi-Agent Text-to-Image System for Enhanced Prompt Interpretation and Interactive Generation (ICCV'25)")
        
        with gr.Accordion("Usage", open=False):
            gr.Markdown("""
            1. **Configure settings**：Toggle "👩‍💻 Human in Loop" to enable control before and after the generation process
            2. **Prompt**：In the text box, describe the image you want to generate
            3. **Click generate**：Click the "🚀 Generate!" button or press Enter
            4. **Interactive process**：
               - The system will analyze your requirements
               - If more information is needed, it will ask 
               - Automatically select the most suitable model to generate the image
               - If Human in Loop is enabled, users can provide prompt clarifications and feedback for the generated image
               - Evaluate the image quality and display the result
            5. **View results**：The generated image will be displayed directly in the conversation
            6. **Start new session**：After completion, click "🆕 New session" to generate another image
            
            ### Tips:
            - The more detailed the description, the better the generation effect
            - Each session generates one image and then completes
            - Use "🆕 New session" to start a fresh generation
            """)
        
        with gr.Accordion("API Settings", open=True):
            with gr.Row():
                with gr.Column(scale=3):
                    api_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="sk-...",
                        type="password",
                        lines=1
                    )
                with gr.Column(scale=1):
                    apply_api_key_btn = gr.Button("Apply API Key", variant="secondary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                human_in_loop_toggle = gr.Checkbox(
                    label="🤖 Human in Loop", 
                    value=False,
                    info="Enable to allow human interaction for prompt clarification and feedback"
                )
        
        chatbot = gr.Chatbot(
            height=700,
            label="Conversation",
            show_label=True
        )
        # Mask drawing UI (hidden by default)
        # Note: When mask is needed, user draws on mask_editor then clicks "Generate!" to continue
        mask_editor = gr.ImageMask(label="Draw the editing mask, then click 'Generate!' to continue", visible=False, type='pil')
        with gr.Row():
            with gr.Column(scale=3):
                msg = gr.Textbox(
                    label="Prompt",
                    placeholder="A cute cat sitting in a garden, sunny, with beautiful flowers in the background",
                    lines=1
                )
            with gr.Column(scale=1):
                generate_btn = gr.Button("🚀 Generate!", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column(scale=3):
                status_display = gr.Textbox(
                    label="Current status",
                    value="Waiting for input...",
                    interactive=False,
                    lines=1
                )
            with gr.Column(scale=1):
                new_session_btn = gr.Button("🆕 New session", variant="secondary", size="lg")
            
        
        # generate session ID
        session_id = gr.State(lambda: str(uuid.uuid4()))
        
        # API key apply handler
        def on_apply_api_key(api_key):
            adapter.set_openai_api_key(api_key)
            return "", "🔑 OpenAI API key applied"
        
        # event handling - press Enter to submit
        msg.submit(
            generate_fn,
            [msg, chatbot, session_id, human_in_loop_toggle, mask_editor],
            [msg, chatbot, status_display, msg, mask_editor]
        )

        # event handling - click the generate button
        generate_btn.click(
            generate_fn,
            [msg, chatbot, session_id, human_in_loop_toggle, mask_editor],
            [msg, chatbot, status_display, msg, mask_editor]
        )
        
        # wire API key apply button after status_display exists
        apply_api_key_btn.click(
            on_apply_api_key,
            [api_key_input],
            [api_key_input, status_display]
        )
        
        new_session_btn.click(
            lambda: str(uuid.uuid4()),
            outputs=[session_id]
        ).then(
            reset_session,
            [session_id],
            [chatbot]
        ).then(
            lambda: ("Waiting for input...", gr.update(interactive=True), False, gr.update(visible=False, value=None)),
            outputs=[status_display, msg, human_in_loop_toggle, mask_editor]
        )
    
    return interface

if __name__ == "__main__":
    # set Gradio cache directory to current working directory, avoid permission issues
    import os
    os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), "gradio_cache")
    os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)
    
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=True,
        debug=True
    ) 