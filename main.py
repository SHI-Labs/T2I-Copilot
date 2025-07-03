import os
from typing import Literal, List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
import base64
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")

import argparse
import random
from utils import setup_logging

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY= os.getenv('OPENAI_API_KEY')

# load models
import torch
from PIL import Image
import sys

from diffusers import FluxPipeline 
from models.Grounded_SAM2.test_REF import referring_expression_segmentation
from models.mask_draw_client import request_mask
from models.PowerPaint.test import PowerPaintController, generate_mask_from_bbox, dilate_mask, parse_bbox

import re
import json
from prompts.system_prompts import make_intention_analysis_prompt, make_gen_image_judge_prompt

import time
from tqdm import tqdm

# Initialize model variables
flux_pipe = None
# flux_pipe_editing = None
powerpaint_controller = None

class CreativityLevel(Enum):
    LOW = "low"      # Ask users for most details
    MEDIUM = "medium"  # Fill some details, ask for important ones
    HIGH = "high"    # Autonomously fill in most details

class T2IConfig:
    """Configuration and state management for T2I workflow."""
    def __init__(self, human_in_loop: bool = True):
        # Global settings
        self.is_human_in_loop = human_in_loop
        self.save_dir = ""
        self.seed = None
        self.image_index = None
        self.logger = None
        
        # Open LLM configurations
        self.use_open_llm = False 
        self.open_llm_model = ""  
        self.open_llm_host = ""  
        self.open_llm_port = ""  

        # Prompt understanding configuration
        self.prompt_understanding = {
            "creativity_level": CreativityLevel.MEDIUM if human_in_loop else CreativityLevel.HIGH,
            "original_prompt": "",
            "prompt_analysis": "",  # JSON string
            "questions": None,
            "user_clarification": None,
            "refined_prompt": "",
        }

        # Initialize first regeneration config as default config
        self.regeneration_count = 0
        self.regeneration_configs = {
            "count_0": {
                "selected_model": "",
                "generating_prompt": "",
                "reference_content_image": "",
                "editing_target": "",
                "reference_mask_dir": "",
                "reasoning": "",
                "confidence_score": 0.0,
                "gen_image_path": "",
                "evaluation_score": 0.0,
                "user_feedback": None,
                "improvement_suggestions": None,
                "editing_mask": None, # given by the user when evaluation
                "unwanted_object": None,
                "task_type": None,
                "bbox_coordinates": None,
                "powerpaint_guidance_scale": 0.0  # Default guidance scale for PowerPaint
            }
        }

    def add_regeneration_config(self):
        """Create a new regeneration configuration."""
        index = self.regeneration_count + 1
        
        # Get reference content from previous config
        prev_config = self.regeneration_configs[f"count_{self.regeneration_count}"]
        prev_gen_image_path = prev_config["gen_image_path"]

        self.regeneration_configs[f"count_{index}"] = {
            "selected_model": "",
            "generating_prompt": "",
            "reference_content_image": prev_gen_image_path,
            "editing_target": "",
            "reference_mask_dir": "",
            "reasoning": "",
            "confidence_score": 0.0,
            "gen_image_path": "",
            "evaluation_score": 0.0,
            "user_feedback": None,
            "improvement_suggestions": None,
            "editing_mask": None, # given by the user when evaluation
            "unwanted_object": None,
            "task_type": None,
            "bbox_coordinates": None,
            "powerpaint_guidance_scale": 0.0  # Default guidance scale for PowerPaint
        }
        self.regeneration_count = index
        return f"count_{index}"

    def get_current_config(self):
        """Get the current regeneration configuration."""
        return self.regeneration_configs[f"count_{self.regeneration_count}"]

    def get_prev_config(self):
        """Get the current regeneration configuration."""
        return self.regeneration_configs[f"count_{self.regeneration_count-1}"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for storage."""
        # Convert the CreativityLevel enum to its string value
        prompt_understanding = self.prompt_understanding.copy()
        prompt_understanding["creativity_level"] = prompt_understanding["creativity_level"].value

        # Log the type and content of prompt_analysis before processing
        # self.logger.debug(f"prompt_analysis type: {type(prompt_understanding['prompt_analysis'])}")
        # self.logger.debug(f"prompt_analysis content: {prompt_understanding['prompt_analysis']}")

        # Ensure prompt_analysis is stored as a dictionary, not a JSON string
        if isinstance(prompt_understanding["prompt_analysis"], str):
            # Handle empty string case first
            if not prompt_understanding["prompt_analysis"].strip():
                self.logger.debug("Before prompt analyze, so getting empty prompt_analysis string, using empty dict")
                prompt_understanding["prompt_analysis"] = {}
            else:
                try:
                    prompt_understanding["prompt_analysis"] = json.loads(prompt_understanding["prompt_analysis"])
                    self.logger.debug("Successfully parsed prompt_analysis JSON string")
                except json.JSONDecodeError:
                    self.logger.error("Invalid JSON string in prompt_analysis")
                    try:
                        # Try to preserve the string content if it's not JSON
                        prompt_understanding["prompt_analysis"] = {
                            "raw_content": prompt_understanding["prompt_analysis"]
                        }
                        self.logger.debug("Preserved raw prompt_analysis content")
                    except Exception as e:
                        self.logger.error(f"Failed to preserve prompt analysis content: {e}")
                        prompt_understanding["prompt_analysis"] = {}
        elif isinstance(prompt_understanding["prompt_analysis"], dict):
            self.logger.debug("prompt_analysis is already a dictionary")
        else:
            self.logger.warning(f"Unexpected prompt_analysis type: {type(prompt_understanding['prompt_analysis'])}")
            prompt_understanding["prompt_analysis"] = {}

        # Return the dictionary directly without wrapping it in another JSON string
        return {
            "is_human_in_loop": self.is_human_in_loop,
            "save_dir": self.save_dir,
            "seed": self.seed,
            "image_index": self.image_index,
            "use_open_llm": self.use_open_llm,
            "open_llm_model": self.open_llm_model,
            "open_llm_host": self.open_llm_host,
            "open_llm_port": self.open_llm_port,
            "prompt_understanding": prompt_understanding,
            "regeneration_configs": self.regeneration_configs,
            "regeneration_count": self.regeneration_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'T2IConfig':
        """Create config from stored dictionary."""
        instance = cls()
        
        # Load global settings directly from the dictionary
        instance.is_human_in_loop = data["is_human_in_loop"]
        instance.save_dir = data["save_dir"]
        instance.seed = data["seed"]
        instance.image_index = data["image_index"]
        
        # Load use_open_llm if it exists in the data
        if "use_open_llm" in data:
            instance.use_open_llm = data["use_open_llm"]
        
        # Load open LLM settings if they exist
        if "open_llm_model" in data:
            instance.open_llm_model = data["open_llm_model"]
        if "open_llm_host" in data:
            instance.open_llm_host = data["open_llm_host"]
        if "open_llm_port" in data:
            instance.open_llm_port = data["open_llm_port"]
        
        # Load prompt understanding config
        instance.prompt_understanding = data["prompt_understanding"]
        instance.prompt_understanding["creativity_level"] = CreativityLevel(
            instance.prompt_understanding["creativity_level"]
        )

        # Load regeneration configs
        instance.regeneration_configs = data["regeneration_configs"]
        instance.regeneration_count = data["regeneration_count"]

        return instance

    def save_to_file(self, filename: str):
        """Save current config state to a JSON file."""
        try:
            config_data = self.to_dict()
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2)
            self.logger.debug(f"Successfully saved config to: {filename}")
        except Exception as e:
            self.logger.error(f"Error saving config to {filename}: {str(e)}")

    @classmethod
    def load_from_file(cls, filename: str) -> 'T2IConfig':
        """Load config state from a JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading config from {filename}: {str(e)}")
            raise

# Function to initialize LLMs based on configuration
def initialize_llms(use_open_llm=False, open_llm_model="mistralai/Mistral-Small-3.1-24B-Instruct-2503", local_host="0.0.0.0", local_port="8000"):
    """Initialize the LLM models based on configuration"""
    global llm, llm_json
    
    if use_open_llm:
        openai_api_base = f"http://{local_host}:{local_port}/v1"
        openai_api_key = "eqr3k3jlk21jdlkdmvli23rjnflwejnfikjcn"  # Not actually validated
        
        llm = ChatOpenAI(
            model=open_llm_model,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            temperature=0.15
        )
        
        llm_json = ChatOpenAI(
            model=open_llm_model,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            temperature=0.15,
            response_format={"type": "json_object"}
        )
        
        print(f"Initialized OpenSource LLM: {open_llm_model}")
    else:
        # Using GPT-4o-mini
        llm = ChatOpenAI(model="gpt-4o-mini")
        llm_json = ChatOpenAI(model="gpt-4o-mini", response_format={"type": "json_object"})
        print("Initialized GPT-4o-mini")
    
    return llm, llm_json

class IntentionAnalyzer:
    """Helper class for intention understanding operations"""
    def __init__(self, llm):
        self.llm = llm
        self.llm_json = llm.bind(response_format={"type": "json_object"})
        self.logger = logger

    def identify_image_path(self, prompt: str) -> str:
        from urllib.parse import urlparse

        # Extract possible image filename or URL from the prompt
        match = re.search(r"[\w/.\-]+\.png|[\w/.\-]+\.jpg|https?://[\w/.\-]+", prompt)
        if match:
            path_or_url = match.group()
            parsed_url = urlparse(path_or_url)

            if parsed_url.scheme in ['http', 'https']:
                # It's a URL
                # print(f"Identified URL: {path_or_url}")
                self.logger.debug(f"Identified URL: {path_or_url}")
                return path_or_url, "url"
            else:
                # It's a local file path
                full_path = os.path.abspath(os.path.expanduser(path_or_url))
                if os.path.exists(full_path):
                    return full_path, "local"
                else:
                    # print(f"Image '{full_path}' not found.")
                    self.logger.error(f"Image '{full_path}' not found.")
                    return None, None
        # print("No valid image file or URL found in the prompt.")
        self.logger.debug("No valid image file or URL found in the prompt.")
        return None, None

    def analyze_prompt(self, prompt: str, creativity_level: CreativityLevel) -> Dict[str, Any]:
        """Analyze the prompt and identify elements that need clarification."""
        self.logger.debug(f"Analyzing prompt: '{prompt}' with creativity level: {creativity_level.value}")
        
        # Identify image input by ".png" or ".jpg"
        # NOTE: currently only support one image (the first identified image) in the prompt
        image_dir_in_prompt, image_type = self.identify_image_path(prompt)
        if image_dir_in_prompt:
            self.logger.debug(f"Identifying image from: {image_dir_in_prompt}; Image type: {image_type}")
            if image_type == "url":
                analysis_prompt = [
                                    (
                                        "system",
                                        make_intention_analysis_prompt()
                                    ),
                                    (
                                        "human",
                                        [
                                            {"type": "text", 
                                            "text": f"Analyze this image generation prompt: '{prompt}' with creativity level: {creativity_level.value}"},
                                            {"type": "image_url", 
                                            "image_url": {"url": f"{image_dir_in_prompt}"}}
                                        ]
                                    )
                                ]
                
            elif image_type == "local":
                # read image and convert to base64
                with open(image_dir_in_prompt, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    analysis_prompt = [
                                    (
                                        "system",
                                        make_intention_analysis_prompt()
                                    ),
                                    (
                                        "human",
                                        [
                                            {"type": "text", 
                                            "text": f"Analyze this image generation prompt: '{prompt}' with creativity level: {creativity_level.value}"},
                                            {"type": "image_url", 
                                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                        ]
                                    )
                                ]
        else:
            analysis_prompt = [
                (
                    "system",
                    make_intention_analysis_prompt()
                ),
                (
                    "human",
                    f"Analyze this image generation prompt: '{prompt}' with creativity level: {creativity_level.value}"
                )
            ]
        
        # Get response as string and parse it to dict
        response = track_llm_call(self.llm_json.invoke, "intention_analysis", analysis_prompt)
        self.logger.debug(f"Raw LLM response: {response}")
        
        # response is <class 'langchain_core.messages.ai.AIMessage'>
        # response.content is <class 'str'>

        try:
            if isinstance(response.content, str):
                parsed_response = json.loads(response.content)
            elif isinstance(response.content, dict):
                parsed_response = response.content
            elif isinstance(response.content, json):
                parsed_response = response.content
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
                
            self.logger.debug(f"Parsed response: {json.dumps(parsed_response, indent=2)}")
            return parsed_response
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response: {str(e)}")
            self.logger.error(f"Response was: {response}")
            raise

    def retrieve_reference(self, analysis: Dict[str, Any]):
        """Retrieve refenrece content or style based on the analysis."""

        # get current config
        current_config = config.get_current_config()
        if "references" in analysis["identified_elements"] and analysis["identified_elements"]["references"].get("content"):
            current_config["reference_content_image"] = analysis["identified_elements"]["references"]["content"]
        
        self.logger.debug(f"Retrieved reference content image: {current_config['reference_content_image']}")
        
    def retrieve_questions(self, analysis: Dict[str, Any], creativity_level: CreativityLevel) -> str:
        """Retrieve questions based on the analysis and creativity level."""
        self.logger.debug(f"Generating questions for creativity level: {creativity_level.value}")
        
        if creativity_level == CreativityLevel.HIGH:
            self.logger.debug("High creativity mode - returning AUTOCOMPLETE")
            return "AUTOCOMPLETE"

        questions = []
        
        for ambiguous_element in analysis["ambiguous_elements"]:
            questions.extend(ambiguous_element["suggested_questions"])
        
        if not questions:
            self.logger.debug("No questions needed - returning SUFFICIENT_DETAIL")
            return "SUFFICIENT_DETAIL"
        
        formatted_questions = "\n".join([f"- {q}" for q in questions])
        self.logger.debug(f"Retrieved questions:\n{formatted_questions}")
        return formatted_questions

    def refine_prompt_with_analysis(self,
                                    original_prompt: str, 
                                    analysis: Dict[str, Any], 
                                    user_responses: Optional[Dict[str, str]] = None,
                                    creativity_level: CreativityLevel = CreativityLevel.MEDIUM
                                ) -> Dict[str, Any]:
        """
        Refine the prompt using the analysis and any user responses.
        Also evaluates detail level when user responses are provided.
        
        Returns:
            Dict containing:
            - refined_prompt: str
            - suggested_creativity_level: CreativityLevel (only when user_responses provided)
        """
        self.logger.debug(f"Original prompt: '{original_prompt}'")
        self.logger.debug(f"User responses: {user_responses}")
        self.logger.debug(f"Creativity level: {creativity_level.value}")
        
        if user_responses:
            refinement_prompt = f"""
            Original prompt: "{original_prompt}"
            Analysis: {json.dumps(analysis, indent=2)}
            User responses: {json.dumps(user_responses, indent=2)}
            Current creativity level: {creativity_level.value}
            
            You are a Flux.1-dev prompt expert. Refine the given prompt from user. Focus on aligning the given prompt and generating the best quality image. Please make sure all the mentioned objects are remained. 
            Steps: 
            0. Convert negative statements into positive ones by rephrasing to focus on what should be included, without mentioning what should not be included. Examples:
               * "Do not wear a coat" -> "Wear a light sweater"
               * "No trees in background" -> "Clear blue sky background"
               * "Remove the hat" -> "Show full hair styling"
               * "Not smiling" -> "Serious expression"
               * "No bright colors" -> "Muted, subtle tones"
            1. Incorporating all information: based on the original prompt, i) adding more details from user responses, and ii) if the creativity_level is MEDIUM or HIGH, adding the creative_fill details from analysis
            2. If there is reference image, must keep the its directory
            3. Suggest creativity level based on detail completeness of user responses:
               - LOW: If user provided very specific details for most aspects
               - MEDIUM: If some details are provided but some flexibility is needed
               - HIGH: If many details are still open to interpretation
            
            Return a JSON with:
            {{
                "refined_prompt": "A well-structured, coherent prompt that integrates the original prompt, user responses, and analysis. Ensure it maintains clarity, follows natural language conventions, and effectively conveys the intended request.",
                "suggested_creativity_level": "LOW|MEDIUM|HIGH",
                "reasoning": "Explain why the suggested creativity level was chosen based on the detail completeness of user responses."
            }}
            """
        else:
            refinement_prompt = f"""
            Original prompt: "{original_prompt}"
            Analysis: {json.dumps(analysis, indent=2)}
            Creativity level: {creativity_level.value}
            
            You are a Flux.1-dev prompt expert. Refine the given prompt from user. Focus on aligning the given prompt and generating the best quality image. Please make sure all the mentioned objects are remained. 
            Steps: 
            0. Convert negative statements into positive ones by rephrasing to focus on what should be included, without mentioning what should not be included. Examples:
               * "Do not wear a coat" -> "Wear a light sweater"
               * "No trees in background" -> "Clear blue sky background"
               * "Remove the hat" -> "Show full hair styling"
               * "Not smiling" -> "Serious expression"
               * "No bright colors" -> "Muted, subtle tones"
            1. If creativity_level is MEDIUM or HIGH, fill in missing details creatively 
            2. Maintain the user's original intent while adding clarity, which can be referenced from the creative_fill from analysis
            3. Ensure all ambiguous elements are resolved according to creativity level
            4. If there is reference image, must keep the its directory
            
            Return a JSON with:
            {{
                "refined_prompt": "A well-structured, coherent prompt that integrates the original prompt, user responses, and analysis. Ensure it maintains clarity, follows natural language conventions, and effectively conveys the intended request.",
                "reasoning": "Explain why the refined prompt was created."
            }}
            """
        
        response = track_llm_call(self.llm_json.invoke, "refine_prompt", refinement_prompt)
        self.logger.debug(f"Refinement result: {response}")

        try:
            if isinstance(response.content, str):
                parsed_response = json.loads(response.content)
            elif isinstance(response.content, dict):
                parsed_response = response.content
            elif isinstance(response.content, json):
                parsed_response = response.content
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
                
            self.logger.debug(f"Parsed analysis: {json.dumps(parsed_response, indent=2)}")
            return parsed_response
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response: {str(e)}")
            self.logger.error(f"Response was: {response}")
            raise

def load_models():
    """Pre-load models"""
    global flux_pipe, powerpaint_controller

    print("Loading Flux default model...")
    with torch.cuda.device(1):
        flux_pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")

    print("Loading PowerPaint model...")
    with torch.cuda.device(0):
        # Initialize the PowerPaint controller
        powerpaint_dir = "./models/PowerPaint"
        checkpoint_dir = os.path.join(powerpaint_dir, "checkpoints/ppt-v2-1")
        weight_dtype = torch.float16
        powerpaint_controller = PowerPaintController(weight_dtype, checkpoint_dir, False, "ppt-v2-1")
    

@tool("Flux.1-dev")
def generate_with_flux(prompt: str, seed: int) -> str:
    """
    Given a prompt, Flux.1-dev generates general purpose images with high quality and consistency.
    
    Args:
        prompt: The text prompt describing the image to generate
        
    Returns:
        Path to generated images
    """
    logger.info(f"Executing Flux with prompt: {prompt}; seed {seed}")
    logger.info("Loading Flux default model...")

    try:
        image = flux_pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=28,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed)
        ).images[0]

        # Set output path based on regeneration count
        if config.regeneration_count > 0:
            output_path = os.path.join(config.save_dir, f"{config.image_index}_regen{config.regeneration_count}_FLUX.1-dev.png")
        else:
            output_path = os.path.join(config.save_dir, f"{config.image_index}_FLUX.1-dev.png")
        image.save(output_path)
        

        logger.debug(f"Successfully generated image at: {output_path}\n")
        return output_path
            
    except Exception as e:
        logger.error(f"Error in Flux generation: {str(e)}")
        return f"Error generating image with Flux: {str(e)}"


@tool("PowerPaint")
def generate_with_powerpaint(prompt: str, existing_image_dir: str, mask_image_dir: str = None, task: str = "text-guided", seed: int = 42, guidance_scale: float = 7.5) -> str:
    """
    PowerPaint: High-Quality Versatile Image Inpainting. Can perform text-guided inpainting (adding and object or replacing an existing object) or object removal.
    
    Args:
        prompt: The text prompt describing the desired modifications or object to remove
        existing_image_dir: The path to the existing image to edit
        mask_image_dir: Path to mask image defining regions to edit
        task: The inpainting task type, either "text-guided" or "object-removal"
        seed: Random seed for reproducibility
        guidance_scale: The guidance scale for the PowerPaint model (higher values increase prompt adherence)

    Returns:
        Path to generated image
    """
    global powerpaint_controller
    
    logger.info(f"Executing PowerPaint with prompt: {prompt}; task: {task}; seed: {seed}")
    if mask_image_dir:
        logger.info(f"Mask image directory: {mask_image_dir}")
    
    # Set up output path
    if config.regeneration_count > 0:
        output_path = os.path.join(config.save_dir, f"{config.image_index}_regen{config.regeneration_count}_PowerPaint.png")
    else:
        output_path = os.path.join(config.save_dir, f"{config.image_index}_PowerPaint.png")
    
    try:
        input_img = Image.open(existing_image_dir).convert("RGB")
        mask_img = Image.open(mask_image_dir).convert("RGB")
        input_image = {"image": input_img, "mask": mask_img}

        if task == "text-guided":

            result = powerpaint_controller.predict(
                    input_image,
                    prompt,
                    fitting_degree=1.0,
                    ddim_steps=45,
                    scale=guidance_scale,
                    seed=seed,
                    negative_prompt="",
                    task=task,
                )
        elif task == "object-removal":
            result = powerpaint_controller.predict(
                    input_image,
                    prompt="",
                    fitting_degree=1.0,
                    ddim_steps=45,
                    scale=guidance_scale,
                    seed=seed,
                    negative_prompt="",
                    task=task,
                )

        # Save the result
        result.save(output_path)
        logger.debug(f"Successfully generated image at: {output_path}\n")
        return output_path
    except Exception as e:
        logger.error(f"Error in PowerPaint generation: {str(e)}; return input image path: {existing_image_dir}")
        return existing_image_dir


def get_bbox_from_gpt(image_path: str, prompt: str, unwanted_object: str) -> str:
    """
    Use GPT to determine the bounding box coordinates for a referring object in an image.
    
    Args:
        image_path: Path to the image
        prompt: The text prompt describing what to add/modify
        unwanted_object: The object to be removed (in "text-guided" task)
        
    Returns:
        Bounding box coordinates as a string in format "[x1,y1,x2,y2]" or mask path
    """
    logger.info(f"Requesting bounding box coordinates from GPT for: {prompt}")
    
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Prepare the system prompt based on task type
    system_prompt = """You are an expert computer vision assistant that helps identify precise locations in images.
    Your task is to 
    1. If the prompt intents to add an object, determine the exact suitable bounding box coordinates where a new object should be placed in an image.
    2. If the prompt intents to replace an object, determine the exact bounding box coordinates where an unwanted object should be removed in an image.
    
    The coordinates should be in the format [x1, y1, x2, y2] where:
    - (x1, y1) is the top-left corner of the bounding box
    - (x2, y2) is the bottom-right corner of the bounding box
    - All values should be integers between 0 and 1024 (the image is 1024x1024 pixels)
    - The coordinates should define a reasonable size for the object being added
    
    Analyze the image carefully and determine the most natural and appropriate location for the requested addition.
    Return ONLY the coordinates in the format [x1, y1, x2, y2] without any additional text or explanation."""
    
    user_prompt = f"Evaluate the prompt intention {prompt}. Provide the exact bounding box coordinates where the object should be placed."
    
    # Create the message for GPT-4o-mini using the same format as other calls in the codebase
    try:    
        messages = [
            (
                "system",
                system_prompt
            ),
            (
                "human",
                [
                    {
                        "type": "text", 
                        "text": user_prompt
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            )
        ]
        
        # Use the existing llm instance that's already initialized with GPT-4o-mini
        response = llm.invoke(messages)
        
        # Extract the response content
        bbox_text = response.content.strip()
        logger.debug(f"GPT response for bbox: {bbox_text}")
        
        # Clean up the response to ensure it's in the correct format
        # Remove any non-coordinate text that might be in the response
        import re
        bbox_match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', bbox_text)
        
        if bbox_match:
            x1, y1, x2, y2 = map(int, bbox_match.groups())
            # Ensure coordinates are within bounds and properly ordered
            x1, x2 = min(max(0, x1), 1024), min(max(0, x2), 1024)
            y1, y2 = min(max(0, y1), 1024), min(max(0, y2), 1024)
            
            # Ensure x1 < x2 and y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
                
            bbox_coords = f"[{x1}, {y1}, {x2}, {y2}]"
            logger.info(f"Generated bounding box coordinates: {bbox_coords}")
            return bbox_coords, "bbox"
        else:
            logger.warning(f"Could not extract valid bounding box from GPT response: {bbox_text}")
            # call test_REF.py to get mask
            if unwanted_object:
                sam_mask_path = referring_expression_segmentation(
                                    image_path=image_path,
                                    text_input=unwanted_object,
                                    output_dir=config.save_dir
                                )
            else:
                sam_mask_path = referring_expression_segmentation(
                                    image_path=image_path,
                                    text_input=prompt,
                                    output_dir=config.save_dir
                                )
            
            # Return a default bounding box in the center of the image
            return sam_mask_path, "mask"
            
    except Exception as e:
        logger.error(f"Error getting bounding box from GPT: {str(e)}")
        # call test_REF.py to get mask
        if unwanted_object:
            sam_mask_path = referring_expression_segmentation(
                                image_path=image_path,
                                text_input=unwanted_object,
                                output_dir=config.save_dir
                            )
        else:
            sam_mask_path = referring_expression_segmentation(
                                image_path=image_path,
                                text_input=prompt,
                                output_dir=config.save_dir
                            )
        
        # Return a default bounding box in the center of the image
        return sam_mask_path, "mask"

class ModelSelector:
    """Helper class for model selection and execution"""
    def __init__(self, llm):
        self.llm = llm
        self.llm_json = llm.bind(response_format={"type": "json_object"})
        self.logger = logger
        self.tools = {
            "Flux.1-dev": generate_with_flux,
            "PowerPaint": generate_with_powerpaint,
        }
        self.available_models = {
            "Flux.1-dev": generate_with_flux.func.__doc__,
            "PowerPaint": generate_with_powerpaint.func.__doc__,
        }

    def _create_system_prompt(self) -> str:
        """Create the system prompt with all examples and guidelines."""
        if config.regeneration_count > 0:
            return f"""Select the most suitable model for the given task.
            
            Available models:
            1. Flux.1-dev: {self.available_models['Flux.1-dev']}
            2. PowerPaint: {self.available_models['PowerPaint']}

            Note:
            - Best cases for selecting Flux.1-dev:
                - If the prompt is general, the Flux.1-dev model should be selected.

                - For atmosphere/mood/lighting/style improvements or enhancing visual qualities, select Flux.1-dev and create a generating_prompt that:
                * Integrates specific atmospheric details into the original prompt
                * Uses descriptive language for the desired mood or visual effect
                * Examples:
                    - "make it more dramatic" -> "A dramatic scene with high contrast lighting, deep shadows, and intense atmosphere, showing [original elements]"
                    - "enhance cozy feeling" -> "A warm and intimate setting with soft, golden lighting and comfortable ambiance, featuring [original elements]"
                    - "more professional atmosphere" -> "A polished and sophisticated environment with clean lines and professional lighting, showcasing [original elements]"
                    - "enhance ghostly features" -> "A haunting scene with ethereal, translucent ghostly elements that emit a subtle glow, featuring [original elements]"
                    - "make waves more dramatic" -> "A scene with powerful, dynamic waves with detailed foam and spray, showing [original elements]"
                    - "increase texture detail" -> "A scene with highly detailed surfaces, emphasizing intricate textures and fine details in [original elements]"
                    - "enhance lighting" -> "A scene with dramatic lighting effects, creating bold contrasts and atmospheric illumination for [original elements]"
                
                - For rearrangement improvements (e.g., repositioning elements), select Flux.1-dev and create a generating_prompt that combines the original prompt with the improvement suggestions to ensure proper placement
                    
            - Best cases for selecting PowerPaint:
                - For high-quality inpainting and object removal, select PowerPaint. Best use cases include:
                    * Precise object removal with natural background filling (e.g., "remove the person from the image", "erase the car")
                    * Adding completely new objects to specific regions (e.g., "add a realistic cat to the sofa", "place a vase of flowers on the table")
                    * When adding an object, provide the exact bounding box coordinates of the referring object to be added in the format of "[x1,y1,x2,y2]" while the coordinates for the referred removing object can be None (The given image resolution is 1024x1024)
                    * Seamless texture replacement (e.g., "replace the brick wall with wood paneling", "change the grass to sand")
                    * Complex scene modifications that require natural blending (e.g., "add a window to the wall", "place a door in the hallway")
                    
            - For generating_prompt:
                - Converting negative statements into positive ones by rephrasing to focus on what should be included, without mentioning what should not be included. Examples:
                    * "Do not wear a coat" -> "Wear a light sweater"
                    * "No trees in background" -> "Clear blue sky background"
                    * "Remove the hat" -> "Show full hair styling"
                    * "Not smiling" -> "Serious expression"
                    * "No bright colors" -> "Muted, subtle tones"
                - For ensuring main subjects appear in the image, structure the prompt to:
                    Start with the primary focal element and its key attributes, then build outward with supporting elements, describing their spatial relationships and interactions. Connect elements with clear positional language and natural transitions. For example, instead of:
                    (x) "A vintage armchair. A sleeping cat. A Persian rug. Antique books. Wooden shelves."
                    Write:
                    (v) "A vintage leather armchair dominates the corner, its worn texture catching the ambient light. A cat sleeps peacefully on the Persian rug spread before it, while antique books line the wooden shelves along the wall."

            - The item "task_type" in returned JSON should be "text-guided" or "object-removal" for PowerPaint.
                * When the task is about adding or replacing an object, the "task_type" should be "text-guided". Its full name is "text-guided inpainting".
                * When the task is about removing an object, the "task_type" should be "object-removal".

            - The item "bbox_coordinates" in returned JSON should be the bounding box coordinates of the object to be added in the format of "[x1,y1,x2,y2]" while the coordinates for the referred removing object can be None (The given image resolution is 1024x1024)
            
            - Return a JSON with:
            {{
                "selected_model": "model_name",
                "reference_content_image": "path to the reference content image",
                "generating_prompt": "model-specific prompt",
                "unwanted_object": "the object to be removed",
                "task_type": "text-guided" or "object-removal" for PowerPaint,
                "bbox_coordinates": "bounding box coordinates of the object to be edited in the format of '[x1,y1,x2,y2]'",
                "reasoning": "Detailed explanation of why this model was chosen",
                "confidence_score": float  # 0.0 to 1.0
            }}

            # Example 1 (Enhancing Dramatic Effects):
            - Given prompt: "An ancient dragon perched on a cliff overlooking a stormy ocean."
            - Given improvement: "Make the storm more intense and add more detail to the dragon's scales."
            - Returned JSON:
                {{
                    "selected_model": "Flux.1-dev",
                    "reference_content_image": None,
                    "generating_prompt": "A colossal ancient dragon with highly detailed, weathered scales perches on a jagged cliff, its wings partially extended. Each scale features deep grooves and sharp ridges, reflecting flickers of ambient lightning. The background consists of a stormy ocean with hurricane-force winds and diagonal rain streaks. The sky is filled with dense, turbulent storm clouds, illuminated by intermittent lightning flashes that cast stark highlights on the dragon's body. Towering waves below crash violently against the rocky shore, creating dynamic white foam and spray patterns. The dragon is positioned as the dominant foreground element, with a cinematic contrast between its dark silhouette and the electric blue highlights from the storm.",
                    "unwanted_object": None,
                    "task_type": None,
                    "bbox_coordinates": None,
                    "reasoning": "Enhancing dramatic effects and textures of existing elements is best handled by Flux.1-dev with detailed descriptions in the prompt",
                    "confidence_score": 0.96
                }}

            # Example 2 (Object Removal):
            - Given improvement: "Remove one person from the image"
            - Returned JSON:
                {{
                    "selected_model": "PowerPaint",
                    "reference_content_image": "PATH/TO/GIVEN_IMAGE",
                    "generating_prompt": "Remove one person from the image",
                    "unwanted_object": "one person",
                    "task_type": "object-removal",
                    "bbox_coordinates": "[250, 100, 750, 300]",
                    "reasoning": "The improvement requires a localized object removal, which is best handled by PowerPaint",
                    "confidence_score": 0.98
                }}

            # Example 3 (Object Replacement):
            - Given prompt: "Replace the cat on the left with a rabbit"
            - Returned JSON:
                {{
                    "selected_model": "PowerPaint",
                    "reference_content_image": "PATH/TO/GIVEN_IMAGE",
                    "generating_prompt": "add a rabbit",
                    "unwanted_object": "a cat",
                    "task_type": "text-guided",
                    "bbox_coordinates": "[140, 100, 200, 160]",
                    "reasoning": "The prompt requires replacing a specific object in a localized region, which is ideal for PowerPaint",
                    "confidence_score": 0.95
                }}
                
            # Example 4 (Object Addition):
            - Given prompt: "Add a vase of flowers on the empty table"
            - Returned JSON:
                {{
                    "selected_model": "PowerPaint",
                    "reference_content_image": "PATH/TO/GIVEN_IMAGE",
                    "generating_prompt": "a beautiful vase filled with colorful flowers including roses, tulips, and daisies",
                    "unwanted_object": None,
                    "task_type": "text-guided",
                    "bbox_coordinates": "[320, 250, 420, 400]",
                    "reasoning": "The prompt requires adding a detailed object to a specific empty region, which PowerPaint excels at with its high-quality inpainting capabilities",
                    "confidence_score": 0.97
                }}

            # Example 5 (Enhancing Dramatic Effects):
            - Given prompt: "A vast ocean under a cloudy sky, with waves rolling toward the shore."
            - Given improvement: "Make the waves more dramatic and add more texture to the sea"
            - Returned JSON:
                {{
                    "selected_model": "Flux.1-dev",
                    "reference_content_image": None,
                    "generating_prompt": "A vast and powerful ocean surging under a storm-laden sky, with towering, dynamic waves rolling toward the shore. The sea surface is richly textured with intricate patterns of foam, swirling currents, and sharp, cresting waves. The water reflects deep blues and grays, with highlights of white spray and turbulence adding a sense of movement. The waves interact dynamically, forming peaks and troughs with complex rippling effects, creating an immersive and dramatic seascape.",
                    "unwanted_object": None,
                    "task_type": None,
                    "bbox_coordinates": None,
                    "reasoning": "Enhancing dramatic effects and textures of existing elements is best handled by Flux.1-dev with detailed descriptions in the prompt",
                    "confidence_score": 0.96
                }}

            # Example 6 (Converting Negative to Positive Statements):
            - Given prompt: "A portrait of a woman without makeup, no jewelry, not smiling, with no bright colors in the background"
            - Returned JSON:
                {{
                    "selected_model": "Flux.1-dev",
                    "reference_content_image": None,
                    "generating_prompt": "Professional portrait photography of a woman with natural, bare skin showing realistic texture and subtle imperfections. She gazes directly at the camera with a serene, thoughtful expression. Her hair is styled simply, falling naturally around her shoulders. She wears a plain, solid-colored top in a neutral tone. The background features soft bokeh in muted sage green and warm gray tones. The lighting is soft, diffused studio lighting from the front-left, creating gentle shadows that accentuate her facial structure. The composition follows the rule of thirds with shallow depth of field, shot with a 85mm lens at f/2.8.",
                    "unwanted_object": None,
                    "task_type": None,
                    "bbox_coordinates": None,
                    "reasoning": "This prompt requires generating a portrait with specific stylistic elements. The negative statements have been converted to positive descriptions that achieve the same intent - focusing on natural appearance, minimal styling, neutral tones, and a contemplative expression rather than stating what should not be included.",
                    "confidence_score": 0.95
                }}
            """
        else:
            return f"""Generate the most suitable prompt for the given task for model Flux.1-dev.
            
            # Flux.1-dev: {self.available_models['Flux.1-dev']}

            Note:
            - For atmosphere/mood/lighting/style improvements or enhancing visual qualities, create a generating_prompt that:
                * Integrates specific atmospheric details into the original prompt
                * Uses descriptive language for the desired mood or visual effect
                * Examples:
                    - "make it more dramatic" -> "A dramatic scene with high contrast lighting, deep shadows, and intense atmosphere, showing [original elements]"
                    - "enhance cozy feeling" -> "A warm and intimate setting with soft, golden lighting and comfortable ambiance, featuring [original elements]"
                    - "more professional atmosphere" -> "A polished and sophisticated environment with clean lines and professional lighting, showcasing [original elements]"
                    - "enhance ghostly features" -> "A haunting scene with ethereal, translucent ghostly elements that emit a subtle glow, featuring [original elements]"
                    - "make waves more dramatic" -> "A scene with powerful, dynamic waves with detailed foam and spray, showing [original elements]"
                    - "increase texture detail" -> "A scene with highly detailed surfaces, emphasizing intricate textures and fine details in [original elements]"
                    - "enhance lighting" -> "A scene with dramatic lighting effects, creating bold contrasts and atmospheric illumination for [original elements]"
             
            - For generating_prompt:
                - Converting negative statements into positive ones by rephrasing to focus on what should be included, without mentioning what should not be included. Examples:
                    * "Do not wear a coat" -> "Wear a light sweater"
                    * "No trees in background" -> "Clear blue sky background"
                    * "Remove the hat" -> "Show full hair styling"
                    * "Not smiling" -> "Serious expression"
                    * "No bright colors" -> "Muted, subtle tones"
                - For ensuring main subjects appear in the image, structure the prompt to:
                    Start with the primary focal element and its key attributes, then build outward with supporting elements, describing their spatial relationships and interactions. Connect elements with clear positional language and natural transitions. For example, instead of:
                    (x) "A vintage armchair. A sleeping cat. A Persian rug. Antique books. Wooden shelves."
                    Write:
                    (v) "A vintage leather armchair dominates the corner, its worn texture catching the ambient light. A cat sleeps peacefully on the Persian rug spread before it, while antique books line the wooden shelves along the wall."

            - Return a JSON with:
            {{
                "selected_model": "model_name",
                "reference_content_image": "path to the reference content image",
                "generating_prompt": "model-specific prompt",
                "unwanted_object": None,
                "task_type": None,
                "bbox_coordinates": None,
                "reasoning": "Detailed explanation of why this model was chosen",
                "confidence_score": float  # 0.0 to 1.0
            }}

            # Example 1 (Enhancing Dramatic Effects):
            - Given prompt: "An ancient dragon perched on a cliff overlooking a stormy ocean."
            - Given improvement: "Make the storm more intense and add more detail to the dragon's scales."
            - Returned JSON:
                {{
                    "selected_model": "Flux.1-dev",
                    "reference_content_image": None,
                    "generating_prompt": "A colossal ancient dragon with highly detailed, weathered scales perches on a jagged cliff, its wings partially extended. Each scale features deep grooves and sharp ridges, reflecting flickers of ambient lightning. The background consists of a stormy ocean with hurricane-force winds and diagonal rain streaks. The sky is filled with dense, turbulent storm clouds, illuminated by intermittent lightning flashes that cast stark highlights on the dragon's body. Towering waves below crash violently against the rocky shore, creating dynamic white foam and spray patterns. The dragon is positioned as the dominant foreground element, with a cinematic contrast between its dark silhouette and the electric blue highlights from the storm.",
                    "unwanted_object": None,
                    "task_type": None,
                    "bbox_coordinates": None,
                    "reasoning": "Enhancing dramatic effects and textures of existing elements is best handled by Flux.1-dev with detailed descriptions in the prompt",
                    "confidence_score": 0.96
                }}

            # Example 2 (Detailed Scene Creation):
            - Given prompt: "A medieval fantasy marketplace"
            - Returned JSON:
                {{
                    "selected_model": "Flux.1-dev",
                    "reference_content_image": None,
                    "generating_prompt": "A vibrant medieval marketplace in a fantasy setting, photographed in natural daylight. Wooden stalls with colorful canopies are arranged across a cobblestone plaza. Merchants display practical goods - rolls of fabric in earth tones, handcrafted pottery, and baskets of fresh produce. Several visitors in period-appropriate attire browse the marketplace. Stone and timber buildings with distinctive medieval architecture frame the scene. The atmosphere is lively yet realistic, with soft shadows cast by the midday sun. The image has a balanced composition with the marketplace as the clear focal point.",
                    "unwanted_object": None,
                    "task_type": None,
                    "bbox_coordinates": None,
                    "reasoning": "This prompt requires creating a scene with specific elements while avoiding excessive details that could lead to hallucinations. The description focuses on realistic, historically plausible elements with clear spatial relationships and limited complexity, which helps Flux.1-dev generate a more coherent and accurate image",
                    "confidence_score": 0.98
                }}

            # Example 3 (Converting Negative to Positive Statements):
            - Given prompt: "A portrait of a woman without makeup, no jewelry, not smiling, with no bright colors in the background"
            - Returned JSON:
                {{
                    "selected_model": "Flux.1-dev",
                    "reference_content_image": None,
                    "generating_prompt": "Professional portrait photography of a woman with natural, bare skin showing realistic texture and subtle imperfections. She gazes directly at the camera with a serene, thoughtful expression. Her hair is styled simply, falling naturally around her shoulders. She wears a plain, solid-colored top in a neutral tone. The background features soft bokeh in muted sage green and warm gray tones. The lighting is soft, diffused studio lighting from the front-left, creating gentle shadows that accentuate her facial structure. The composition follows the rule of thirds with shallow depth of field, shot with a 85mm lens at f/2.8.",
                    "unwanted_object": None,
                    "task_type": None,
                    "bbox_coordinates": None,
                    "reasoning": "This prompt requires generating a portrait with specific stylistic elements. The negative statements have been converted to positive descriptions that achieve the same intent - focusing on natural appearance, minimal styling, neutral tones, and a contemplative expression rather than stating what should not be included.",
                    "confidence_score": 0.95
                }}
            """

    def _create_task_prompt(self) -> str:
        """Create the task-specific prompt based on current state."""
        if config.regeneration_count > 0:
            prev_config = config.get_prev_config()
            if prev_config['user_feedback']:
                return f"""Analyze this regeneration request:
                Previous result: {prev_config['gen_image_path']}
                User feedback: {prev_config['user_feedback']}
                Ultimate guiding principle prompt: {config.prompt_understanding['original_prompt']}
                First Round Prompt Understanding: {config.prompt_understanding}"""
            else:
                return f"""Analyze this regeneration request:
                Previous result: {prev_config['gen_image_path']}
                Improvement needed: {prev_config['improvement_suggestions']}
                Ultimate guiding principle prompt: {config.prompt_understanding['original_prompt']}
                First Round Prompt Understanding: {config.prompt_understanding}"""
        else:
            if config.is_human_in_loop:
                return f"""Analyze this initial generation request:
                Original prompt: {config.prompt_understanding['original_prompt']}
                User clarification: {config.prompt_understanding['user_clarification']}
                Prompt understanding: {config.prompt_understanding}"""
            else:
                return f"""Analyze this initial generation request:
                Original prompt: {config.prompt_understanding['original_prompt']}
                Prompt understanding: {config.prompt_understanding}"""

    def select_model(self) -> Dict[str, Any]:
        """Analyze the refined prompt and select the most suitable model."""
        try:
            # Create base system prompt and task-specific prompt
            base_prompt = self._create_system_prompt()
            task_prompt = self._create_task_prompt()

            self.logger.info(f"System Prompt for model selection: {base_prompt}")
            self.logger.info(f"User Prompt for model selection: {task_prompt}")

            # Make the API call with structured prompts
            response = track_llm_call(self.llm_json.invoke, "model_selection", [
                ("system", base_prompt),
                ("human", task_prompt)
            ])

            return self._parse_llm_response(response)

        except Exception as e:
            self.logger.error(f"Error in model selection: {str(e)}. Return basic configuration.")
            # Fallback to Flux.1-dev with basic configuration
            return {
                "selected_model": "Flux.1-dev",
                "reference_content_image": None,
                "generating_prompt": config.prompt_understanding.get('refined_prompt', config.prompt_understanding['original_prompt']),
                "unwanted_object": None,
                "task_type": None,
                "bbox_coordinates": None,
                "reasoning": "Fallback selection due to error in model selection process",
                "confidence_score": 0.5
            }

    def _parse_llm_response(self, response) -> Dict[str, Any]:
        """Parse LLM response to dictionary."""
        try:
            if isinstance(response.content, str):
                return json.loads(response.content)
            elif isinstance(response.content, (dict, json)):
                return response.content
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response: {str(e)}")
            self.logger.error(f"Raw response: {response.content}")
            # Return fallback configuration
            return {
                "selected_model": "Flux.1-dev",
                "reference_content_image": None,
                "generating_prompt": config.prompt_understanding.get('refined_prompt', config.prompt_understanding['original_prompt']),
                "unwanted_object": None,
                "task_type": None,
                "bbox_coordinates": None,
                "reasoning": "Fallback selection due to JSON parsing error",
                "confidence_score": 0.5
            }

def intention_understanding_node(state: MessagesState) -> Command[Literal["model_selection"]]:
    """Process user's initial prompt and handle interactions."""

    analyzer = IntentionAnalyzer(llm)
    last_message = state["messages"][-1]
    
    logger.info("-"*50)
    logger.info("INSIDE INTENTION UNDERSTANDING NODE")
    logger.info(f"Processing message: {last_message.content}")
    logger.info(f"Message count: {len(state["messages"])}")
    logger.info(f"Overall messages: {state["messages"]}")
    logger.info(f"Current config: {config.to_dict()}")

    # First interaction - analyze the prompt
    if len(state["messages"]) == 1:
        config.prompt_understanding["original_prompt"] = last_message.content
        logger.info("Step 1: Analyzing prompt")
        try:
            # Analyze prompt
            analysis = analyzer.analyze_prompt(last_message.content, config.prompt_understanding["creativity_level"])
            config.prompt_understanding["prompt_analysis"] = json.dumps(analysis)
            logger.info(f"Analysis result: {config.prompt_understanding['prompt_analysis']}")
            
            # Retrieve references info
            analyzer.retrieve_reference(analysis)
            logger.info(f"Current config for retrieved reference info:\n {config.to_dict()}")

            # Retrieve questions
            questions = analyzer.retrieve_questions(analysis, config.prompt_understanding["creativity_level"])
            logger.info(f"Suggested questions for users:\n {questions}")
            
            if questions == "SUFFICIENT_DETAIL" or questions == "AUTOCOMPLETE" or not config.is_human_in_loop:
                # Refine prompt directly
                refinement_result = analyzer.refine_prompt_with_analysis(
                    last_message.content,
                    analysis,
                    creativity_level=config.prompt_understanding["creativity_level"]
                )
                config.prompt_understanding["refined_prompt"] = refinement_result['refined_prompt']
                logger.info(f"With SUFFICIENT_DETAIL, AUTOCOMPLETE, or non-human-in-loop mode, Refinement result: {json.dumps(refinement_result, indent=2)}")
                
                command = Command(
                                update={"messages": state["messages"] + [AIMessage(content=f"Refined prompt: {config.prompt_understanding['refined_prompt']}")]},
                                goto="model_selection"
                            )
                logger.debug(f"Command: {command}")

                return command
            else:
                if config.is_human_in_loop:
                    config.prompt_understanding["questions"] = questions
                    logger.info(f"Need for information from users about: {questions}")
                    
                    # Clear the console output to make the prompt more visible
                    if os.name == 'posix':  # For Unix/Linux/MacOS
                        os.system('clear')
                    elif os.name == 'nt':   # For Windows
                        os.system('cls')

                    # Print a very distinctive prompt
                    print("\n" + "="*80)
                    print("USER INPUT REQUIRED - PLEASE RESPOND TO THE FOLLOWING QUESTIONS:")
                    print("="*80)
                    print(questions)
                    print("-"*80)

                    # Flush stdout to ensure prompt is displayed
                    sys.stdout.flush()

                    # Use a more robust input method
                    print("Enter your response below and press Enter when finished:")
                    user_responses = ""
                    try:
                        # This approach makes it clearer we're waiting for input
                        user_responses = input("> ")
                    except EOFError:
                        logger.error("Received EOF while waiting for input")
                        user_responses = "No input provided due to EOF"
                            
                    # Store user responses in config
                    config.prompt_understanding['user_clarification'] = user_responses
                    
                    # Parse user response
                    analysis = json.loads(config.prompt_understanding['prompt_analysis'])
                    
                    # Refine prompt with user input
                    refinement_result = analyzer.refine_prompt_with_analysis(
                        config.prompt_understanding['original_prompt'],
                        analysis,
                        user_responses,
                        config.prompt_understanding["creativity_level"]
                    )
                    config.prompt_understanding["refined_prompt"] = refinement_result['refined_prompt']
                    if 'suggested_creativity_level' in refinement_result:
                        if "LOW" in refinement_result['suggested_creativity_level']:
                            config.prompt_understanding["creativity_level"] = CreativityLevel.LOW
                        elif "MEDIUM" in refinement_result['suggested_creativity_level']:
                            config.prompt_understanding["creativity_level"] = CreativityLevel.MEDIUM
                        elif "HIGH" in refinement_result['suggested_creativity_level']:
                            config.prompt_understanding["creativity_level"] = CreativityLevel.HIGH
                        logger.info(f"Update creativity level to {config.prompt_understanding["creativity_level"]}")
                    
                    logger.info(f"Final refinement result: {json.dumps(refinement_result, indent=2)}")
                    
                    command = Command(
                                    update={"messages": state["messages"] + [AIMessage(content=f" User provides clarification. Refined prompt: {config.prompt_understanding['refined_prompt']}")]},
                                    goto="model_selection"
                                )
                    logger.debug(f"Command: {command}")
                    return command
                else:
                    # If not human_in_the_loop, proceed with auto-refinement
                    refinement_result = analyzer.refine_prompt_with_analysis(
                        last_message.content,
                        analysis,
                        creativity_level=CreativityLevel.HIGH
                    )
                    config.prompt_understanding["refined_prompt"] = refinement_result['refined_prompt']
                    logger.info(f"Auto-refinement result (non-human-in-loop): {json.dumps(refinement_result, indent=2)}")
                    
                    command = Command(
                                    update={"messages": state["messages"] + [AIMessage(content=f"LLM Refined prompt: {config.prompt_understanding['refined_prompt']}")]},
                                    goto="model_selection"
                                )
                    logger.debug(f"Command: {command}")
                return command
                
        except Exception as e:
            logger.error(f"Error in first interaction: {str(e)}")
            raise
    # Unexpected state
    else:
        logger.warning(f"Unexpected state: messages={last_message}")
        command = Command(
                            update={"messages": state["messages"]},
                            goto="model_selection"
                        )
        logger.debug(f"Command: {command}")
        return command

def model_selection_node(state: MessagesState) -> Command[Literal["evaluation", "model_selection"]]:
    """Process model selection and image generation."""
    selector = ModelSelector(llm)
    
    logger.info("-"*50)
    logger.info("INSIDE MODEL SELECTION NODE")
    current_config = config.get_current_config()
    logger.info(f"Current config: {current_config}")
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
                if config.is_human_in_loop:
                    logger.info(f"Human-in-the-loop mode enabled. Requesting mask for {current_config["reference_content_image"]}...")
                    mask_image_path = request_mask(current_config["reference_content_image"])
                else:
                    # For automated mask generation based on task type
                    if current_config["task_type"] == "text-guided":
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
                        elif current_config["bbox_coordinates"] is not None:
                            logger.info(f"Generating mask for text-guided inpainting: '{current_config["generating_prompt"]}'")
                            gpt_mask_path = generate_mask_from_bbox(parse_bbox(current_config["bbox_coordinates"]), current_config["reference_content_image"])
                            if gpt_mask_path:
                                current_config["reference_mask_dir"] = gpt_mask_path
                                logger.info(f"Using bbox-generated mask: {current_config["reference_mask_dir"]}")
                            else:
                                logger.info("Failed to generate mask from bbox.")
                        
                        elif current_config["bbox_coordinates"] is None or current_config["reference_mask_dir"] is None:
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
                            
                    elif current_config["task_type"] == "object-removal" and referring_expression_segmentation is not None:
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
                if config.is_human_in_loop:
                    logger.info(f"Human-in-the-loop mode enabled. Requesting mask for {current_config["reference_content_image"]}...")
                    current_config["reference_mask_dir"] = request_mask(current_config["reference_content_image"])
                else:
                    # For automated mask generation based on task type
                    if current_config["task_type"] == "text-guided":
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
                        elif current_config["bbox_coordinates"] is not None:
                            logger.info(f"Generating mask for text-guided inpainting: '{current_config["generating_prompt"]}'")
                            gpt_mask_path = generate_mask_from_bbox(parse_bbox(current_config["bbox_coordinates"]), current_config["reference_content_image"])
                            if gpt_mask_path:
                                current_config["reference_mask_dir"] = gpt_mask_path
                                logger.info(f"Using bbox-generated mask: {current_config["reference_mask_dir"]}")
                            else:
                                logger.info("Failed to generate mask from bbox.")
                        
                        elif current_config["bbox_coordinates"] is None or current_config["reference_mask_dir"] is None:
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

                    elif current_config["task_type"] == "object-removal" and referring_expression_segmentation is not None:
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

        # Execute the selected model
        gen_image_path = execute_model(
            model_name=current_config['selected_model'],
            prompt=current_config['generating_prompt'],
            task_type=current_config['task_type'],
            mask_dir=current_config['reference_mask_dir'],
            reference_content_image=current_config['reference_content_image'],
        )
        
        current_config["gen_image_path"] = gen_image_path

        command = Command(
            update={"messages": state["messages"] + [
                AIMessage(content=f"Generated images using {current_config['selected_model']}. Image path saved in: {current_config['gen_image_path']}")
            ]},
            goto=END
        )
        logger.debug(f"Command: {command}")
        return command

    except Exception as e:
        logger.error(f"Error in model selection: {str(e)}")

        # Check if this is a regeneration attempt
        if config.regeneration_count > 0:
            # Get previous config and use its generated image path
            prev_config = config.get_prev_config()
            prev_image_path = prev_config["gen_image_path"]
            logger.info(f"Returning previous generated image due to error: {prev_image_path}")
            
            # Update current config with previous image path
            current_config["gen_image_path"] = prev_image_path
            
            return Command(
                update={"messages": state["messages"] + [
                    AIMessage(content=f"Error occurred during regeneration. Using previous generated image: {prev_image_path}")
                ]},
                goto=END
            )
        
        # If not a regeneration, raise the error
        raise

def execute_model(model_name: str, prompt: str, task_type: str, mask_dir: str, reference_content_image: str) -> str:
    """Execute the selected model and return paths to generated images."""
    global flux_pipe, powerpaint_controller
    global model_inference_times
    selector = ModelSelector(llm)
    if model_name not in selector.tools:
        raise ValueError(f"Unknown model: {model_name}")

    # get regen count
    regen_count = config.regeneration_count
    if regen_count == 0:
        seed = config.seed
    else:
        # random seed
        seed = random.randint(0, 1000000)

    # Execute the selected model
    if model_name == "Flux.1-dev":
        t0 = time.time()
        result = selector.tools[model_name].invoke({"prompt": prompt, "seed": seed})
        t1 = time.time()
        model_inference_times["Flux.1-dev"].append(t1 - t0)
        return result
    elif model_name == "PowerPaint":
        
        # Set default guidance scale if not provided
        if task_type == "object-removal":
            guidance_scale = 10
        else:
            guidance_scale = 7.5
        t0 = time.time()
        logger.info(f"exisiting_image_dir: {reference_content_image}")
        logger.info(f"mask_image_dir: {mask_dir}")
        logger.info(f"task_type: {task_type}")
        logger.info(f"Using guidance scale: {guidance_scale}")
            
        result = selector.tools[model_name].invoke({
                                                "prompt": prompt,
                                                "existing_image_dir": reference_content_image,
                                                "mask_image_dir": mask_dir,
                                                "task": task_type,
                                                "seed": seed,
                                                "guidance_scale": guidance_scale,
                                                })
        t1 = time.time()
        model_inference_times["PowerPaint"].append(t1 - t0)
        return result
    else:
        raise ValueError(f"Unknown model: {model_name}")

def evaluation_node(state: MessagesState) -> Command[Literal["model_selection", END]]:
    """Evaluate generated images and handle regeneration if needed."""
    last_message = state["messages"][-1]
    
    logger.info("-"*50)
    logger.info("INSIDE EVALUATION NODE")
    logger.info(f"Current config: {config.to_dict()}")
    
    try:
        current_config = config.get_current_config()
        
        # Prepare evaluation prompt
        with open(current_config['gen_image_path'], "rb") as image_file:
            base64_gen_image = base64.b64encode(image_file.read()).decode("utf-8")
        evaluation_prompt = [
            (
                "system",
                make_gen_image_judge_prompt(config)
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": f"original prompt: {config.prompt_understanding['original_prompt']}\n Prompt analysis: {config.prompt_understanding['prompt_analysis']}\n Prompt used for generating the image: {current_config['generating_prompt']}\n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_gen_image}"
                        }
                    }
                ]
            )
        ]
        
        # Get evaluation from GPT-4
        evaluation_result = track_llm_call(llm_json.invoke, "evaluation", evaluation_prompt)
        evaluation_data = json.loads(evaluation_result.content)

        # Update config with evaluation score
        current_config["evaluation_score"] = evaluation_data["overall_score"]
        current_config["improvement_suggestions"] = evaluation_data["improvement_suggestions"]
        logger.info(f"Evaluation result: {json.dumps(evaluation_data, indent=2)}")
        
        # Define threshold for acceptable quality
        QUALITY_THRESHOLD = 8.0
        MAX_REGENERATION_ATTEMPTS = 3

        # Log the evaluation score and threshold
        logger.debug(f"Evaluation score: {current_config['evaluation_score']}")
        logger.debug(f"Quality threshold: {QUALITY_THRESHOLD}")
        logger.debug(f"Regeneration count: {config.regeneration_count}")
        
        if config.is_human_in_loop:
            
            logger.info("Human-in-loop is enabled, requesting user feedback.")
            print("\nCurrent evaluation score:", current_config['evaluation_score'])
            print("Evaluation feedback:", current_config['improvement_suggestions'])
            # TODO: visualize the image for user to evalute
            
            feedback_type = input("\nWould you like to provide feedback?\n0. I like the image and no need to regenerate\n1. Provide text suggestions\n2. Draw regions to edit with editing instructions\n3. Skip feedback but want regenerate the image\nEnter choice (0-3): ")
            if feedback_type == "0":
                logger.info("User likes the image, no need to regenerate.")
                final_message = (
                    f"Final image generated with score: {current_config['evaluation_score']}\n"
                    f"Detailed feedback: {current_config['improvement_suggestions']}\n"
                    f"User feedback: User likes the image, no need to regenerate."
                )
                comment = Command(
                    update={"messages": state["messages"] + [AIMessage(content=final_message)]},
                    goto=END
                )
                return comment
            elif feedback_type == "1":
                user_suggestion = input("Enter your suggestions: ")
                current_config["user_feedback"] = user_suggestion
            
            elif feedback_type == "2":
                logger.info(f"Human in the loop, asking user for the mask")
                # NOTE: open a canvas locally for user to draw the mask (need to launch a local server for transmitting the image for canvas)
                mask_dir = request_mask(current_config["gen_image_path"])
                current_config["editing_mask"] = mask_dir
                logger.info(f"Mask dir: {current_config['editing_mask']}")
                
                user_suggestion = input("Enter your suggestions: ")
                current_config["user_feedback"] = user_suggestion

            elif feedback_type == "3":
                logger.info("User skip feeback, but want regenerate the image")
                current_config["user_feedback"] = "User skip feeback, but want regenerate the image"
            
            regen_key = config.add_regeneration_config()
            prev_config = config.get_prev_config()
            current_config = config.get_current_config()
            current_config["reference_mask_dir"] = prev_config["editing_mask"]
            
            return Command(
                update={
                    "messages": state["messages"] + [AIMessage(content= f"User suggestions: {current_config['user_feedback']}")],  
                },
                goto="model_selection"
            )

        elif current_config['evaluation_score'] < QUALITY_THRESHOLD and config.regeneration_count < (MAX_REGENERATION_ATTEMPTS-1):
            logger.info("Image quality below threshold, preparing to regenerate.")
            # Increment regeneration counter
            regen_key = config.add_regeneration_config()
            
            # Update state and return to model selection
            logger.info(f"Regenerating image (attempt {config.regeneration_count})")
            return Command(
                update={
                    "messages": state["messages"] + [
                        AIMessage(content=f"Image quality below threshold ({current_config['evaluation_score']} < {QUALITY_THRESHOLD}). Regenerating with feedback.")
                    ]
                },
                goto="model_selection"
            )
        
        # If quality is acceptable or max attempts reached
        final_message = (
            f"Final image generated with score: {current_config['evaluation_score']}\n"
            f"Detailed feedback: {current_config['improvement_suggestions']}"
        )
        
        if config.regeneration_count >= MAX_REGENERATION_ATTEMPTS and not config.is_human_in_loop:
            final_message += f"\nReached maximum regeneration attempts ({MAX_REGENERATION_ATTEMPTS})"
        
        logger.info("Image quality is acceptable or maximum regeneration attempts reached.")
        comment = Command(
            update={"messages": state["messages"] + [AIMessage(content=final_message)]},
            goto=END
        )
        return comment
    
    except Exception as e:
        logger.error(f"Error in evaluation node: {str(e)}")

        # Check if this is a regeneration attempt
        if config.regeneration_count > 0:
            # Get previous config and use its generated image path
            prev_config = config.get_prev_config()
            prev_image_path = prev_config["gen_image_path"]
            logger.info(f"Returning previous generated image due to error: {prev_image_path}")
            
            final_message = (
                f"Error occurred during evaluation. Using previous generated image: {prev_image_path}\n"
                f"Error details: {str(e)}"
            )
            
            return Command(
                update={"messages": state["messages"] + [AIMessage(content=final_message)]},
                goto=END
            )
        
        # If not a regeneration, raise the error
        raise

# Create the workflow
def create_t2i_workflow():
    """Create the T2I workflow with appropriate settings."""
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("intention", intention_understanding_node)
    workflow.add_node("model_selection", model_selection_node)
    workflow.add_node("evaluation", evaluation_node)
    
    workflow.add_edge(START, "intention")
    workflow.add_edge("intention", "model_selection")
    workflow.add_edge("model_selection", "evaluation")
    workflow.add_edge("evaluation", END)
    
    # Compile workflow
    compiled_workflow = workflow.compile()
    
    return compiled_workflow

def run_workflow(workflow: StateGraph, initial_prompt: str):
    """Run the T2I workflow with human interaction."""
    
    # Start workflow with initial state
    result = workflow.invoke(
        {"messages": [HumanMessage(content=initial_prompt)]},
    )
    
    return result

def track_llm_call(llm_func, llm_type, *args, **kwargs):
    global llm_latencies, llm_token_counts
    start = time.time()
    response = llm_func(*args, **kwargs)
    end = time.time()
    latency = end - start
    # Try to get token usage from different possible locations
    prompt_tokens = completion_tokens = total_tokens = 0
    # 1. langchain_openai new ChatOpenAI usage_metadata
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        prompt_tokens = usage.get('input_tokens', 0)
        completion_tokens = usage.get('output_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
    # 2. openai API style
    elif hasattr(response, "usage") and response.usage:
        usage = response.usage
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
    # 3. langchain_core.messages.AIMessage style (sometimes usage is in .additional_kwargs)
    elif hasattr(response, "additional_kwargs") and response.additional_kwargs:
        usage = response.additional_kwargs.get("usage", {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
    # 4. or just 0 if not found
    llm_latencies[llm_type].append(latency)
    llm_token_counts[llm_type].append((prompt_tokens, completion_tokens, total_tokens))
    return response

def main(benchmark_name, human_in_the_loop, model_version, use_open_llm=False, open_llm_model="mistralai/Mistral-Small-3.1-24B-Instruct-2503", open_llm_host="0.0.0.0", open_llm_port="8000", calculate_latency=False):
    """Main CLI entry point."""
    # Declare globals
    global logger, config
    global llm_latencies, llm_token_counts
    global model_inference_times
    global llm, llm_json

    # Initialize LLMs based on open_llm flag
    llm, llm_json = initialize_llms(use_open_llm, open_llm_model=open_llm_model, local_host=open_llm_host, local_port=open_llm_port)

    # Get workflow
    workflow = create_t2i_workflow()

    # Load models after logger and save_dir are set
    load_models()
    print("Models loaded")

    # store benchmark dir into one list
    benchmark_list = []
    benchmark_list.append(os.path.join("eval_benchmark/", benchmark_name))

    for benchmark_dir in benchmark_list:

        # Read prompts from the file
        with open(benchmark_dir, 'r') as file:
            if "DrawBench" in str(benchmark_name):
                if "seed" in str(benchmark_name):
                    lines = [line.strip().split('\t') for line in file]
                    prompts = [line[0] for line in lines]
                    seeds = [int(line[1]) for line in lines]
                    bench_result_folder = 'DrawBench-fixseed'
                else:
                    prompts = [line.strip().split('\t')[0] for line in file]
                    bench_result_folder = 'DrawBench'
                prompt_keys = prompts
            elif "GenAIBenchmark" in str(benchmark_name):
                prompts = json.load(file)
                prompt_keys = list(prompts.keys())
                bench_result_folder = 'GenAIBenchmark-fixseed'
            else:
                lines = [line.strip().split('\t') for line in file]
                prompts = [line[0] for line in lines]
                seeds = [int(line[1]) for line in lines]
                prompt_keys = prompts
                bench_result_folder = os.path.basename(benchmark_dir)

        # Create model type suffix for directory
        model_suffix = model_version
        if use_open_llm:
            # Get model name for the suffix - extract just the model name without org prefix
            model_name = open_llm_model.split('/')[-1]
            model_suffix += f"_open_llm_{model_name}"
        
        if human_in_the_loop:
            save_dir = os.path.join("results", bench_result_folder, f"AgentSys_{model_suffix}_human_in_loop")
        else:
            save_dir = os.path.join("results", bench_result_folder, f"AgentSys_{model_suffix}")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Load progress if exists
        progress_file = os.path.join(save_dir, "progress.json")
        if os.path.exists(progress_file):
            with open(progress_file, "r") as file:
                progress_data = json.load(file)
                total_time = progress_data.get("total_time", 0)
                start_idx = progress_data.get("current_idx", 0)
                inference_times = progress_data.get("inference_times", [])
        else:
            total_time = 0
            start_idx = 0
            inference_times = []

        # Calculate the latency and token counts
        single_turn_times = []
        multi_turn_times = []
        end2end_times = []
        model_inference_times = {"Flux.1-dev": [], "PowerPaint": []}
        llm_latencies = {"intention_analysis": [], "refine_prompt": [], "model_selection": [], "evaluation": []}
        llm_token_counts = {"intention_analysis": [], "refine_prompt": [], "model_selection": [], "evaluation": []}
        single_turn_count = 0
        multi_turn_count = 0
        multi_turn_turns = []
        max_gpu_memories = []
        # resume stats
        stats_file = os.path.join(save_dir, "stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            single_turn_times = stats.get("single_turn_times", single_turn_times)
            multi_turn_times = stats.get("multi_turn_times", multi_turn_times)
            end2end_times = stats.get("end2end_times", end2end_times)
            model_inference_times = stats.get("model_inference_times", model_inference_times)
            llm_latencies = stats.get("llm_latencies", llm_latencies)
            llm_token_counts = stats.get("llm_token_counts", llm_token_counts)
            single_turn_count = stats.get("single_turn_count", single_turn_count)
            multi_turn_count = stats.get("multi_turn_count", multi_turn_count)
            multi_turn_turns = stats.get("multi_turn_turns", multi_turn_turns)
            total_time = stats.get("total_time", total_time)
            inference_times = stats.get("inference_times", inference_times)
            max_gpu_memories = stats.get("max_gpu_memories", max_gpu_memories)
            

        for idx, key in tqdm(enumerate(prompt_keys[start_idx:]), total=len(prompts) - start_idx):
            # Initialize a fresh config for each prompt
            config = T2IConfig(human_in_loop=human_in_the_loop)
            config.save_dir = save_dir
            config.use_open_llm = use_open_llm 
            config.open_llm_model = open_llm_model 
            config.open_llm_host = open_llm_host 
            config.open_llm_port = open_llm_port 
            
            if "DrawBench" in str(benchmark_name):
                text_prompt = key
                if "seed" in str(benchmark_name):
                    config.seed = seeds[idx + start_idx]
                else:
                    config.seed = torch.random.seed()
                config.image_index = f"{(idx+start_idx):03d}"
            elif "GenAIBenchmark" in str(benchmark_name):
                text_prompt = prompts[key]['prompt'] 
                if "seed" in str(benchmark_name):
                    config.seed = prompts[key]["random_seed"]
                else:
                    config.seed = torch.random.seed()
                config.image_index = prompts[key]['id']
            else:
                text_prompt = key
                if "seed" in str(benchmark_name):
                    config.seed = seeds[idx + start_idx]
                else:
                    config.seed = torch.random.seed()
                config.image_index = f"{(idx+start_idx):03d}"
                print(f"Working on Benchmark name: {benchmark_name}")

            # Setup logging for this iteration
            logger = setup_logging(save_dir, filename=f"{config.image_index}.log", console_output=False)
            config.logger = logger

            if idx >= 1: 
                break

            # start timing
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.reset_peak_memory_stats(1)
            start_time = time.time()

            logger.info("\n" + "="*83)
            logger.info(f"New Session Started for index {config.image_index}: {datetime.now()}")
            logger.info("="*50)

            logger.info(f"Starting workflow with prompt: {text_prompt}, seed: {config.seed}")
            result = run_workflow(workflow, text_prompt)

            # Save config state after generation
            config_save_path = os.path.join(save_dir, f"{config.image_index}_config.json")
            config.save_to_file(config_save_path)
            logger.info(f"Saved config state to: {config_save_path}")
            # End timing & record stats
            end_time = time.time()
            inference_time = end_time - start_time
            end2end_times.append(inference_time)
            # Determine single/multi-turn
            if config.regeneration_count == 0:
                single_turn_times.append(inference_time)
                single_turn_count += 1
            else:
                multi_turn_times.append(inference_time)
                multi_turn_count += 1
                multi_turn_turns.append(config.regeneration_count + 1)
            total_time += inference_time
            inference_times.append(inference_time)
            mem0 = torch.cuda.max_memory_allocated(0) / 1024**3
            mem1 = torch.cuda.max_memory_allocated(1) / 1024**3
            max_gpu_memories.append(max(mem0, mem1))
            logger.info(f"Inference time for prompt {key}: {inference_time:.4f} seconds")

            logger.info("Workflow completed")    

            # Save progress
            progress_data = {
                "total_time": total_time,
                "current_idx": idx + start_idx + 1,
                "inference_times": inference_times
            }
            with open(progress_file, "w") as file:
                json.dump(progress_data, file)  
                
            # Save stats.json
            stats = {
                "single_turn_times": single_turn_times,
                "multi_turn_times": multi_turn_times,
                "end2end_times": end2end_times,
                "model_inference_times": model_inference_times,
                "llm_latencies": llm_latencies,
                "llm_token_counts": llm_token_counts,
                "single_turn_count": single_turn_count,
                "multi_turn_count": multi_turn_count,
                "multi_turn_turns": multi_turn_turns,
                "total_time": total_time,
                "inference_times": inference_times,
                "max_gpu_memories": max_gpu_memories
            }
            with open(stats_file, "w") as f:
                json.dump(stats, f)
        
        # Calculate and print average time
        # avg_time = total_time / progress_data["current_idx"]
        # print(f"\nAverage inference time per image: {avg_time:.4f} seconds")
        # print(f"Total time for {progress_data["current_idx"]} images: {total_time:.4f} seconds")
        
        # summary output
        if calculate_latency:
            total_prompts = single_turn_count + multi_turn_count
            print("\n==== Statistics ====")
            print(f"Single-turn avg time: {sum(single_turn_times)/len(single_turn_times) if single_turn_times else 0:.4f} s")
            print(f"Multi-turn avg time: {sum(multi_turn_times)/len(multi_turn_times) if multi_turn_times else 0:.4f} s")
            print(f"End-to-end avg time: {sum(end2end_times)/len(end2end_times) if end2end_times else 0:.4f} s")
            for model in model_inference_times:
                times = model_inference_times[model]
                print(f"{model} avg inference time: {sum(times)/len(times) if times else 0:.4f} s, total: {sum(times):.2f} s")
            for k in llm_latencies:
                if llm_latencies[k]:
                    print(f"LLM {k} avg latency: {sum(llm_latencies[k])/len(llm_latencies[k]):.4f} s")
                else:
                    print(f"LLM {k} avg latency: 0.0000 s (fail to log)")
            for k in llm_token_counts:
                if llm_token_counts[k]:
                    avg_prompt = sum(x[0] for x in llm_token_counts[k]) / len(llm_token_counts[k])
                    avg_completion = sum(x[1] for x in llm_token_counts[k]) / len(llm_token_counts[k])
                    avg_total = sum(x[2] for x in llm_token_counts[k]) / len(llm_token_counts[k])
                    print(f"LLM {k} avg prompt tokens: {avg_prompt:.2f}, completion tokens: {avg_completion:.2f}, total tokens: {avg_total:.2f}")
                else:
                    print(f"LLM {k} avg prompt tokens: 0.00, completion tokens: 0.00, total tokens: 0.00  (fail to log)")
            print(f"Single-turn end count: {single_turn_count} ({single_turn_count/total_prompts*100 if total_prompts else 0:.2f}%)")
            print(f"Multi-turn end count: {multi_turn_count} ({multi_turn_count/total_prompts*100 if total_prompts else 0:.2f}%)")
            if multi_turn_turns:
                print(f"Multi-turn average turns: {sum(multi_turn_turns)/len(multi_turn_turns):.2f}")
            else:
                print("Multi-turn average turns: 0")
            if max_gpu_memories:
                print(f"GPU max memory usage: {max(max_gpu_memories):.2f} GB / 48 GB (L40S)")
            else:
                print(f"GPU max memory usage: 0.00 GB / 48 GB (L40S)  (fail to log)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T2I-Copilot Agent System")
    parser.add_argument('--benchmark_name', default='cool_sample.txt', type=str, help='Path to the benchmark directory: cool_sample.txt, GenAIBenchmark/genai_image_seed.json, ambiguous_seed.txt')
    parser.add_argument('--human_in_the_loop', action='store_true', help='Use human in the loop')
    parser.add_argument('--model_version', default='vRelease', type=str, help= 'Model version')
    parser.add_argument('--use_open_llm', action='store_true', help='Use open source LLM instead of GPT-4o-mini')
    parser.add_argument('--open_llm_model', default='mistralai/Mistral-Small-3.1-24B-Instruct-2503', type=str, 
                        help='Open LLM model to use (mistralai/Mistral-Small-3.1-24B-Instruct-2503, Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct)')
    parser.add_argument('--open_llm_host', default='0.0.0.0', type=str, help='Host address for the open LLM server')
    parser.add_argument('--open_llm_port', default='8000', type=str, help='Port for the open LLM server')
    parser.add_argument('--calculate_latency', action='store_true', help='Calculate and print latency statistics')

    args = parser.parse_args()
    main(args.benchmark_name, args.human_in_the_loop, args.model_version, args.use_open_llm, args.open_llm_model, args.open_llm_host, args.open_llm_port, args.calculate_latency)

