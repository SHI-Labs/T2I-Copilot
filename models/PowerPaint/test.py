import argparse
import os
import random
import json
import requests
import base64

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_model
from transformers import CLIPTextModel

from diffusers import UniPCMultistepScheduler
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from powerpaint.models.BrushNet_CA import BrushNetModel
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline

from powerpaint.utils.utils import TokenizerWrapper, add_tokens

# Import the request_mask function from mask_draw_client
outer_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(outer_dir)
from mask_draw_client import request_mask
from Grounded_SAM2.test_REF import referring_expression_segmentation

# # Print the current working directory
# print(f"Current working directory: {os.getcwd()}")

# # Print the directory of the current file
# print(f"Directory of current file: {os.path.dirname(os.path.abspath(__file__))}")

# # Print all paths in sys.path (where Python looks for imports)
# print("\nPaths available for imports (sys.path):")
# for i, path in enumerate(sys.path):
#     print(f"{i}: {path}")

# exit()
torch.set_grad_enabled(False)

def parse_bbox(bbox_str):
    """
    Parse a string representation of a bounding box.
    Accepts formats like: 
    - "(x1,y1,x2,y2)" (tuple)
    - "[x1,y1,x2,y2]" (list)
    - "x1,y1,x2,y2" (comma-separated values)
    """
    # Remove parentheses or brackets if present
    bbox_str = bbox_str.strip()
    if (bbox_str.startswith('(') and bbox_str.endswith(')')) or \
       (bbox_str.startswith('[') and bbox_str.endswith(']')):
        bbox_str = bbox_str[1:-1]
    
    # Split by comma and convert to integers
    try:
        # Handle spaces after commas
        coords = [int(x.strip()) for x in bbox_str.split(',')]
        if len(coords) != 4:
            raise ValueError("Bounding box must contain exactly 4 values")
        return tuple(coords)  # Return as tuple for consistency
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid bounding box format: {e}")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_task(prompt, negative_prompt, control_type, version):
    pos_prefix = neg_prefix = ""
    if control_type == "object-removal" or control_type == "image-outpainting":
        if version == "ppt-v1":
            pos_prefix = "empty scene blur " + prompt
            neg_prefix = negative_prompt
        promptA = pos_prefix + " P_ctxt"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + " P_obj"
        negative_promptB = neg_prefix + " P_obj"
    elif control_type == "shape-guided":
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_shape"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + "P_shape"
        negative_promptB = neg_prefix + "P_ctxt"
    else:
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_obj"
        promptB = pos_prefix + " P_obj"
        negative_promptA = neg_prefix + "P_obj"
        negative_promptB = neg_prefix + "P_obj"

    return promptA, promptB, negative_promptA, negative_promptB


def generate_mask_from_bbox(bbox_coordinates, input_image_path):
    """
    Generate a mask image based on bounding box coordinates.
    
    Args:
        bbox_coordinates (tuple): Tuple of (x1, y1, x2, y2) coordinates
        output_path (str, optional): Path to save the mask image. If None, a temporary file is created.
        
    Returns:
        str: Path to the generated mask image
    """
    try:
        # Validate bbox coordinates
        if len(bbox_coordinates) != 4:
            print("Error: bbox_coordinates must be a tuple of (x1, y1, x2, y2)")
            return None
            
        x1, y1, x2, y2 = bbox_coordinates
        
        # Ensure coordinates are in the correct order (x1 < x2, y1 < y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Create a black image (mask background)
        width, height = max(x2, 1024), max(y2, 1024)  # Ensure minimum size
        mask = Image.new('RGB', (width, height), (0, 0, 0))
        
        # Create a drawing context
        draw = ImageDraw.Draw(mask)
        
        # Draw a white rectangle for the mask area
        draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
        
        # Save the mask
        output_path = input_image_path.replace(".png", "_bbox_mask.png")
        # if output_path is None:
        #     # Create a temporary file
        #     import tempfile
        #     fd, output_path = tempfile.mkstemp(suffix='.png')
        #     os.close(fd)
        
        mask.save(output_path)
        print(f"Mask generated and saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error generating mask from bbox: {e}")
        return None

def dilate_mask(mask_path, kernel_size=100):
    """
    Load a mask image, expand it using dilation to make the boundary unavailable,
    and save the result with "_dilate.png" appended to the original filename.
    
    Args:
        mask_path (str): Path to the input mask image
        kernel_size (int): Size of the dilation kernel (larger = more expansion)
        
    Returns:
        str: Path to the saved dilated mask
    """
    try:
        # Get the output path
        filename, ext = os.path.splitext(mask_path)
        output_path = f"{filename}_dilate{ext}"
        
        # Load the mask image
        if mask_path.endswith(('.jpg', '.jpeg', '.png')):
            # Load with OpenCV for easier processing
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Try with PIL if OpenCV fails
                mask_pil = Image.open(mask_path).convert('L')
                mask = np.array(mask_pil)
        else:
            print(f"Unsupported file format: {mask_path}")
            return None
        
        # Ensure mask is binary (0 or 255)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Create a kernel for dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Dilate the mask to expand it
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        
        # Save the dilated mask
        cv2.imwrite(output_path, dilated_mask)
        
        print(f"Expanded mask saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error expanding mask: {e}")
        return None

class PowerPaintController:
    def __init__(self, weight_dtype, checkpoint_dir, local_files_only, version) -> None:
        self.version = version
        self.checkpoint_dir = checkpoint_dir
        self.local_files_only = local_files_only

        # brushnet-based version
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            revision=None,
            torch_dtype=weight_dtype,
            local_files_only=local_files_only,
        )
        text_encoder_brushnet = CLIPTextModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="text_encoder",
            revision=None,
            torch_dtype=weight_dtype,
            local_files_only=local_files_only,
        )
        brushnet = BrushNetModel.from_unet(unet)
        base_model_path = os.path.join(checkpoint_dir, "realisticVisionV60B1_v51VAE")
        self.pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
            base_model_path,
            brushnet=brushnet,
            text_encoder_brushnet=text_encoder_brushnet,
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=False,
            safety_checker=None,
        )
        self.pipe.unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
            revision=None,
            torch_dtype=weight_dtype,
            local_files_only=local_files_only,
        )
        self.pipe.tokenizer = TokenizerWrapper(
            from_pretrained=base_model_path,
            subfolder="tokenizer",
            revision=None,
            torch_type=weight_dtype,
            local_files_only=local_files_only,
        )

        # add learned task tokens into the tokenizer
        add_tokens(
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder_brushnet,
            placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
            initialize_tokens=["a", "a", "a"],
            num_vectors_per_token=10,
        )
        load_model(
            self.pipe.brushnet,
            os.path.join(checkpoint_dir, "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"),
        )

        self.pipe.text_encoder_brushnet.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "PowerPaint_Brushnet/pytorch_model.bin")), strict=False
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        self.pipe.enable_model_cpu_offload()
        self.pipe = self.pipe.to("cuda")

    def get_depth_map(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    def predict(
        self,
        input_image,
        prompt,
        fitting_degree,
        ddim_steps,
        scale,
        seed,
        negative_prompt,
        task,
    ):
        size1, size2 = input_image["image"].convert("RGB").size

        if task != "image-outpainting":
            if size1 < size2:
                input_image["image"] = input_image["image"].convert("RGB").resize((640, int(size2 / size1 * 640)))
            else:
                input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 640), 640))
        else:
            if size1 < size2:
                input_image["image"] = input_image["image"].convert("RGB").resize((512, int(size2 / size1 * 512)))
            else:
                input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 512), 512))


        if self.version != "ppt-v1":
            if task == "image-outpainting":
                prompt = prompt + " empty scene"
            if task == "object-removal":
                prompt = prompt + " empty scene blur"
        promptA, promptB, negative_promptA, negative_promptB = add_task(prompt, negative_prompt, task, self.version)
        print(promptA, promptB, negative_promptA, negative_promptB)

        img = np.array(input_image["image"].convert("RGB"))
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image["image"] = input_image["image"].resize((H, W))
        input_image["mask"] = input_image["mask"].resize((H, W))
        set_seed(seed)

        # for brushnet-based method
        np_inpimg = np.array(input_image["image"])
        np_inmask = np.array(input_image["mask"]) / 255.0
        np_inpimg = np_inpimg * (1 - np_inmask)
        input_image["image"] = Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")
        result = self.pipe(
            promptA=promptA,
            promptB=promptB,
            promptU=prompt,
            tradoff=fitting_degree,
            tradoff_nag=fitting_degree,
            image=input_image["image"].convert("RGB"),
            mask=input_image["mask"].convert("RGB"),
            num_inference_steps=ddim_steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            brushnet_conditioning_scale=1.0,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            negative_promptU=negative_prompt,
            guidance_scale=scale,
            width=H,
            height=W,
        ).images[0]

        return result

    def process_image(
        self,
        input_image_path,
        mask_image_path,
        output_path,
        prompt,
        negative_prompt="",
        task="text-guided",
        fitting_degree=1.0,
        ddim_steps=45,
        scale=7.5,
        seed=42,
        human_in_the_loop=False,
        bbox_coordinates=None,
    ):
        # Load input image
        input_img = Image.open(input_image_path).convert("RGB")
        
        # Handle mask based on task type and human-in-the-loop mode
        if human_in_the_loop:
            if mask_image_path is None:
                print(f"Human-in-the-loop mode enabled. Requesting mask for {input_image_path}...")
                mask_image_path = request_mask(input_image_path)
                if not mask_image_path:
                    print("Error: Failed to get mask from human. Using provided mask instead.")
            else:
                print(f"Using provided mask: {mask_image_path}")
        else:
            if mask_image_path is None:
                # For automated mask generation based on task type
                if task == "text-guided" and bbox_coordinates is not None:
                    print(f"Generating mask for text-guided inpainting: '{prompt}'")
                    gpt_mask_path = generate_mask_from_bbox(bbox_coordinates, input_image_path)
                    if gpt_mask_path:
                        mask_image_path = gpt_mask_path
                        print(f"Using bbox-generated mask: {mask_image_path}")
                    else:
                        print("Failed to generate mask from bbox. Using provided mask instead.")
                elif task == "object-removal" and referring_expression_segmentation is not None:
                    print(f"set guidance scale to 10 (suggesting to be 10 or above for removing task)")
                    scale = 10
                    print(f"Generating mask for object removal inpainting: '{prompt}'")
                    try:
                        # Call the referring_expression_segmentation function
                        sam_mask_path = referring_expression_segmentation(
                            image_path=input_image_path,
                            text_input=prompt
                        )
                        # expand the mask to make the mask boundary unavailiable
                        print(f"Expanding mask: {sam_mask_path}")
                        sam_mask_path = dilate_mask(sam_mask_path)
                        if sam_mask_path and os.path.exists(sam_mask_path):
                            mask_image_path = sam_mask_path
                            print(f"Using SAM-generated mask: {mask_image_path}")
                        else:
                            print("Failed to generate mask with SAM. Using provided mask instead.")
                    except Exception as e:
                        print(f"Error generating mask with SAM: {e}")
                        print("Using provided mask instead.")
            else:
                print(f"Using provided mask: {mask_image_path}")
        
        # Load mask image
        mask_img = Image.open(mask_image_path).convert("RGB")
        
        input_image = {"image": input_img, "mask": mask_img}
        
        # Process based on task type
        result = self.predict(
            input_image,
            prompt,
            fitting_degree,
            ddim_steps,
            scale,
            seed,
            negative_prompt,
            task,
        )
        
        # Save the result
        result.save(output_path)
        print(f"Result saved to {output_path}")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PowerPaint: High-Quality Versatile Image Inpainting")
    
    # Basic configuration
    parser.add_argument("--weight_dtype", type=str, default="float16", choices=["float16", "float32"], 
                        help="Weight data type (float16 or float32)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/ppt-v2-1",
                        help="Directory containing model checkpoints")
    parser.add_argument("--version", type=str, default="ppt-v2-1", choices=["ppt-v1", "brushnet", "ppt-v2-1"],
                        help="PowerPaint version to use")
    parser.add_argument("--local_files_only", action="store_true",
                        help="Use cached files without requesting from the hub")
    
    # Input/output paths
    parser.add_argument("--input_image", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--mask_image", type=str, required=False,
                        help="Path to the mask image (white areas will be inpainted)")
    parser.add_argument("--output_path", type=str, default="output.png",
                        help="Path to save the output image")
    
    # Task selection
    parser.add_argument("--task", type=str, default="text-guided", 
                        choices=["text-guided", "object-removal"],#, "shape-guided", "image-outpainting"],
                        help="Inpainting task type")
    
    # Common parameters
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Negative text prompt for generation")
    parser.add_argument("--fitting_degree", type=float, default=1.0,
                        help="Fitting degree (0-1)")
    parser.add_argument("--steps", type=int, default=45,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Bounding box coordinates for text-guided inpainting
    parser.add_argument("--bbox_coordinates", type=parse_bbox, default=None,
                        help="Bounding box coordinates as (x1,y1,x2,y2) or [x1,y1,x2,y2] for text-guided inpainting")
    
    # Human-in-the-loop mode
    parser.add_argument("--human_in_the_loop", action="store_true",
                        help="Enable human-in-the-loop mode to request mask drawing from a human")
    
    args = parser.parse_args()
    
    
    # Validate new task types
    if args.task == "text-guided" and not args.prompt:
        parser.error("For text-guided inpainting task, --prompt must be provided to describe the object addition")
    
    # Check if referring_expression_segmentation is available for object removal inpainting
    if args.task == "object-removal" and not args.human_in_the_loop and referring_expression_segmentation is None:
        print("Warning: referring_expression_segmentation function is not available. Using provided mask instead.")
    
    # Initialize the pipeline controller
    weight_dtype = torch.float16 if args.weight_dtype == "float16" else torch.float32
    controller = PowerPaintController(weight_dtype, args.checkpoint_dir, args.local_files_only, args.version)
    
    # Process the image
    controller.process_image(
        input_image_path=args.input_image,
        mask_image_path=args.mask_image,
        output_path=args.output_path,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        task=args.task,
        fitting_degree=args.fitting_degree,
        ddim_steps=args.steps,
        scale=args.guidance_scale,
        seed=args.seed,
        human_in_the_loop=args.human_in_the_loop,
        bbox_coordinates=args.bbox_coordinates,
    )
