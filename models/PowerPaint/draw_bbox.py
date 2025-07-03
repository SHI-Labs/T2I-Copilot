import os
from PIL import Image, ImageDraw

def draw_bbox_on_image(image_path, bbox_coordinates, output_path=None, color=(255, 0, 0), width=3):
    """
    Draw a bounding box on an image and save the result.
    
    Args:
        image_path (str): Path to the input image
        bbox_coordinates (tuple): Tuple of (x1, y1, x2, y2) coordinates
        output_path (str, optional): Path to save the output image. If None, will use original filename with '_bbox' suffix
        color (tuple): RGB color for the bounding box (default: red)
        width (int): Width of the bounding box line in pixels
        
    Returns:
        str: Path to the saved image with bounding box
    """
    try:
        # Validate bbox coordinates
        if len(bbox_coordinates) != 4:
            raise ValueError("bbox_coordinates must be a tuple of (x1, y1, x2, y2)")
            
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Create a drawing context
        draw = ImageDraw.Draw(image)
        
        # Draw the bounding box
        x1, y1, x2, y2 = bbox_coordinates
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        
        # Generate output path if not provided
        if output_path is None:
            filename, ext = os.path.splitext(image_path)
            output_path = f"{filename}_bbox{ext}"
        
        # Save the image with the bounding box
        image.save(output_path)
        print(f"Image with bounding box saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error drawing bounding box: {e}")
        return None

# Example usage:
if __name__ == "__main__":
    # Example parameters
    image_path = "00137_FLUX.1-dev.png"
    bbox_coordinates = (250, 100, 750, 300)  # (x1, y1, x2, y2)
    
    # Draw the bounding box and save the result
    result_path = draw_bbox_on_image(
        image_path=image_path,
        bbox_coordinates=bbox_coordinates,
        color=(255, 0, 0),  # Red color
        width=3  # 3 pixels wide
    )
    
    if result_path:
        print(f"Successfully drew bounding box on {image_path}")
    else:
        print("Failed to draw bounding box")