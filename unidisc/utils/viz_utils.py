from PIL import Image, ImageDraw, ImageFont
import textwrap


def create_text_image(
    text,
    desired_width,
    font_path="/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    font_size=20,
    text_color=(0, 0, 0),
    bg_color=(255, 255, 255),
    line_spacing=5
):
    """
    Creates a Pillow Image with the given text wrapped to fit the desired width.

    Parameters:
    - text (str): The text to render on the image.
    - desired_width (int): The width of the image in pixels.
    - font_path (str): Path to the .ttf font file.
    - font_size (int): Size of the font.
    - text_color (tuple): RGB color tuple for the text.
    - bg_color (tuple): RGB color tuple for the background.
    - line_spacing (int): Space between lines in pixels.

    Returns:
    - Image: A Pillow Image object with the rendered text.
    """
    # Load the font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        raise IOError(f"Font file not found at path: {font_path}")

    # Create a dummy image to get drawing context
    dummy_img = Image.new('RGB', (desired_width, 1))
    draw = ImageDraw.Draw(dummy_img)

    # Wrap text based on pixel width
    lines = []
    words = text.split()
    if not words:
        lines = ['']
    else:
        line = words[0]
        for word in words[1:]:
            test_line = f"{line} {word}"
            # Use font.getlength instead of font.getsize
            try:
                line_width = font.getlength(test_line)
            except AttributeError:
                # Fallback for older Pillow versions
                bbox = font.getbbox(test_line)
                line_width = bbox[2] - bbox[0]
            
            if line_width <= desired_width:
                line = test_line
            else:
                lines.append(line)
                line = word
        lines.append(line)

    # Calculate the height required for the text
    ascent, descent = font.getmetrics()
    line_height = ascent + descent + line_spacing
    img_height = line_height * len(lines) + line_spacing

    # Create the final image
    img = Image.new('RGB', (desired_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw each line of text
    y_text = line_spacing
    for line in lines:
        draw.text((0, y_text), line, font=font, fill=text_color)
        y_text += line_height

    return img




"""
import os
import random
from PIL import Image, ImageEnhance
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
from glob import glob

coco_categories = [
    "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "fire hydrant", "cat", "dog", "horse",
    "elephant", "bear", "zebra", "bowl", "banana",
    "pizza", "couch", "bed",
]

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",  # Options: "train", "validation", "test"
    label_types=["detections", "segmentations"],
    classes=coco_categories,
    max_samples=100  # Adjust as needed
)
"""

def extract_objects_coco(dataset, output_dir, mask_ratio_threshold=0.7):
    # extracted_objects_dir = "./objects"
    # os.makedirs(extracted_objects_dir, exist_ok=True)
    # extract_objects_coco(dataset, extracted_objects_dir)
    """
    Extracts objects from the COCO dataset based on segmentation masks.
    
    Parameters:
    - dataset: The FiftyOne dataset to process.
    - output_dir: Directory where extracted objects will be saved.
    - mask_ratio_threshold: Minimum ratio of mask area to bounding box area to include the object.
    """
    for sample in dataset:
        # Load the original image
        try:
            image = Image.open(sample.filepath).convert("RGBA")
        except Exception as e:
            print(f"Error loading image {sample.filepath}: {e}")
            continue

        width, height = image.size

        # Access segmentations
        if not hasattr(sample, 'segmentations') or sample.segmentations is None:
            continue  # Skip samples without segmentations

        segmentations = sample.segmentations.detections

        for idx, segmentation in enumerate(segmentations):
            # Get the mask as a boolean numpy array
            mask_array = segmentation.mask  # Shape: (mask_height, mask_width), dtype: bool

            if mask_array is None:
                continue  # Skip if mask is not available

            # Get the bounding box in absolute pixel coordinates
            bbox = segmentation.bounding_box  # [x_min, y_min, width, height] in relative coords
            x_min = int(bbox[0] * width)
            y_min = int(bbox[1] * height)
            bbox_width = int(bbox[2] * width)
            bbox_height = int(bbox[3] * height)

            # Calculate mask area and bounding box area
            mask_area = np.sum(mask_array)
            bbox_area = bbox_width * bbox_height

            if bbox_area == 0:
                print(f"Bounding box has zero area for sample {sample.id}, detection {idx}. Skipping.")
                continue

            mask_ratio = mask_area / bbox_area

            if mask_ratio < mask_ratio_threshold:
                print(f"Mask ratio {mask_ratio:.2f} below threshold for sample {sample.id}, detection {idx}. Skipping.")
                continue  # Skip masks that don't meet the area ratio threshold

            # Ensure the mask size matches the bounding box size
            mask_height, mask_width = mask_array.shape
            if (mask_width, mask_height) != (bbox_width, bbox_height):
                print(f"Mask size {mask_array.shape} does not match bounding box size {(bbox_height, bbox_width)} for sample {sample.id}. Resizing mask.")
                # Resize the mask to match the bounding box dimensions
                mask_image = Image.fromarray(mask_array.astype(np.uint8) * 255, mode='L')
                mask_image = mask_image.resize((bbox_width, bbox_height), Image.NEAREST)
                mask_array = np.array(mask_image) > 0
            else:
                # Convert boolean mask to uint8
                mask_uint8 = (mask_array * 255).astype(np.uint8)
                # Create a PIL Image from the mask
                mask_image = Image.fromarray(mask_uint8, mode='L')

            # Create a full-sized mask and paste the object mask into it
            full_mask = Image.new("L", (width, height))
            try:
                full_mask.paste(mask_image, (x_min, y_min))
            except ValueError as ve:
                print(f"Error pasting mask for sample {sample.id}, detection {idx}: {ve}")
                continue

            # Create an RGBA image for the object with transparency
            object_image = Image.new("RGBA", (width, height))
            try:
                object_image.paste(image, mask=full_mask)
            except ValueError as ve:
                print(f"Error pasting image with mask for sample {sample.id}, detection {idx}: {ve}")
                continue

            # Calculate absolute bounding box coordinates
            x_max = x_min + bbox_width
            y_max = y_min + bbox_height

            # Ensure coordinates are within image boundaries
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, width)
            y_max = min(y_max, height)

            diff_y = y_max - y_min
            diff_x = x_max - x_min
            if diff_x < 128 or diff_y < 128:
                continue  # Skip if the bounding box is too small

            # Crop the object to its bounding box
            object_crop = object_image.crop((x_min, y_min, x_max, y_max))

            # Optional: Further crop using the mask to tightly bound the object
            cropped_mask = np.array(full_mask)[y_min:y_max, x_min:x_max]
            if not np.any(cropped_mask):
                print(f"Empty mask after cropping for sample {sample.id}, detection {idx}. Skipping.")
                continue  # Skip if the mask is empty after cropping

            # Find the bounding box of the non-zero regions in the cropped mask
            ys, xs = np.where(cropped_mask)
            if len(xs) == 0 or len(ys) == 0:
                print(f"No non-zero pixels found in mask for sample {sample.id}, detection {idx}. Skipping.")
                continue  # Skip if no non-zero pixels

            tight_x_min = xs.min()
            tight_y_min = ys.min()
            tight_x_max = xs.max()
            tight_y_max = ys.max()

            # Further crop the image
            object_crop = object_crop.crop((tight_x_min, tight_y_min, tight_x_max + 1, tight_y_max + 1))

            # Save the object image
            object_class = segmentation.label.replace(" ", "_")
            object_filename = f"{object_class}_{sample.id}_{idx}.png"
            object_filepath = os.path.join(output_dir, object_filename)
            try:
                object_crop.save(object_filepath)
                print(f"Saved object {object_filepath}")
            except Exception as e:
                print(f"Error saving object {object_filepath}: {e}")
                continue

def augment_image_with_random_object_coco(original_image, extracted_objects_dir):
    import os
    import random
    from PIL import Image, ImageEnhance
    import numpy as np
    from glob import glob

    try:
        if isinstance(original_image, str):
            original_image = Image.open(original_image).convert('RGBA')
        elif isinstance(original_image, Image.Image):
            original_image = original_image.convert('RGBA')
        else:
            raise ValueError(f"Unsupported type for original_image: {type(original_image)}")
    except Exception as e:
        print(f"Error loading original image {original_image}: {e}")
        return

    width, height = original_image.size

    # Get a list of extracted object images
    object_image_paths = glob(os.path.join(extracted_objects_dir, '*.png'))

    if not object_image_paths:
        print("No extracted object images found.")
        return

    # Choose a random object image
    object_image_path = random.choice(object_image_paths)
    try:
        object_image = Image.open(object_image_path).convert('RGBA')
    except Exception as e:
        print(f"Error loading object image {object_image_path}: {e}")
        return

    # Resize object image
    obj_width, obj_height = object_image.size

    # Modify the scaling logic to ensure object fits
    max_scale = min(width / obj_width, height / obj_height, 0.8)  # Never scale larger than 70%
    min_scale = min(0.3, max_scale)  # Use either 0.2 or max_scale, whichever is smaller
    
    if max_scale < 0.3:  # If even 20% is too big, try another object
        print("Selected object too large for image, choosing another...")
        return None  # Return None to indicate we need to try again
        
    scaling_factor = random.uniform(min_scale, max_scale)
    new_obj_width = int(obj_width * scaling_factor)
    new_obj_height = int(obj_height * scaling_factor)
    object_image = object_image.resize((new_obj_width, new_obj_height), Image.LANCZOS)

    # Optional: Adjust brightness for better blending
    brightness_factor = random.uniform(0.8, 1.2)  # Slightly darken or brighten
    enhancer = ImageEnhance.Brightness(object_image)
    object_image = enhancer.enhance(brightness_factor)

    # Randomize position
    max_x = width - new_obj_width
    max_y = height - new_obj_height
    position = (random.randint(0, max_x), random.randint(0, max_y))

    # Overlay object image
    composite_image = original_image.copy()
    try:
        composite_image.paste(object_image, position, object_image)
    except ValueError as ve:
        print(f"Error pasting object onto original image: {ve}")
        return

    return composite_image.convert('RGB')

# # Example usage
# original_image_path = '00e36460e7a9adde.jpg'  # Replace with your image path
# output_image_path = 'fixed.png'
# augment_image_with_random_object_coco(original_image_path, extracted_objects_dir, output_image_path)
