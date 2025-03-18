from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap

def custom_sort_key(name):
    if "GT_" in name:
        return (0, name)
    elif "seq256" in name:
        return (1, name)
    elif "seq1024" in name:
        return (2, name)
    elif "seq4096" in name:
        return (3, name)
    else:
        return (4, name)

def visualize_datasets(root_dir, img_resolution=256, text_img_width=100, text_wrap_width=10, selected_datasets=None):
    root_path = Path(root_dir)
    for folder_path in root_path.iterdir():
        if folder_path.is_dir():
            datasets = {}
            for dataset_path in folder_path.iterdir():
                if dataset_path.is_dir():
                    if dataset_path.name == "output": continue
                    if selected_datasets and not any(x in dataset_path.name for x in selected_datasets):
                        continue
                    images = []
                    image_paths = [p for p in dataset_path.iterdir() if p.stem.isdigit()]
                    for image_path in sorted(image_paths, key=lambda x: int(x.stem)):
                        images.append(Image.open(image_path))
                    datasets[dataset_path.name] = images

            num_images = len(images)

            viz_per_image = False
            if viz_per_image:
                for index in range(num_images):
                    widths = [img.width for img in images]
                    max_width = max(widths)

                    # Create per_index image
                    per_index_heights = [images[index].resize((img_resolution, img_resolution), Image.LANCZOS).height for images in datasets.values() if len(images) > index]
                    per_index_total_height = sum(per_index_heights)
                    per_index_image = Image.new('RGB', (img_resolution + text_img_width, per_index_total_height))  # Set width to img_resolution + space for text
                    y_offset = 0
                    for dataset_name, images in sorted(datasets.items(), key=lambda x: custom_sort_key(x[0])):
                        if len(images) > index:
                            img = images[index].resize((img_resolution, img_resolution), Image.LANCZOS)  # Resize image to img_resolution x img_resolution
                            text_img = Image.new('RGB', (text_img_width, img_resolution), (255, 255, 255))  # Create a white image for text
                            draw = ImageDraw.Draw(text_img)
                            font = ImageFont.load_default()
                            wrapped_text = textwrap.fill(dataset_name, width=text_wrap_width)  # Wrap text to fit within the image
                            draw.text((10, 10), wrapped_text, fill=(0, 0, 0), font=font)
                            combined_img = Image.new('RGB', (img_resolution + text_img_width, img_resolution))  # Combined width of text and image
                            combined_img.paste(text_img, (0, 0))
                            combined_img.paste(img, (text_img_width, 0))
                            per_index_image.paste(combined_img, (0, y_offset))
                            y_offset += img.height

                    (folder_path / "output").mkdir(parents=True, exist_ok=True)
                    per_index_image.save(folder_path / "output" / f'{index}_per_index_viz.png')

            # Create combined image for the entire dataset
            num_datasets = len(datasets)
            combined_image_width = (img_resolution + text_img_width) * num_images  # Each column is an index + space for text
            combined_image_height = img_resolution * num_datasets  # Each row is a dataset
            combined_image = Image.new('RGB', (combined_image_width, combined_image_height))

            for row_index, (dataset_name, images) in enumerate(sorted(datasets.items(), key=lambda x: custom_sort_key(x[0]))):
                for col_index, img in enumerate(images):
                    resized_img = img.resize((img_resolution, img_resolution), Image.LANCZOS)
                    text_img = Image.new('RGB', (text_img_width, img_resolution), (255, 255, 255))  # Create a white image for text
                    draw = ImageDraw.Draw(text_img)
                    font = ImageFont.load_default()
                    wrapped_text = textwrap.fill(dataset_name, width=text_wrap_width)  # Wrap text to fit within the image
                    draw.text((10, 10), wrapped_text, fill=(0, 0, 0), font=font)
                    combined_img = Image.new('RGB', (img_resolution + text_img_width, img_resolution))  # Combined width of text and image
                    combined_img.paste(text_img, (0, 0))
                    combined_img.paste(resized_img, (text_img_width, 0))
                    x_offset = col_index * (img_resolution + text_img_width)
                    y_offset = row_index * img_resolution
                    combined_image.paste(combined_img, (x_offset, y_offset))
                    

            (folder_path / "output").mkdir(parents=True, exist_ok=True)
            combined_image.save(folder_path / "output" / f'combined_viz_{img_resolution}.png')

visualize_datasets('output', img_resolution=256, text_img_width=100, text_wrap_width=10, selected_datasets=["GT_256", 'titok128', 'titok256', 'cosmos'])