from pathlib import Path
import random
from tqdm import tqdm
import typer
from datasets import load_dataset
from PIL import Image
import numpy as np
import torch
import transformers

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    output_dir: Path,
    num_pairs: int = 100,
    shard_start: int = 0,
    num_shards: int = 1,
    mask_img: bool = False,
    mask_txt: bool = False,
):
    """
    Generate a dataset of image-caption pairs from the synthetic dataset, optionally
    masking text and/or image content.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-high-quality-captions/resolve/main/data/data-{i:06d}.tar"
    urls = [base_url.format(i=i) for i in range(shard_start, shard_start + num_shards)]
    dataset = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)

    if mask_txt:
        print("Initializing pipeline...")
        pipeline = transformers.pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Pipeline initialized on {pipeline.device}")

    for idx, sample in tqdm(enumerate(dataset)):
        if idx >= num_pairs:
            break

        pair_dir = output_dir / f"pair_{idx:06d}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        img = sample['jpg'].convert('RGB')
        if img.width != img.height:
            min_dim = min(img.width, img.height)
            left = (img.width - min_dim) // 2
            top = (img.height - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))
            print(f"Cropped image from {img.width}x{img.height} to {min_dim}x{min_dim}")
            
        caption = sample['json']['short_caption']
        original_caption = caption

        if mask_txt:
            mask_percent = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90])
            messages = [
                {"role": "system", "content": f"You are a helpful assistant that masks important parts of captions. Respond only with the caption where important parts are replaced with <m>. Each <m> is a single token so use multiple <m> tokens to mask more text. Keep the masking natural and meaningful. For example, if the caption is 'A man in a red shirt is playing a guitar in a park', you might output 'A <m> in <m><m><m> is playing a <m> in a park'. Please mask approximately {mask_percent}% of the caption."},
                {"role": "user", "content": f"Mask important parts of this caption: {caption}"}
            ]
            masked_caption = pipeline(messages, max_new_tokens=200, pad_token_id=pipeline.tokenizer.eos_token_id)
            caption = masked_caption[0]["generated_text"][-1]["content"].strip().removeprefix("'").removesuffix("'").removeprefix('"').removesuffix('"')
            
            if "<m>" not in caption:
                words = original_caption.split()
                if len(words) > 1:
                    # Choose random start & end in range of words
                    start_idx = random.randint(0, len(words) - 1)
                    end_idx = random.randint(start_idx, len(words) - 1)
                    # Replace consecutive chunk with <m>
                    for i in range(start_idx, end_idx + 1):
                        words[i] = "<m>"
                    caption = "".join(words)

        if mask_img:
            arr = np.zeros((img.height, img.width), dtype=np.bool_)
            height, width = arr.shape[:2]

            # Pick a random rectangle
            rect_w = random.randint(max(1, width // 5), min(width * 9 // 10, width))
            rect_h = random.randint(max(1, height // 5), min(height * 9 // 10, height))
            start_x = random.randint(0, width - rect_w)
            start_y = random.randint(0, height - rect_h)

            arr[start_y:start_y + rect_h, start_x:start_x + rect_w] = True

            # Convert array back to PIL image
            mask_img = Image.fromarray(arr)
            mask_img.save(pair_dir / "mask.png")

        img.save(pair_dir / "image.jpg")
        (pair_dir / "caption.txt").write_text(original_caption)
        (pair_dir / "mask_caption.txt").write_text(caption)

        if (idx + 1) % 10 == 0:
            print(f"Saved {idx + 1} pairs...")

    print(f"Successfully saved {num_pairs} image-caption pairs to {output_dir}")

if __name__ == "__main__":
    app()
