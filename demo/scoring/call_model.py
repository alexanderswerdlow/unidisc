import base64
import io
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import typer
from PIL import Image
from tqdm import tqdm
from image_utils import Im

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

def square_crop(image: Image.Image) -> Image.Image:
    """Crop the image to a square (centered)."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side
    return image.crop((left, top, right, bottom))

def process(image: Image.Image, desired_resolution: int = 512) -> Image.Image:
    """Square-crop and resize the image."""
    cropped_image = square_crop(image.convert("RGB"))
    return cropped_image.resize((desired_resolution, desired_resolution), Image.LANCZOS)

def encode_image(file: Path | io.BytesIO | Image.Image) -> dict:
    """Encode an image as base64 data in a dict of the form {'url': 'data:image/jpeg;base64,...'}."""
    if isinstance(file, Image.Image):
        buffered = io.BytesIO()
        file.save(buffered, format="JPEG")
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    elif isinstance(file, Path):
        with file.open("rb") as img_file:
            base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    else:
        base64_str = base64.b64encode(file.getvalue()).decode("utf-8")
    return {"url": f"data:image/jpeg;base64,{base64_str}"}

def encode_array_image(array: np.ndarray) -> dict:
    """Encode a mask array as base64 data in a dict of the form {'url': 'data:image/jpeg;base64,...'}."""
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    im = Image.fromarray(array)
    buffered = io.BytesIO()
    im.save(buffered, format="JPEG", quality=95)
    base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"url": f"data:image/jpeg;base64,{base64_str}"}

def call_unidisc_api(
    image_path: Path | None,
    caption: str | None,
    mask_path: Path | None,
    cfg: dict,
) -> list:
    """
    Build the payload and call the UniDisc API, returning a list of
    output pieces. Each piece is a dict with either:
      {"type": "text", "text": "..."}
    or {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    """
    # Prepare message content as in reference code
    messages = []
    if caption:
        messages.append({"type": "text", "text": caption})

    if image_path and image_path.exists():
        resolution = int(cfg.get("resolution", 512))
        current_image = process(Image.open(image_path), resolution)
        img_data = encode_image(current_image)["url"]
        messages.append({
            "type": "image_url",
            "image_url": {"url": img_data},
            "is_mask": False
        })

        if mask_path and mask_path.exists():
            mask_array = np.array(Image.open(mask_path))
            mask_data_url = encode_array_image(mask_array)["url"]
            messages.append({
                "type": "image_url",
                "image_url": {"url": mask_data_url},
                "is_mask": True
            })

    config_payload = {
        "max_tokens": int(cfg.get("max_tokens", 32)),
        "resolution": int(cfg.get("resolution", 512)),
        "sampling_steps": int(cfg.get("sampling_steps", 32)),
        "top_p": float(cfg.get("top_p", 0.95)),
        "temperature": float(cfg.get("temperature", 0.9)),
        "maskgit_r_temp": float(cfg.get("maskgit_r_temp", 4.5)),
        "cfg": float(cfg.get("cfg", 2.5)),
        "sampler": cfg.get("sampler", "maskgit_nucleus"),
        "use_reward_models": bool(cfg.get("use_reward_models", False)),
    }

    port = cfg.get('port', 8001)
    hostname = f"{port}" if ":" in port else f"localhost:{port}"

    payload = {
        "messages": [{"role": "user", "content": messages}],
        "model": "unidisc",
        **config_payload
    }

    api_url = f"http://{hostname}/v1/chat/completions"
    response = requests.post(api_url, json=payload)
    if response.status_code != 200:
        return [{"type": "text", "text": f"API Error: {response.text}", "error": True}]

    response_json = response.json()
    if "choices" not in response_json:
        return [{"type": "text", "text": f"Malformed response: {response.text}", "error": True}]

    # The reference code expects "content" to be a list with items typed "text" or "image_url"
    content = response_json["choices"][0]["message"]["content"]
    if isinstance(content, list):
        return content
    else:
        # If it's not a list, wrap it
        return [{"type": "text", "text": content}]

def decode_image_base64(url_str: str) -> Image.Image:
    """Given a 'data:image/...;base64,...' string, return the PIL.Image."""
    # e.g. "data:image/jpeg;base64,xxxx..."
    base64_part = url_str.split("base64,")[-1]
    raw = base64.b64decode(base64_part)
    return Image.open(io.BytesIO(raw))

def run_inference_for_folder(
    folder: Path,
    output_folder: Path,
    cfg: dict,
    use_image: bool,
    use_img_mask: bool,
    use_caption: bool,
    use_cap_mask: bool,
):
    """
    For a single folder with an image, caption, and mask, call the API,
    then write out the returned content (images/text).
    """

    image_file = None
    caption_file = None
    mask_file = None
    for f in folder.iterdir():
        name_lower = f.name.lower()
        if name_lower.startswith("image") and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image_file = f
        if name_lower.startswith("mask") and f.suffix.lower() == ".png":
            mask_file = f
        if name_lower.startswith("caption") and f.suffix.lower() in [".txt"]:
            caption_file = f
        if name_lower.startswith("mask_caption") and f.suffix.lower() == ".txt":
            mask_caption_file = f

    results = call_unidisc_api(
        image_path=image_file if use_image else None,
        caption=mask_caption_file.read_text().strip() if (mask_caption_file and use_cap_mask) else (caption_file.read_text().strip() if (caption_file and use_caption) else None),
        mask_path=mask_file if use_img_mask else None,
        cfg=cfg,
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    text_parts = []
    img_count = 0
    for i, item in enumerate(results):
        if item["type"] == "text":
            text_parts.append(item["text"])
        elif item["type"] == "image_url":
            out_img = decode_image_base64(item["image_url"]["url"])
            out_img_name = output_folder / f"image.png"
            out_img.save(out_img_name)
            img_count += 1
        if "error" in item:
            text_parts.append(item["text"])

    cfg['mode'] = f"{'img_' if use_image else ''}{'imgmask_' if use_img_mask else ''}{'cap_' if use_caption else ''}{'capmask_' if use_cap_mask else ''}"
    cfg['use_image'] = use_image
    cfg['use_img_mask'] = use_img_mask
    cfg['use_caption'] = use_caption
    cfg['use_cap_mask'] = use_cap_mask

    if len(text_parts) > 0:
        out_txt = output_folder / "caption.txt"
        out_txt.write_text("\n".join(text_parts))
    else:
        shutil.copy(caption_file, output_folder / "caption.txt")
        print(f"No text found, copied input caption to output: mode={cfg['mode']}")

    if img_count == 0:
        shutil.copy(image_file, output_folder / "image.png")
        print(f"No image found, copied input image to output: mode={cfg['mode']}")
    
    config_file = output_folder / "config.json"
    config_file.write_text(json.dumps(cfg, indent=2))

    input_img = (mask_file if use_img_mask else (image_file if use_image else None))
    input_txt = (mask_caption_file if use_cap_mask else (caption_file if use_caption else None))

    input_img = Im(input_img) if input_img else Im.new(h=512, w=512)
    input_txt = input_txt.read_text().strip() if input_txt else "Empty caption"

    input_img.save(output_folder / "input_image.png")
    (output_folder / "input_caption.txt").write_text(input_txt)

@app.command()
def main(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    param_file: Path | None = None,
    num_pairs: int | None = None,
    num_workers: int = 32,
    batch_sleep: float = 0.2,
    use_image: bool = False,
    use_img_mask: bool = False,
    use_caption: bool = False,
    use_cap_mask: bool = False,
    iterate_over_modes: bool = False,
    single_config: bool = False,
):
    """
    Generate datasets by calling the UniDisc API on each (image, caption, mask) triplet in input_dir.
    
    Modified version:
      - Queues tasks in order on a single global ThreadPoolExecutor.
      - After queueing each batch, sleeps for a bit.
      - Does not wait indefinitely for a batch to finish before moving on.
    """

    if use_img_mask:
        assert use_image

    if input_dir is None or output_dir is None:
        raise ValueError("Both input_dir and output_dir must be provided.")

    if param_file is not None:
        all_configs = json.loads(param_file.read_text())
        if not isinstance(all_configs, list):
            raise ValueError("param_file must contain a JSON list of configs.")
    else:
        all_configs = []
        for cfg in [2.5]:
            # for sampler in ["maskgit_nucleus", "maskgit"]:
            for port in ["babel-10-9:8000", "babel-6-29:8001"]:
                all_configs.append(dict(port=port, cfg=cfg))

    if single_config:
        all_configs = [{'port': 'localhost:8000', 'cfg': 2.5}]

    subfolders = sorted([f for f in input_dir.iterdir() if f.is_dir()])
    if num_pairs is not None:
        subfolders = subfolders[:num_pairs]

    configs = []
    from decoupled_utils import sanitize_filename
    for i, cfg in enumerate(all_configs):
        # Build config name from key/values if not explicitly provided
        if "name" not in cfg:
            # Sanitize values and join with underscores
            cfg_name = "_".join(
                f"{k}={str(v).replace('/', '_').replace(' ', '_')}" 
                for k, v in sorted(cfg.items())
            )
        else:
            cfg_name = cfg["name"]
        cfg_output_dir = output_dir / sanitize_filename(cfg_name)
        cfg_output_dir.mkdir(parents=True, exist_ok=True)
        configs.append((cfg, cfg_name, cfg_output_dir))

    # Compute the batch size so each config receives a fair share of workers per batch.
    # E.g., if num_workers=10 and there are 2 configs then each config gets ~5 workers in a batch.
    batch_size = max(1, num_workers // len(configs))
    total_folders = len(subfolders)
    print(f"Processing {total_folders} folders across {len(configs)} configs with batch size {batch_size} per config.")

    modes = []
    if iterate_over_modes:
        modes.append(dict(use_image=False, use_img_mask=False, use_caption=True, use_cap_mask=False)) # T2I
        modes.append(dict(use_image=True, use_img_mask=False, use_caption=False, use_cap_mask=False)) # I2T
        modes.append(dict(use_image=True, use_img_mask=True, use_caption=True, use_cap_mask=True)) # Both masked
        modes.append(dict(use_image=True, use_img_mask=False, use_caption=True, use_cap_mask=True))
        modes.append(dict(use_image=False, use_img_mask=False, use_caption=True, use_cap_mask=False))
    else:
        modes.append(dict(use_image=use_image, use_img_mask=use_img_mask, use_caption=use_caption, use_cap_mask=use_cap_mask))

    # Use a single, global ThreadPoolExecutor so we can submit tasks in order
    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for batch_start in range(0, total_folders, batch_size):
            batch_folders = subfolders[batch_start : batch_start + batch_size]
            for cfg, cfg_name, cfg_out in configs:
                for folder in batch_folders:
                    for mode in modes:
                        use_image = mode['use_image']
                        use_img_mask = mode['use_img_mask']
                        use_caption = mode['use_caption']
                        use_cap_mask = mode['use_cap_mask']
                        key = ""
                        if use_img_mask:
                            key += "imgmask_"
                        elif use_image:
                            key += "img_"

                        if use_cap_mask:
                            key += "capmask_"
                        elif use_caption:
                            key += "cap_"

                        key = key.removesuffix('_')
                        folder_output = cfg_out / f"{key}__{folder.name}"
                        futures.append(
                            executor.submit(
                                run_inference_for_folder,
                                folder=folder,
                                output_folder=folder_output,
                                cfg=cfg,
                                use_image=use_image,
                                use_img_mask=use_img_mask,
                                use_caption=use_caption,
                                use_cap_mask=use_cap_mask,
                            )
                        )
            print(f"Queued batch {batch_start} to {batch_start + len(batch_folders)}. Sleeping for {batch_sleep} seconds...")
            time.sleep(batch_sleep)

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing folders..."):
            future.result()

    print("All processing complete.")

if __name__ == "__main__":
    app()