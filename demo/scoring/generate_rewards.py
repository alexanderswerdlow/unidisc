import json
from pathlib import Path

import torch
import typer
from image_utils import Im
from omegaconf import OmegaConf
from tqdm import tqdm
from accelerate.state import PartialState
from accelerate.utils import gather_object
from PIL import Image

from decoupled_utils import set_global_breakpoint
from model import Diffusion

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

set_global_breakpoint()

@app.command()
def main(
    input_dir: Path | None = None,
    output_file: Path | None = None,
    batch_size: int = 32,
    resolution: int = 512,
    num_pairs: int | None = None,
    num_dirs: int | None = None,
):
    """
    Process datasets contained in subdirectories of `input_dir`, distributed across multiple GPUs.
    Each GPU processes complete datasets for better efficiency.
    """
    distributed_state = PartialState()
    device = distributed_state.device
    dtype = torch.bfloat16

    # Initialize model without Accelerator
    model = Diffusion(None, None, device, disable_init=True)
    model.device = device
    model.dtype = dtype

    reward_config = OmegaConf.create({
        "dfn_score": 1.0,
        "hpsv2_score": 1.0,
        "clip_score": 1.0,
        "laion_aesthetic_score": 1.0,
        "text_reward_model_score": 1.0
    })

    all_rewards = {}
    # Get all dataset directories and distribute them across GPUs
    dataset_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not dataset_dirs:
        if distributed_state.is_main_process:
            print("No dataset directories found in the input directory.")
        raise typer.Exit()
    
    if num_dirs is not None:
        dataset_dirs = dataset_dirs[:num_dirs]

    # Split datasets across processes
    with distributed_state.split_between_processes(dataset_dirs) as process_dataset_dirs:
        for ds_dir in tqdm(process_dataset_dirs, desc=f"Processing datasets (GPU {distributed_state.process_index})"):
            if distributed_state.is_main_process:
                print(f"Processing dataset: {ds_dir.name}")
            
            pair_dirs = sorted([p for p in ds_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
            if num_pairs is not None:
                pair_dirs = pair_dirs[:num_pairs]
            if not pair_dirs:
                if distributed_state.is_main_process:
                    print(f"  No pair subdirectories found in {ds_dir.name}, skipping.")
                continue

            images = []
            captions = []
            for pair_dir in sorted(pair_dirs, key=lambda p: p.name):
                image_path = pair_dir / "image.png"
                caption_path = pair_dir / "caption.txt"

                if not (image_path.exists() and caption_path.exists()):
                    print(f"  Skipping {pair_dir}: missing image.png or caption.txt")
                    continue

                try:
                    img = Image.open(image_path)
                    if resolution != img.height or resolution != img.width:
                        print(f"WARNING!!! Image resolution {img.height}x{img.width} does not match resolution {resolution}x{resolution}")
                        min_dim = min(img.width, img.height)
                        left = (img.width - min_dim) // 2
                        top = (img.height - min_dim) // 2
                        img = img.crop((left, top, left + min_dim, top + min_dim))
                        img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
                    images.append(Im(img).torch.unsqueeze(0))
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    continue

                try:
                    caption = caption_path.read_text().strip()
                    captions.append(caption)
                except Exception as e:
                    print(f"Error reading caption {caption_path}: {e}")
                    continue

            num_pairs = len(images)
            if num_pairs == 0:
                print(f"No valid pairs found in dataset {ds_dir.name}, skipping.")
                continue

            dataset_reward_batches = []
            dataset_raw_rewards = []
            for i in tqdm(range(0, num_pairs, batch_size), desc="Processing pairs"):
                batch_imgs = torch.cat(images[i : i + batch_size], dim=0).to(device) / 255.0
                batch_texts = captions[i : i + batch_size]
                with torch.inference_mode():
                    rewards, raw_rewards = model.get_rewards(reward_config, batch_imgs, batch_texts, None, return_raw_rewards=True)
                dataset_reward_batches.append(rewards.cpu())
                dataset_raw_rewards.append(raw_rewards)

            dataset_rewards_tensor = torch.cat(dataset_reward_batches, dim=0)
            dataset_raw_rewards_dict = {}
            for key in raw_rewards.keys():
                dataset_raw_rewards_dict[key] = torch.cat(
                    [batch[key] for batch in dataset_raw_rewards], dim=0
                )

            all_rewards[ds_dir.name] = {
                "rewards": dataset_rewards_tensor.tolist(),
                "raw_rewards": {k: v.tolist() for k, v in dataset_raw_rewards_dict.items()},
                "folder_names": [f.name for f in pair_dirs],
                "folder_paths": [f.as_posix() for f in pair_dirs]
            }
            if distributed_state.is_main_process:
                print(f"Finished processing {num_pairs} pairs from {ds_dir.name}")

    gathered_rewards = gather_object([all_rewards])
    
    all_keys = set()
    all_gathered_rewards = {}
    for i in range(len(gathered_rewards)):
        assert len(set(gathered_rewards[i].keys()).intersection(all_keys)) == 0
        all_keys.update(gathered_rewards[i].keys())
        all_gathered_rewards.update(gathered_rewards[i])

    gathered_rewards = all_gathered_rewards

    if distributed_state.is_main_process:
        print("All rewards:")
        print(json.dumps(gathered_rewards, indent=2))

        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(gathered_rewards, f, indent=2)
            print(f"Rewards saved to {output_file}")
        except Exception as e:
            print(f"Error saving rewards to file: {e}")


if __name__ == "__main__":
    app() 