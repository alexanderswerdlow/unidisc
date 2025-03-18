from collections import defaultdict
from pathlib import Path
import json
import re
import typer

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

# Pre-compile the regex pattern to extract the prefix.
PREFIX_PATTERN = re.compile(r"(.+?)__pair_")

def extract_prefix(folder: str) -> str:
    """
    Extracts the prefix from a folder name using the PREFIX_PATTERN.
    If the pattern does not match, returns the folder name as-is.
    """
    match = PREFIX_PATTERN.match(folder)
    return match.group(1) if match else folder

def get_ignored_reward_keys(prefix: str) -> set[str]:
    """
    Returns a set of raw reward keys to ignore based on the prefix.
    Adjust this mapping as your application logic requires.
    """
    if "capmask" not in prefix and "cap" in prefix:
        return {"text_reward_model_score"}
    elif "imgmask" not in prefix and "img" in prefix:
        return {"laion_aesthetic_score"}
    return set()

@app.command()
def main(
    input_file: Path,
    save_image: bool = False
):
    """
    Reads a generated JSON rewards file and, for each dataset,
    processes each unique prefix. For each prefix, it finds the matching examples
    in the dataset and computes:
      - The overall normalized reward average (normalized from 0 to 1).
      - The normalized average for each raw reward type (ignoring certain types based on the prefix).
      
    The prefix is extracted from each folder name by matching everything before the 
    '__pair_' substring. If the folder name does not match, the entire folder name is used.
    
    Normalization is performed using the global minimum and maximum values for each 
    reward type (computed over all datasets and indices where the reward is not ignored).
    """
    try:
        content = input_file.read_text()
        data = json.loads(content)
    except Exception as e:
        typer.echo(f"Error reading JSON file: {e}")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------------
    # First pass: Compute global normalization stats for each prefix and reward_key.
    #
    # For every dataset, we iterate over its folder_names. For each index,
    # we compute the folder prefix and (using get_ignored_reward_keys) decide which
    # reward keys to process. For each such key and index (if the index exists in the
    # reward key's list), we update our normalization mapping.
    #
    # norm_stats is a dict mapping prefix -> dict mapping reward_key -> (global_min, global_max)
    # ------------------------------------------------------------------------
    norm_stats: dict[str, dict[str, tuple[float, float]]] = {}
    for dataset in data.values():
        folder_names: list[str] = dataset.get("folder_names", [])
        raw_rewards: dict[str, list[float]] = dataset.get("raw_rewards", {})
        for i, folder in enumerate(folder_names):
            current_prefix = extract_prefix(folder)
            ignore_keys = get_ignored_reward_keys(current_prefix)
            for reward_key, values in raw_rewards.items():
                if reward_key in ignore_keys:
                    continue
                if i >= len(values):
                    continue
                value = values[i]
                if current_prefix not in norm_stats:
                    norm_stats[current_prefix] = {}
                if reward_key not in norm_stats[current_prefix]:
                    norm_stats[current_prefix][reward_key] = (value, value)
                else:
                    curr_min, curr_max = norm_stats[current_prefix][reward_key]
                    norm_stats[current_prefix][reward_key] = (min(curr_min, value), max(curr_max, value))

    # Determine unique prefixes from all datasets.
    unique_prefixes: set[str] = set()
    for dataset in data.values():
        folder_names = dataset.get("folder_names", [])
        for folder in folder_names:
            unique_prefixes.add(extract_prefix(folder))
    unique_prefixes = sorted(unique_prefixes)

    print(f"Found {len(unique_prefixes)} unique prefixes: {unique_prefixes}")
    
    # ------------------------------------------------------------------------
    # For each prefix, process and sort dataset outputs by overall normalized reward.
    #
    # In each dataset we find the indices with the current prefix, then for each
    # reward key (that is not globally ignored for this prefix) we first normalize
    # each reward value using the pre-computed min and max and then average the values.
    # The overall average is computed (as in the original code) by summing the averages
    # for each reward key and dividing by the total number of raw reward keys.
    # ------------------------------------------------------------------------
    for prefix in unique_prefixes:
        typer.echo(f"Prefix: {prefix}")
        dataset_outputs = []  # List of tuples: (overall_avg, output_string)
        img_outputs = defaultdict(list)
        for dataset_name, dataset in data.items():
            output_lines = []
            output_lines.append(f"  Dataset: {dataset_name}")

            folder_names: list[str] = dataset.get("folder_names", [])
            folder_paths: list[str] = dataset.get("folder_paths", [])
            raw_rewards: dict[str, list[float]] = dataset.get("raw_rewards", {})

            if not folder_names:
                output_lines.append("  No folder names provided in this dataset.")
                dataset_outputs.append((float("-inf"), "\n".join(output_lines)))
                continue

            # Compute the indices in this dataset with the target prefix.
            indices = [
                idx for idx, folder in enumerate(folder_names)
                if extract_prefix(folder) == prefix
            ]

            if save_image:
                num_to_save = 2
                _folder_paths = sorted([Path(p) for p in folder_paths])
                for idx in indices[:num_to_save]:
                    img_outputs[_folder_paths[idx].name].append((dataset_name, _folder_paths[idx]))

            ignore_keys = get_ignored_reward_keys(prefix)
            reward_details = ""
            total_norm_rewards = 0.0
            for reward_key, values in raw_rewards.items():
                if reward_key in ignore_keys:
                    continue

                # Retrieve the global min and max for this reward key under this prefix.
                norm_info = norm_stats.get(prefix, {}).get(reward_key)
                if norm_info is None:
                    reward_details += f"{reward_key}: No data, "
                    continue
                min_val, max_val = norm_info

                # Normalize the values using the global min and max.
                group_norm_values = []
                for i in indices:
                    if i < len(values):
                        orig_value = values[i]
                        normalized = ((orig_value - min_val) / (max_val - min_val)) if max_val != min_val else 0.0
                        group_norm_values.append(normalized)
                if group_norm_values:
                    avg_norm = sum(group_norm_values) / len(group_norm_values)
                    reward_details += f"{reward_key}: {avg_norm:.4f}, "
                    total_norm_rewards += avg_norm
                else:
                    reward_details += f"{reward_key}: No data, "

            # Compute the overall average normalized reward for sorting.
            overall_avg = total_norm_rewards / len(raw_rewards) if raw_rewards else 0.0

            reward_details = f"    Avg: {overall_avg:.4f}, " + reward_details
            output_lines.append(reward_details)
            dataset_outputs.append((overall_avg, "\n".join(output_lines)))
        
        # Sort the dataset outputs by overall average normalized reward (descending).
        for avg, out in sorted(dataset_outputs, key=lambda x: x[0], reverse=True):
            typer.echo(out)

        typer.echo("-" * 40)

        if save_image:
            from unidisc.utils.viz_utils import create_text_image
            from PIL import Image
            from image_utils import Im
            for k, v in img_outputs.items():
                imgs = []
                for _dataset_name, _folder_path in v:
                    def get_img(_img_path, _txt_path):
                        _img = Image.open(_img_path).resize((1024, 1024))
                        out = f'{_dataset_name}: {_txt_path.read_text()}'
                        txt_img = create_text_image(out, desired_width=_img.width)
                        _img = Im.concat_vertical(_img, txt_img)

                    input_img = None
                    if (_folder_path / "input_image.png").exists():
                        input_img = get_img(_folder_path / "input_image.png", _folder_path / "input_caption.txt")
                        
                    out_img = get_img(_folder_path / "image.png", _folder_path / "caption.txt")
                    if input_img:
                        imgs.append(Im.concat_vertical(input_img, out_img))
                    else:
                        imgs.append(out_img)

                Im.concat_horizontal([x.pil for x in imgs]).save(f"{k}.png")

if __name__ == "__main__":
    app()
