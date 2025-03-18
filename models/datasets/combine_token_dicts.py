from pathlib import Path
from typing import Optional

import torch
import typer
from tensordict import TensorDict
from typing_extensions import Annotated
import time
import shutil
from decoupled_utils import rprint

app = typer.Typer(pretty_exceptions_show_locals=False)
typer.main.get_command_name = lambda name: name

def split_dataset(dataset, n: int, m: int):
    # Ensure m is valid
    if m < 0 or m >= n:
        raise ValueError(f"m must be between 0 and {n-1}, but got {m}.")
    
    # Calculate the size of each subset
    total_len = len(dataset)
    subset_size = total_len // n
    remainder = total_len % n

    # Calculate the start and end index of the m-th subset
    start_idx = m * subset_size + min(m, remainder)
    end_idx = start_idx + subset_size + (1 if m < remainder else 0)

    # Return the m-th subset
    return dataset[slice(start_idx, end_idx)]
    
@app.command()
def main(
    data_dir: Path,
    splits: Optional[list[str]] = ["train", "val"],
    add_vggface2_text_tokens: bool = False,
    use_tmp: bool = False,
    use_all: bool = False,
    allow_zero_idx: bool = False,
    use_timestamp: bool = False,
    delete_after_combining: bool = False,
    allow_existing: bool = False,
    force_overwrite: bool = False,
    move_files: bool = False,
    allow_tmp: bool = False,
    mem_efficient: bool = False,
    output_dir: Optional[Path] = None,
    require_image_tokens: bool = False,
    min_idx: Optional[int] = None,
    max_idx: Optional[int] = None,
    split_num: Optional[int] = None,
    split_idx: Optional[int] = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for split in splits:
        if allow_tmp:
            all_folders = sorted([folder for folder in data_dir.iterdir() if folder.is_dir() and split in folder.name and "_" in folder.name and (allow_existing or "existing" not in folder.name)])
            print(f"All folders: len({len(all_folders)})")

            from collections import defaultdict
            unique_ids = defaultdict(list)
            for folder in all_folders:
                folder_id = int(folder.name.split("_")[-1])
                unique_ids[folder_id].append(folder)

            folders = []
            for folder_id, _folders in unique_ids.items():
                if len(_folders) == 1:
                    folders.append(_folders[0])
                else:
                    for folder in _folders:
                        if "tmp" not in folder.name:
                            folders.append(folder)

            folders = sorted(folders)
            print(f"Using {len(folders)} folders for {split}")
        else:
            folders = sorted([folder for folder in data_dir.iterdir() if folder.is_dir() and split in folder.name and "_" in folder.name and (use_all or (not use_tmp or "tmp" in folder.name)) and (allow_existing or "existing" not in folder.name)])

        if min_idx is not None and max_idx is not None:
            print(f"Filtering with min_idx: {min_idx} and max_idx: {max_idx}")
            _tmp_folders = []
            for folder in folders:
                _name = int(folder.name.split("_")[-1])
                if min_idx <= _name <= max_idx:
                    _tmp_folders.append(folder)
            folders = _tmp_folders
            print(f"Filtered folders and got: {len(folders)}")

        if split_num is not None and split_idx is not None:
            folders = split_dataset(folders, split_num, split_idx)
            print(f"Filtered folders and got: {len(folders)}")
            
        initial_folder_count = len(folders)
        folders = [folder for folder in folders if any(folder.iterdir())]
        removed_folders_count = initial_folder_count - len(folders)
        print(f"Removed {removed_folders_count} empty folders")
        if len(folders) == 0:
            print(f"No folders found for {split}")
            continue
        print(f"{split} folders: {folders}")
        _tensors = [TensorDict.load_memmap(folder) for folder in folders if (folder / "meta.json").exists()]
        _tensors = [tensor for tensor in _tensors if tensor.shape[0] > 0]
        for _tensor in _tensors:
            if "write_flag" not in _tensor:
                _tensor["write_flag"] = torch.ones((len(_tensor), 1), dtype=torch.bool)
        loaded_tensors = torch.cat(_tensors, dim=0)
        del _tensors

        if add_vggface2_text_tokens:
            loaded_tensors.set("txt_input_ids", loaded_tensors["img_input_ids"].new_zeros(loaded_tensors["img_input_ids"].shape[0], 47), inplace=True)
            loaded_tensors.set("txt_attention_mask", loaded_tensors["img_input_ids"].new_zeros(loaded_tensors["img_input_ids"].shape[0], 1), inplace=True)
            print(f"Added VGGFace2 text tokens to {split}")

        index_keys = ("img_label", "img_input_ids", "txt_input_ids", "input_ids")
        if not mem_efficient:
            for key in index_keys:
                if key in loaded_tensors:
                    loaded_tensors[key] = loaded_tensors[key].to(torch.int32)

        if "img_input_ids" in loaded_tensors:
            written_indices = ((loaded_tensors["write_flag"] > 0).squeeze(-1) & (loaded_tensors["img_input_ids"] > 0).all(dim=-1))
        else:
            if mem_efficient:
                written_indices = (loaded_tensors["write_flag"] > 0).squeeze(-1)
            else:
                written_indices = ((loaded_tensors["write_flag"] > 0).squeeze(-1) & (loaded_tensors["input_ids"] > 0).any(dim=-1))

        print(f"Valid elements for {split}: {written_indices.shape[0]}")
        loaded_tensors = loaded_tensors[written_indices]
        invalid_indices = loaded_tensors["idx"].squeeze(-1) == -1
        if require_image_tokens:
            invalid_modality = ~(loaded_tensors["modality"] > 0).any(dim=-1)
            invalid_indices |= invalid_modality
            print(f"Found {invalid_modality.sum()} invalid indices for {split} due to missing image tokens")
        print(f"Invalid indices for {split}: {invalid_indices.sum()}")

        loaded_tensors = loaded_tensors[~invalid_indices]
        if allow_zero_idx is False:
            _, idx = torch.unique(loaded_tensors["idx"].to(device), dim=0, sorted=True, return_inverse=True)
            loaded_tensors = loaded_tensors[torch.unique(idx, return_inverse=False).to(loaded_tensors.device)]

        print(f"After filtering: {loaded_tensors.shape[0]}")

        if loaded_tensors.shape[0] == 0:
            rprint(f"WARNING!!! No valid elements for {split}")
            return

        for _key in ["img_input_ids", "input_ids"]:
            if _key in loaded_tensors:
                assert 0 <= loaded_tensors[_key].min() and loaded_tensors[_key].max() < torch.iinfo(torch.int16).max
                loaded_tensors[_key] = loaded_tensors[_key].to(torch.int16)

        index_keys = ("img_label", "txt_attention_mask", "attention_mask")
        for key in index_keys:
            if key in loaded_tensors:
                loaded_tensors[key] = loaded_tensors[key].squeeze(-1)

        if "write_flag" in loaded_tensors:
            del loaded_tensors["write_flag"]

        if split_idx is not None:
            split = f"split_{split_idx}_{split}"

        if use_timestamp:
            loaded_tensors.memmap(data_dir / f"{split}_existing_{int(time.time())}")
        else:
            if (data_dir / f"{split}").exists():
                print("Already exists!")
                if force_overwrite:
                    shutil.rmtree(data_dir / f"{split}")
                else:
                    breakpoint()

            if output_dir is not None:
                loaded_tensors.memmap(output_dir / f"{split}")
            else:
                loaded_tensors.memmap(data_dir / f"{split}")

        if delete_after_combining:
            for folder in folders:
                try:
                    rprint(f"Removing folder: {folder}")
                    shutil.rmtree(folder)
                except Exception as e:
                    rprint(f"Error removing folder: {e}")

    if force_overwrite:
        from pathlib import Path
        for train_folder in Path(data_dir).glob('train_*'):
            rprint(f"Removing folder: {train_folder}")
            if train_folder.is_file():
                train_folder.unlink()
            else:
                shutil.rmtree(train_folder)

        train_dir = data_dir / 'train'
        if train_dir.exists() and train_dir.is_dir():
            for item in train_dir.iterdir():
                shutil.move(str(item), str(train_dir.parent))
            shutil.rmtree(train_dir)

    elif move_files:
        train_dir = data_dir / 'train'
        if train_dir.exists() and train_dir.is_dir():
            for item in train_dir.iterdir():
                shutil.move(str(item), str(train_dir.parent))

            # Check if train_dir is empty after moving files
            if train_dir.exists() and train_dir.is_dir():
                if not any(train_dir.iterdir()):
                    shutil.rmtree(train_dir)
                    rprint(f"Removed empty train directory: {train_dir}")

if __name__ == "__main__":
    app()