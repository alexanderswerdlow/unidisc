from email.mime import image
import os
import random
import typing
from pathlib import Path
from typing import Optional
import subprocess
import datasets
import torch
from numpy import pad
from PIL import Image, ImageFile
from tensordict import TensorDict
from torchvision import transforms
from decoupled_utils import get_world_size
import time
import re
import shutil
from constants import UNIDISC_DIR
from decoupled_utils import barrier, get_rank, gprint, is_local_main_process, is_main_process, is_torch_cuda_available, is_torch_xla_available, rprint
from models.datasets.webdataset_utils import get_data
import hashlib
from decoupled_utils import sanitize_filename
from omegaconf import OmegaConf, read_write
from models.datasets.misc_image_datasets import *
from copy import deepcopy
from datasets import Dataset, DatasetDict
import numpy as np
from PIL import Image
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Subset

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
    indices = list(range(start_idx, end_idx))
    if isinstance(dataset, torch.utils.data.Dataset):
        return Subset(dataset, indices)
    else:
        return dataset[slice(start_idx, end_idx)]

def get_webdataset_indexed(config, tokenizer, transform, cond_transform, n_samples, name, should_tokenize=False):
    should_tokenize = ("tokenize" in name) or should_tokenize
    import wids  # You need to use the custom sampler!!

    custom_ignore_func_dict = {
        "pixelprose": lambda x: len(x[".txt"]) > 400,
    }

    valid_func = None
    for k in custom_ignore_func_dict.keys():
        if k in name:
            valid_func = custom_ignore_func_dict[k]
            break

    from dataloader import tokenize_text

    def process(x, idx):
        data = {}

        if "mmc4" in name:
            print(x['.json']['image_info'][0])
            breakpoint()

        img = x[".jpg"].convert("RGB")
        data["is_valid"] = True
        if valid_func is not None and valid_func(x) is False:
            print(f"Invalid")
            data["is_valid"] = False

        data["img"] = transform(img)
        if cond_transform is not None:
            data["cond_img"] = cond_transform(x[".jpg"].convert("RGB"))

        if data["img"].shape[0] != 3:
            raise Exception(f"Image shape: {data['img'].shape}, {x['.jpg'].size}, {x['.jpg'].mode}")

        if "pixelprose" in name:
            before = x[".txt"]
            x[".txt"] = re.sub(r"This image displays.*?(?=[a-zA-Z0-9])", "", x[".txt"])
            if abs(len(before) - len(x[".txt"])) > 100:
                data["is_valid"] = False

        if not "imagenet" in name:
            if should_tokenize:
                tokens = tokenize_text(tokenizer, config.data.block_size, x[".txt"])
                data["input_ids"] = tokens["input_ids"]
                data["attention_mask"] = tokens["attention_mask"].float()
            else:
                data[".txt"] = x[".txt"]

        data["idx"] = idx

        return data

    disable_split = False
    if isinstance(config.data.raw_data_dir, str) and '*' in config.data.raw_data_dir:
        import glob
        index_path = sorted(glob.glob(config.data.raw_data_dir))
        if not index_path:
            raise ValueError(f"No files found matching the pattern: {config.data.raw_data_dir:}")
        print(f"Expanded glob pattern to {len(index_path)} files")
        if os.getenv("SLURM_ARRAY_TASK_COUNT", None) is not None:
            index_path = split_dataset(index_path, int(os.getenv("SLURM_ARRAY_TASK_COUNT")), int(os.getenv("SLURM_ARRAY_TASK_ID")))
            print(f"After splitting, dataset is length {len(index_path)}")
        shards = []
        for shard in index_path:
            shards.append({"url": shard, "nsamples": wids.wids.compute_num_samples(shard)})
            print(f"Shard: {shard}")
        index_path = shards
        disable_split = True
    elif Path(config.data.raw_data_dir).is_file():
        index_path = config.data.raw_data_dir
    else:
        default_path = Path(config.data.raw_data_dir) / "index.json"
        shard_path = Path(config.data.raw_data_dir) / "shardindex.json"
        index_path = str(default_path if default_path.exists() else shard_path)
        
    assert getattr(config.data, "shard_list_path", None) is None, "shard_list_path is deprecated, use raw_data_dir instead"
    dataset = wids.ShardListDataset(index_path)  # lru_size=20
    dataset = CustomTransformDataset(dataset, process)

    if n_samples is not None:
        from torch.utils.data import Subset
        indices = torch.randperm(len(dataset))[:n_samples]
        dataset = Subset(dataset, indices)

    if config.data.split_dataset and not disable_split:
        gprint(f"Original dataset was length {len(dataset)}")
        dataset = split_dataset(dataset, int(os.getenv("SLURM_ARRAY_TASK_COUNT")), int(os.getenv("SLURM_ARRAY_TASK_ID")))
        gprint(f"After splitting, dataset is length {len(dataset)}")

    return dataset


def _copy_data(src_path, dst_path, use_rsync=True):
    dst_path.mkdir(parents=True, exist_ok=True)
    if use_rsync:
        rprint(f"Rsyncing data from {src_path} to {dst_path}")
        rsync_command = ["rsync", "-av", str(src_path) + "/", str(dst_path) + "/"]
        try:
            result = subprocess.run(rsync_command, check=True, capture_output=True, text=True)
            rprint(f"Rsync output: {result.stdout}")
            rprint(f"Successfully rsynced data from {src_path} to {dst_path}")
        except subprocess.CalledProcessError as e:
            rprint(f"Rsync failed: {e}")
            rprint(f"Rsync stderr: {e.stderr}")
            raise
    else:
        rprint(f"Copying tensordict from {src_path} to {dst_path}")
        shutil.copytree(src_path, dst_path)
        rprint(f"Copied tensordict from {src_path} to {dst_path}")

def copy_data(shm_path, src_path, dst_path):
    shm_path.mkdir(parents=True, exist_ok=True)
    use_rsync = True
    if not dst_path.exists() or use_rsync:
        _copy_data(src_path, dst_path, use_rsync=use_rsync)
    else:
        src_files = sum(1 for _ in src_path.rglob('*'))
        dst_files = sum(1 for _ in dst_path.rglob('*'))
        src_size = sum(f.stat().st_size for f in src_path.rglob('*') if f.is_file())
        dst_size = sum(f.stat().st_size for f in dst_path.rglob('*') if f.is_file())
        size_diff_percent = abs(src_size - dst_size) / max(src_size, dst_size) * 100
        if src_files != dst_files or size_diff_percent > 10:
            rprint(f"Src files: {src_files}, Dst files: {dst_files} contain different number of files, {src_size} {dst_size}, size difference {size_diff_percent}, Deleting {dst_path}")
            shutil.rmtree(dst_path)
            rprint(f"Deleted {dst_path}, copying from {src_path}")
            _copy_data(src_path, dst_path, use_rsync=False)
            rprint(f"Deleted and re-copied tensordict from {src_path} to {dst_path}")
        else:
            rprint(f"Tensordict already exists at {dst_path}, loading from there")

def get_tensordict(config, path, dataset_idx, dataset_name=None):
    parquet_files = list(Path(path).glob('*.arrow'))
    if parquet_files:
        # Does not load into memory by default
        from datasets import load_from_disk
        dataset = load_from_disk(path)
        rprint(f"Loaded {len(dataset)} samples from {path} as parquet")
        return dataset
    
    if getattr(config.data, "force_dummy_tensordict", False):
        return get_dummy_tensordict(config, 1000000, dataset_idx=dataset_idx)
    
    if config.data.move_tensordict_to_shm:
        assert config.data.keep_tensordict_on_disk is True
        shm_path = Path(getattr(config.data, "shm_path", Path("/dev/shm") / Path.home().name))
        src_path = Path(path)
        dst_path = shm_path / (dataset_name if dataset_name is not None else src_path.name)
        if getattr(config.data, "skip_copy_tensordict_to_shm", False):
            gprint(f"Skipping copy of tensordict to SHM")
        elif is_torch_xla_available():
            if is_main_process():
                copy_data(shm_path, src_path, dst_path)
            
            barrier()
            if not is_main_process():
                import time
                from torch_xla._internal import tpu
                host_ip = tpu.get_worker_ips()[0]
                file_dst_path = Path(shm_path)
                src_path_on_host = f"aswerdlow@{host_ip}:{file_dst_path}"
                gprint(f"Copying data from {src_path_on_host} to {file_dst_path}")
                file_dst_path.mkdir(parents=True, exist_ok=True)
                max_retries = 5
                retry_delay = 15
                for attempt in range(max_retries):
                    try:
                        gprint(f"After main copy, rsyncing data from {src_path_on_host} to {file_dst_path}")
                        command = f'bash {(UNIDISC_DIR / "scripts/rsync_data.sh").resolve()} {src_path_on_host}/ {file_dst_path}/'
                        os.environ.pop('SSH_AUTH_SOCK', None) # Breaks without this
                        gprint(command)
                        subprocess.run(command, shell=True, check=True)
                        gprint(f"Successfully rsynced data from {src_path_on_host} to {file_dst_path}")
                        break
                    except subprocess.CalledProcessError as e:
                        if attempt < max_retries - 1:
                            gprint(f"Rsync attempt {attempt + 1} failed. Retrying in {retry_delay} seconds..., {e}")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            raise RuntimeError(f"Failed to rsync data after {max_retries} attempts: {e}")
                        
                gprint(f"Finished rsyncing data from {src_path_on_host} to {file_dst_path}")
            barrier()
        else:
            if is_local_main_process():
                copy_data(shm_path, src_path, dst_path)

            # For now we assume we are on SPMD and there is only one process per worker [node]
            if not is_torch_xla_available():
                barrier()

    else:
        dst_path = Path(path)
        
    path = dst_path
    data = TensorDict.load_memmap(path)
    if config.data.keep_tensordict_on_disk:
        rprint(f"Keeping tensordict on disk at {path}")
    else:
        data = data.clone()  # Move to CPU memory
    rprint(f"Loaded {len(data)} samples from {path}")
    return data


def get_dummy_tensordict(config, dataset_size, txt_length=None, img_length=None, dataset_idx=0):
    if img_length is None:
        img_length = config.model.img_length
    if txt_length is None:
        txt_length = config.model.txt_length
    return TensorDict(
        {
            "input_ids": torch.ones(dataset_size, config.model.length, dtype=torch.int32),
            "attention_mask": torch.ones(dataset_size, config.model.length, dtype=torch.bool),
            "img_input_ids": torch.ones(dataset_size, img_length, dtype=torch.int16),
            "txt_input_ids": torch.ones(dataset_size, txt_length, dtype=torch.int32),
            "txt_attention_mask": torch.ones(dataset_size, txt_length, dtype=torch.bool),
            "idx": torch.arange(dataset_size, dtype=torch.int32).view(-1, 1),
            "dataset_idx": torch.full((dataset_size,), fill_value=dataset_idx, dtype=torch.int32),
            "write_flag": torch.zeros(dataset_size, 1, dtype=torch.bool),
        },
        batch_size=[dataset_size],
    )


def get_token_dataset(config, name, is_train, n_samples, n_duplicate, tokenizer):
    assert getattr(config.data, "token_data_dir", None) is None, "token_data_dir is deprecated, use data_dir_train and data_dir_val instead"
    if "dummy" in name:
        return get_dummy_tensordict(config, n_samples if n_samples is not None else 100000)
    data_key = (
        config.data.data_dir_train if is_train else (config.data.data_dir_val if config.data.data_dir_val is not None else config.data.data_dir_train)
    )
    image_datasets_key = getattr(config.data, "image_data_train", None) if is_train else getattr(config.data, "image_data_val", None)

    if config.data.use_weighted_tensordict_sampler:
        _dataset_cls = MultipleTensorDictDataset
        _datasets = [get_tensordict(config, x['dir'], dataset_idx=i, dataset_name=x['name']) for i, x in enumerate(data_key)]
        _weights = [x['weight'] for x in data_key]
        _dataset_names = [x['name'] for x in data_key]
        _kwargs = dict()
        _kwargs["config"] = config
        _kwargs["tokenizer"] = tokenizer

        if any(not isinstance(x, TensorDict) for x in _datasets):
            _kwargs["returns_parquet"] = True
        elif getattr(config.data, "add_text_to_weighted_sampler", False):
            from datasets import load_dataset, interleave_datasets
            rprint("Loading smollm datasets")
            ds1 = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", cache_dir=config.data.cache_dir, streaming=True)
            ds2 = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", cache_dir=config.data.cache_dir, streaming=True)
            # DKYoon/SlimPajama-6B, "cerebras/SlimPajama-627B"
            ds3 = load_dataset("DKYoon/SlimPajama-6B", split="train", cache_dir=config.data.cache_dir, streaming=True)
            ds4 = load_dataset("bigcode/starcoderdata", split="train", cache_dir=config.data.cache_dir, streaming=True)
            rprint(f"Finished loading datasets")
            if getattr(config.data, "code_only", False):
                _dataset = ds4
            else:
                _dataset = interleave_datasets([ds1, ds2, ds3, ds4], probabilities=[0.3, 0.3, 0.2, 0.2], seed=config.seed)

            rprint(f"Finished interleaving datasets")
            _datasets.append(_dataset)
            _weights.append(1)
            _dataset_names.append("SlimPajama-627B")
            _kwargs["returns_tokenized_text"] = True
            rprint(f"Finished creating dataset")
        elif image_datasets_key is not None:
            returns_raw_images = False
            tokenize_vqvae_in_dataloader = False
            allow_label = False
            for key in image_datasets_key:
                _key = OmegaConf.to_object(key)
                if _key.get("raw_images", False) or config.data.force_raw_images_in_multiple_tensordict:
                    rprint(f"WARNING!!! Using raw images")
                    returns_raw_images = True

                if _key.get("tokenize_vqvae_in_dataloader", False):
                    tokenize_vqvae_in_dataloader = True

                if _key.get("allow_label", False):
                    rprint(f"WARNING!!! Using allow_label")
                    allow_label = True

                if config.data.force_raw_images_in_multiple_tensordict:
                    tokenize_vqvae_in_dataloader = False
                    _key["tokenize_vqvae_in_dataloader"] = False
                    _key["disable_text_modality"] = True

                image_config = OmegaConf.merge(deepcopy(config),
                    {
                        "data": {
                            **{k:v for k,v in _key.items() if k not in {"dir", "weight", "name", "raw_images"}}
                        },
                    }
                )
                image_dataset = get_image_dataset(
                    mode="train" if is_train else "val",
                    config=image_config,
                    tokenizer=tokenizer,
                    allow_aug=is_train,
                    force_aug=False,
                    name=key["train"] if is_train else key["val"],
                )
                _datasets.append(image_dataset)
                _weights.append(key["weight"])
                _dataset_names.append(key["name"])

            _kwargs["returns_raw_images"] = returns_raw_images
            _kwargs["returns_tokenize_vqvae_in_dataloader"] = tokenize_vqvae_in_dataloader
            _kwargs["allow_label"] = allow_label

        if n_samples is not None:
            if getattr(config.data, "force_no_shuffle_tensordict", False):
                _datasets = [data[:n_samples] for data in _datasets]
            else:
                _datasets = [data[torch.randperm(len(data), generator=torch.Generator().manual_seed(config.seed))[:n_samples]] for data in _datasets]
            rprint(f"Sampled {n_samples} samples from {len(_datasets)}, is_train: {is_train}.")
        
        data = _dataset_cls(datasets=_datasets, weights=_weights, dataset_names=_dataset_names, **_kwargs)
    else:
        data = get_tensordict(config, data_key, 0)
        if n_samples is not None:
            if getattr(config.data, "force_no_shuffle_tensordict", False):
                indices = list(range(n_samples))
            else:
                indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(config.seed))[:n_samples]
            data = data[indices]
            rprint(f"Sampled {n_samples} samples from {len(data)}, is_train: {is_train}. First 2 indices: {indices[:2]}")

    if n_duplicate is not None:
        data = torch.cat([data for _ in range(n_duplicate)], dim=0)
        rprint(f"Duplicated {n_duplicate} times, now {len(data)} samples")

    return data


class UnpairedDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, img_dataset, txt_dataset, shuffle=True):
        self.img_dataset = img_dataset
        self.txt_dataset = txt_dataset
        self.shuffle = shuffle

    def __len__(self):
        if self.shuffle:
            return min(len(self.img_dataset), len(self.txt_dataset))
        else:
            return max(len(self.img_dataset), len(self.txt_dataset))

    def __getitem__(self, idx):
        while True:
            try:
                if self.shuffle:
                    img_idx = torch.randint(0, len(self.img_dataset), (1,)).item()
                    txt_idx = torch.randint(0, len(self.txt_dataset), (1,)).item()
                else:
                    txt_idx = idx
                    img_idx = idx % len(self.img_dataset)
                return dict(**self.img_dataset[img_idx], **self.txt_dataset[txt_idx])
            except Exception as e:
                gprint(e)
                import traceback

                traceback.print_exc()
                idx = (idx + 1) % len(self)


def get_unpaired_dataset(config=None, tokenizer=None, mode="train", **kwargs):
    image_dataset = get_image_dataset(config=config, mode=mode, tokenizer=tokenizer, **kwargs)
    from models.datasets.text_datasets import get_text_dataset

    text_dataset = get_text_dataset(
        dataset_name=getattr(config.data, "txt_train", "text8") if mode == "train" else getattr(config.data, "txt_val", "text8"),
        tokenizer=tokenizer,
        mode="test" if (mode == "validation" and getattr(config.data, "txt_val", "text8") == "lm1b") else mode,
        wrap=config.data.wrap,
        block_size=config.model.txt_length,  # Intentional
        cache_dir=config.data.cache_dir,
        num_proc=config.data.num_proc,
        streaming=config.data.streaming,
    )
    return UnpairedDatasetWrapper(image_dataset, text_dataset, shuffle=getattr(config.data, "force_disable_shuffle", False) is False)


def get_transform(resolution, orig_mode, allow_aug, force_aug, aggressive_aug=False):
    if orig_mode == "train" and (allow_aug or force_aug):
        if aggressive_aug:
            rprint("Using aggressive augmentations")
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop((resolution, resolution), scale=(0.8, 1.0), ratio=(0.97, 1.03)),
                    transforms.RandomHorizontalFlip(1.0 if force_aug else 0.5),
                    transforms.ToTensor(),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.RandomCrop((resolution, resolution)),
                    transforms.RandomHorizontalFlip(1.0 if force_aug else 0.5),
                    transforms.ToTensor(),
                ]
            )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop((resolution, resolution)),
                transforms.ToTensor(),
            ]
        )
    return transform

def load_vqvae_from_cache(config, full_cache_path):
    global_cache_parent = os.environ.get("DIFFUSION_DATA_DIR", None)
    if global_cache_parent is not None:
        global_full_cache_path = Path(global_cache_parent) / full_cache_path.relative_to(Path(config.data.cache_dir))
        gprint(f"Checking global cache path: {global_full_cache_path}")
        if global_full_cache_path.exists() and len(list(global_full_cache_path.iterdir())) > 0:
            gprint(f"Loading data from global cache: {global_full_cache_path}")
            full_cache_path = global_full_cache_path

    if not (full_cache_path.exists() and len(list(full_cache_path.iterdir())) > 0):
        gprint(f"Cache path {full_cache_path} does not exist or is empty")
        return None

    gprint(f"Loading data from: {full_cache_path}, found {len(list(full_cache_path.iterdir()))} shards")
    ret = []
    kwargs = dict()

    if config.data.keep_hf_dataset_in_memory:
        kwargs["keep_in_memory"] = True
        if config.loader.num_workers > 0:
            for _ in range(5):
                gprint(f"WARNING!!!! Keeping dataset in memory and num_workers > 0, this will cause excessive memory usage")
        else:
            gprint(f"Loading datasets into memory")

    for folder in full_cache_path.iterdir():
        if folder.is_dir():
            ret.append(datasets.load_from_disk(folder, **kwargs))
    
    ret = datasets.concatenate_datasets(ret).with_format("torch")
    gprint(f"Loaded data from cache: {full_cache_path} with {len(ret)} samples")
    return ret

def get_vqvae_dataloader(config, name, split):
    cache_key = f'vqvae_tokenized_{name}_{split}_{config.data.resolution}'
    vae_ckpt_hash = ""

    if hasattr(config.model, "use_custom_vae_ckpt") and config.model.use_custom_vae_ckpt:
        vae_ckpt_hash = hashlib.md5(str(Path(config.model.use_custom_vae_ckpt).name).encode()).hexdigest()[:8]
        cache_key += f"_{vae_ckpt_hash}"
    if hasattr(config.model, "vae_type") and config.model.vae_type != "VQ-16":
        cache_key += f"_{config.model.vae_type}"
    if getattr(config.data, "vqvae_cache_suffix", None) is not None:
        cache_key += f"_{config.data.vqvae_cache_suffix}"

    cache_dir = config.data.cache_dir
    full_cache_path = Path(cache_dir) / "tokens" / sanitize_filename(cache_key)
    return full_cache_path


def get_image_dataset(mode, config, tokenizer, allow_aug=True, force_aug=False, name=None, **kwargs):
    rprint(f"Getting image dataset with mode {mode}")
    if getattr(config.data, "tokenizers_parallelism", None) is not None:
        rprint(f"Setting tokenizers parallelism to {config.data.tokenizers_parallelism}")
        os.environ["TOKENIZERS_PARALLELISM"] = "false" if config.data.tokenizers_parallelism is False else "true"

    resolution = config.data.resolution
    name = name or config.data.train
    streaming = config.data.streaming
    precache = config.data.precache
    dynamic = streaming or precache is False

    orig_mode = mode
    block_size = getattr(config.data, "block_size", 1024)
    is_train = orig_mode == "train"

    n_duplicate_train = getattr(config.data, "n_duplicate_train", None)
    n_duplicate_val = getattr(config.data, "n_duplicate_val", None)
    n_duplicate = n_duplicate_train if is_train else n_duplicate_val

    n_val_samples = getattr(config.data, "n_val_samples", None)
    n_train_samples = getattr(config.data, "n_train_samples", None)
    n_samples = n_train_samples if is_train else n_val_samples

    raw_data_dir = getattr(config.data, "raw_data_dir", getattr(config.data, "data_dir", None))
    rprint(f"Data dir is {raw_data_dir}")
    unified_model = getattr(config.model, "unified_model", False) and getattr(config.data, "unpaired", False) is False

    cond_resolution = getattr(config.data, "cond_resolution", None)
    
    if "sora" in name:
        return get_sora_dataset(config=config, tokenizer=tokenizer, **kwargs)
    elif "tokens" in name:
        print(f"Loading token dataset {name}")
        assert config.data.use_token_dataset, "data.use_token_dataset must be true to load token datasets"
        return get_token_dataset(config, name, is_train, n_samples, n_duplicate, tokenizer)

    dataset_splits = {
        "cassiekang/cub200_dataset": (
            "train"
            if ((orig_mode == "train" and n_train_samples is not None) or (orig_mode != "train" and n_val_samples is not None))
            else "train+test"
        ),
        "nlphuji/flickr30k": "test",
        "richwardle/reduced-imagenet": "train",
        "tglcourse/lsun_church_train": "train" if is_train else "test",
        "pixparse/cc12m-wds": "train",
        "imagenet": "train" if is_train else "val",
        "imagefolder": "train" if is_train else "validation",
        "ILSVRC/imagenet-1k": "train" if is_train else "validation",
        "pouya-haghi/imagenet-subset": "validation",
        "laion/clevr-webdataset": "train" if is_train else "validation",
        "pcuenq/lsun-bedrooms": "train" if is_train else "test",
        "facebook/winoground": "test",
        "sayakpaul/coco-30-val-2014": "train"
    }

    split = dataset_splits[name] if name in dataset_splits else "train"

    if n_samples is not None:
        split = f"{split}[:{n_samples}]"

    extra_kwargs = dict()
    cache_dir = Path(config.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if "HF_HUB_DATASETS_TOKEN" in os.environ:
        extra_kwargs["token"] = os.environ["HF_HUB_DATASETS_TOKEN"]

    if name == "mmc4" or name == "cambrian":
        from unidisc.tokenizers.tokenize_interleaved import JsonlDataset
        dataset = JsonlDataset(glob_pattern=config.data.raw_data_dir)
        
        if n_samples is not None:
            from torch.utils.data import Subset
            indices = list(range(len(dataset)))[:n_samples]
            dataset = Subset(dataset, indices)
            
        if config.data.split_dataset:
            if getattr(config.data, "split_dataset_total_count", None) is not None and \
                getattr(config.data, "split_dataset_cur_idx", None) is not None:
                gprint(f"Splitting dataset into {config.data.split_dataset_total_count} shards, original length {len(dataset)}")
                dataset = split_dataset(dataset, config.data.split_dataset_total_count, config.data.split_dataset_cur_idx)

            gprint(f"Original dataset was length {len(dataset)}")
            total_count, cur_idx = int(os.getenv("SLURM_ARRAY_TASK_COUNT")), int(os.getenv("SLURM_ARRAY_TASK_ID"))
            dataset = split_dataset(dataset, total_count, cur_idx)
            gprint(f"After splitting, dataset is length {len(dataset)}")

        return dataset

    if name == "imagefolder":
        from datasets.data_files import DataFilesDict
        with open(config.data.train_data_dir, "r") as f:
            train_txt = [f"{config.data.data_dir}/{line.strip()}" for line in f.readlines()]
        with open(config.data.val_data_dir, "r") as f:
            val_txt = [f"{config.data.data_dir}/{line.strip()}" for line in f.readlines()]
        data_files = DataFilesDict({"train": train_txt, "validation": val_txt})
        extra_kwargs["data_files"] = data_files

    if config.data.tokenize_vqvae_in_dataloader and not getattr(config.data, "allow_aug_vqvae_dataloader", False):
        rprint(f"WARNING!!!! Disabling augmentations for VQVAE dataloader")
        allow_aug = False
        force_aug = False

    transform = get_transform(resolution, orig_mode, allow_aug, force_aug, getattr(config.data, "aggressive_aug", False))
    if cond_resolution is not None:
        cond_transform = get_transform(cond_resolution, orig_mode, allow_aug, force_aug)
    else:
        cond_transform = None

    if kwargs.get("transform", None) is not None:
        rprint(f"Using transform from kwargs: {kwargs['transform']}")
        transform = kwargs.pop("transform")

    if name == "torchvision_imagenet":
        from torchvision.datasets import ImageFolder

        raw_data_dir = Path(config.data.raw_data_dir)
        raw_data_dir = raw_data_dir / "train" if orig_mode == "train" else raw_data_dir / "val"
        dataset = ImageFolder(raw_data_dir, transform=transform)
        dataset = CustomTransformDataset(dataset, lambda x, idx: {"img": x[0], "label": x[1]})
        return dataset

    if "pixparse/cc12m-wds-fast" in name or "pixparse/cc3m-wds-fast" in name or "indexed" in name:
        return get_webdataset_indexed(config, tokenizer, transform, cond_transform, n_samples, name, should_tokenize=True)

    if name == "vggface2":
        dataset = VGGFace(
            Path(raw_data_dir),
            is_train,
            transform=transform,
            filter_resolution=(resolution - 48),
            cond_transform=cond_transform,
            v2=getattr(config.data, "add_vggface_v2_attributes", False),
        )
        rprint(f"VGGFace2 has size {len(dataset)}")
        return dataset

    if name == "cub2011_custom":
        from models.datasets.cub200 import TextDataset
        dataset = TextDataset(data_dir='/path/to/cub200/birds', split='train' if is_train else 'test')
        return dataset

    wds_config = OmegaConf.create(
        {
            "train_data": None,
            "val_data": None,
            "dataset_type": "webdataset",
            "train_data_upsampling_factors": None,
            "batch_size": config.loader.batch_size if mode == "train" else config.loader.eval_batch_size,
            "workers": config.loader.num_workers,
            "distributed": True,
            "seed": config.seed,
            "val_num_samples": None,
            "train_num_samples": config.data.webdataset_train_num_samples,
            "val_num_samples": config.data.webdataset_val_num_samples,
            "world_size": config.trainer.devices * config.trainer.num_nodes,
            "block_size": block_size,
        }
    )
    if config.data.dataset_type == "webdataset":
        clean_brace_escape = lambda x: x.replace("[", "{").replace("]", "}")
        wds_config.train_data = clean_brace_escape(config.data.webdataset_train_data)
        wds_config.val_data = clean_brace_escape(config.data.webdataset_val_data)

        if getattr(config.data, "webdataset_prefix", None) is not None:
            wds_config.train_data = config.data.webdataset_prefix.replace("LITERALQUOTE", "'").replace("LITERALSPACE", " ") + wds_config.train_data
            wds_config.val_data = config.data.webdataset_prefix.replace("LITERALQUOTE", "'").replace("LITERALSPACE", " ") + wds_config.val_data

        if getattr(config.data, "webdataset_postfix", None) is not None:
            wds_config.train_data = wds_config.train_data + config.data.webdataset_postfix.replace("LITERALQUOTE", "'").replace("LITERALSPACE", " ")
            wds_config.val_data = wds_config.val_data + config.data.webdataset_postfix.replace("LITERALQUOTE", "'").replace("LITERALSPACE", " ")

        return get_data(wds_config, (transform, transform), epoch=0, tokenizer=tokenizer)
    if name == "laion400m":
        # TODO: Debug if these configs are correct!!!! Not fully sure how the webdataset sharded dataloader should work.
        wds_config.train_data = "/grogu/datasets/laion400m/dataset/{00000..00625}.tar"
        wds_config.val_data = "/grogu/datasets/laion400m/dataset/{00000..00625}.tar"
        return get_data(wds_config, (transform, transform), epoch=0, tokenizer=tokenizer)
    elif name == "cc12m_3m":
        # TODO: Debug if these configs are correct!!!! Not fully sure how the webdataset sharded dataloader should work.
        wds_config.train_data = config.data.raw_data_dir + "/cc3m-train-{0000..0575}.tar"
        wds_config.val_data = config.data.raw_data_dir + "/cc3m-validation-{0000..0015}.tar"
        return get_data(wds_config, (transform, transform), epoch=0, tokenizer=tokenizer)
    elif name == "facecaption":
        if getattr(config.data, "webdataset_iterable", False):
            wds_config.train_data = "/grogu/user/mprabhud/data/diffusion/facecaption/{00000..00001}.tar"
            wds_config.val_data = "/grogu/user/mprabhud/data/diffusion/facecaption/{00000..00001}.tar"
            return get_data(wds_config, (transform, transform), epoch=0, tokenizer=tokenizer)
        elif getattr(config.data, "webdataset_indexed", False) is False:
            return get_webdataset_indexed(config, tokenizer, transform, cond_transform, n_samples, name, should_tokenize=True)
        else:
            raise Exception("Unknown webdataset type")

    # hf webdataset
    if name == "pixparse/cc12m-wds":
        extra_kwargs["data_dir"] = config.data.raw_data_dir

    if name == "generated_images":
        extra_kwargs["data_files"] = {"train": getattr(config.data, "parquet_path", None)}

    if name != "imagefolder":
        rprint(f"Loading dataset {name}, split={split}, streaming={streaming}, cache_dir={cache_dir}, extra_kwargs={extra_kwargs}, dynamic={dynamic}")

    load_map = {"pixparse/cc12m-wds": "webdataset", "laion400m": "webdataset", "generated_images": "parquet"}
    load_name = load_map.get(name, name)
    if streaming is False:
        extra_kwargs["num_proc"] = 16

    if config.data.tokenize_vqvae_in_dataloader:
        full_cache_path = get_vqvae_dataloader(config, name, split)
        _ret = load_vqvae_from_cache(config, full_cache_path)
        if _ret is not None: return _ret
        from model import get_image_batch, get_vae
        rank = get_rank()
        vae = get_vae(config, device="cpu").eval()
        vae.to(f"cuda:{rank}")

        def tokenize_vqvae(batch):
            device = f"cuda:{rank}"
            img_input_ids = get_image_batch(config, vae, batch, device)
            batch.pop("img")
            batch["img_input_ids"] = img_input_ids
            return batch

    if config.data.keep_hf_dataset_in_memory:
        extra_kwargs["keep_in_memory"] = True
        gprint(f"WARNING!!!! Keeping dataset in memory")

    if name == "geneval":
        def create_blank_image():
            return Image.new("RGB", (resolution, resolution), color=(255, 255, 255))

        # https://github.com/djghosh13/geneval/blob/main/prompts/generation_prompts.txt
        prompts_path = Path.home() / ".cache" / "unidisc" / "geneval_generation_prompts.txt"
        if not prompts_path.exists():
            prompts_path.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/djghosh13/geneval/main/prompts/generation_prompts.txt",
                prompts_path
            )
        with open(prompts_path, "r") as f:
            captions = [line.strip() for line in f.readlines()]

        dataset = Dataset.from_dict({
            "caption": captions,
            "image": [
                create_blank_image() for i in range(len(captions))
            ],
        })
    elif name == "MJHQ":
        def create_blank_image():
            return Image.new("RGB", (resolution, resolution), color=(255, 255, 255))
        prompts_path = Path.home() / ".cache" / "unidisc" / "MJHQ_meta_data.json"
        if not prompts_path.exists():
            prompts_path.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(
                "https://huggingface.co/datasets/playgroundai/MJHQ-30K/resolve/main/meta_data.json",
                prompts_path
            )
        
        with open(prompts_path, "r") as f:
            data = json.load(f)
            captions = [item["prompt"] for item in data.values()]

        dataset = Dataset.from_dict({
            "caption": captions,
            "image": [
                create_blank_image() for i in range(len(captions))
            ],
        })
    else:
        dataset = datasets.load_dataset(load_name, split=split, streaming=streaming, cache_dir=cache_dir, **extra_kwargs)

    dataset_keys = {
        "cassiekang/cub200_dataset": ("image", "text"),
        "Andron00e/CUB200-custom": ("image",),
        "nlphuji/flickr30k": ("image", "caption"),
        "ILSVRC/imagenet-1k": ("image", "label"),
        "richwardle/reduced-imagenet": ("image",),
        "tglcourse/lsun_church_train": ("image",),
        "imagefolder": ("image",),
        "pixparse/cc12m-wds": ("jpg", "txt"),
        "pravsels/FFHQ_1024": ("image",),
        "pravsels/SFHQ_256": ("image",),
        "jxie/celeba-hq": ("image",),
        "tglcourse/lsun_church_train": ("image",),
        "pouya-haghi/imagenet-subset": ("image",),
        "DeepLearner101/ImageNetSubsetValidate": ("image",),
        "PixArt-alpha/SAM-LLaVA-Captions10M": ("__key__", "txt"),
        "generated_images": ("__key__", "caption"),
        "laion/clevr-webdataset": ("jpg","txt"),
        "pcuenq/lsun-bedrooms": ("image",),
        "facebook/winoground": ("image_0", "image_1", "caption_0", "caption_1"),
        "sayakpaul/coco-30-val-2014": ("image", "caption"),
        "geneval": ("image", "caption"),
        "MJHQ": ("image", "caption"),
    }

    from dataloader import tokenize_text

    def preprocess_images(example, index: typing.Optional[typing.Any] = None):
        data = {}
        if dataset_keys[name][0] == "__key__":
            images = []
            is_valid = []
            for key, _image_path in zip(example[dataset_keys[name][0]], example["image_path"]):
                img_path = (
                    (Path(config.data.raw_data_dir) / key).with_suffix(".jpg") if not key.endswith(".jpg") else (Path(config.data.raw_data_dir) / key)
                )
                allow_relative = False
                if Path(_image_path).exists() and Path(_image_path).stat().st_size > 0:
                    img = Image.open(_image_path)
                    is_valid.append(True)
                elif allow_relative and img_path.exists() and img_path.stat().st_size > 0:
                    img = Image.open(img_path)
                    is_valid.append(True)
                else:
                    img = Image.new("RGB", (resolution, resolution), color=(255, 255, 255))
                    is_valid.append(False)
                images.append(img)
            data["is_valid"] = is_valid
            if sum(data["is_valid"]) < len(data["is_valid"]):
                gprint(f"WARNING!!! Found {len(data['is_valid']) - sum(data['is_valid'])} invalid images")
        else:
            images = [image.convert("RGB") for image in example[dataset_keys[name][0]]]

        data["img"] = [transform(image) for image in images]
        if cond_resolution is not None:
            data["cond_img"] = [cond_transform(image) for image in images]

        if index is not None:
            data["idx"] = index

        if "idx" in example:
            data["idx"] = example["idx"]

        if dynamic and dataset_keys[name][0] is not None:
            data["img"] = torch.stack(data["img"])

        if "label" in example:
            data["label"] = example["label"]
        if (unified_model or getattr(config.data, "txt_only", False)) and not getattr(config.data, "disable_text_modality", False):
            tokenizer.padding_side = "right"
            tokenizer.truncation_side = "right"
            
            if name == "facebook/winoground":
                caption_0 = example["caption_0"]
                caption_1 = example["caption_1"]
                img_0 = example["image_0"]
                img_1 = example["image_1"]
                # tokenize and store captions separately
                tokens_0 = tokenize_text(tokenizer, block_size, caption_0)
                tokens_1 = tokenize_text(tokenizer, block_size, caption_1)
                data["caption_0_input_ids"] = tokens_0["input_ids"]
                data["caption_0_attention_mask"] = tokens_0["attention_mask"].float()
                data["caption_1_input_ids"] = tokens_1["input_ids"]
                data["caption_1_attention_mask"] = tokens_1["attention_mask"].float()
                # convert img_0 and img_1 which are lists of PIL images to tensors
                # convert some rgba pil images to rgb
                data["img_0"] = torch.stack([transform(img.convert("RGB")) for img in img_0])
                data["img_1"] = torch.stack([transform(img.convert("RGB")) for img in img_1])
            else:
                text_data = example[dataset_keys[name][1]]
                if isinstance(text_data[0], list):
                    # Flickr has a list of captions for each image
                    text_data = [random.choice(_data) for _data in text_data]

                tokens = tokenize_text(tokenizer, block_size, text_data)
                data["input_ids"] = tokens["input_ids"]
                data["attention_mask"] = tokens["attention_mask"].float()

        return data

    if precache is False:
        tokenized_dataset = dataset.with_transform(preprocess_images)
    else:
        extra_kwargs = dict()
        if streaming is False:
            extra_kwargs["load_from_cache_file"] = True
        else:
            if name == "pixparse/cc12m-wds":
                extra_kwargs["remove_columns"] = ["__key__", "jpg", "__url__", "json", "txt"]
            elif name == "ILSVRC/imagenet-1k":
                extra_kwargs["remove_columns"] = ["image"]

        tokenized_dataset = dataset.map(preprocess_images, batched=True, with_indices=True, **extra_kwargs)
        allowed_column_names = ["img", "input_ids", "attention_mask", "tokens", "text", "idx"]
        current_column_names = tokenized_dataset.column_names
        if current_column_names is not None:
            for column_name in current_column_names:
                if column_name not in allowed_column_names:
                    tokenized_dataset = tokenized_dataset.remove_columns(column_name)

    if n_duplicate is not None:
        tokenized_dataset = datasets.concatenate_datasets([tokenized_dataset] * n_duplicate)

    ret = tokenized_dataset if dynamic else tokenized_dataset.with_format("torch")
    if isinstance(dataset, torch.utils.data.IterableDataset) or "cc12m" in name:
        ret = ResilientIterableDatasetWrapper(ret)

    if config.data.tokenize_vqvae_in_dataloader:
        assert config.data.force_mp_spawn
        ret = ret.shard(num_shards=get_world_size(), index=get_rank(), contiguous=True, keep_in_memory=True)
        gprint(f"Rank {rank} has {len(ret)} samples. World size is {get_world_size()}")
        ret = ret.map(tokenize_vqvae, batch_size=getattr(config.data, "vqvae_batch_size", 128), batched=True, keep_in_memory=True)
        ret.reset_format()
        allowed_column_names = ["img_input_ids"]
        map_column_list = getattr(config.data, "map_columns", None)
        if map_column_list is not None:
            for old_column_name, new_column_name in map_column_list.items():
                ret = ret.rename_column(old_column_name, new_column_name)
        if getattr(config.data, "allow_label", False):
            allowed_column_names.append("label")
        if getattr(config.data, "allowed_columns_vqvae_dataloader", None):
            allowed_column_names.extend(list(config.data.allowed_columns_vqvae_dataloader))
        current_column_names = ret.column_names
        if current_column_names is not None:
            for column_name in current_column_names:
                if column_name not in allowed_column_names:
                    ret = ret.remove_columns(column_name)
        rank_cache_path = full_cache_path / f"rank_{rank}"
        gprint(f"Rank {rank} has saved to {rank_cache_path} with {len(ret)} samples")
        ret.save_to_disk(rank_cache_path)
        barrier()
        gprint(f"Rank {rank} has finished saving to {rank_cache_path}. Sleeping for a bit. You may want to Ctrl+C now")
        time.sleep(60 * 30)
        ret = load_vqvae_from_cache(config, full_cache_path)
        gprint(f"Rank {rank} has finished loading from file: {rank_cache_path}")

    return ret
