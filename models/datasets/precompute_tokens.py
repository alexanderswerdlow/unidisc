import os
import shutil
import signal
import sys
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path

from accelerate.utils import gather_object, gather
from torchinfo import summary

from unidisc.tokenizers.chameleon_tokenizers import tokenize_chameleon, tokenize_chameleon_fast, get_chameleon_images, decode_ids, decode_ids_batched, tokenize_chameleon_mmc4, tokenize_regular_cambrian_mmc4
from utils import _print_config, set_numa_affinity, set_omega_conf_resolvers

sys.path.append(str(Path(__file__).parent.parent.parent / "unidisc/misc/hydra_submitit_launcher"))

import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import fsspec
import hydra
import numpy as np
import omegaconf
import rich.syntax
import rich.tree
import torch
from accelerate import Accelerator
from PIL import Image
from tensordict import TensorDict
from tqdm import tqdm
try:
    from viztracer import VizTracer
except ImportError:
    print("VizTracer not installed, skipping")

from dataloader import get_dataloaders, get_tokenizer, tokenize_text
from decoupled_utils import (barrier, breakpoint_on_error, get_local_rank, get_rank, get_world_size,
                             is_local_main_process, is_main_process,
                             rank_zero_fn, rprint, set_global_breakpoint,
                             set_global_exists, gprint)
from model import decode_latents, get_image_batch, get_vae
from models.datasets.combine_token_dicts import main as combine_token_dicts
from models.datasets.vggface_v2_attributes import (get_inference_func,
                                                   get_output)
from utils import (_print_config, set_numa_affinity, set_omega_conf_resolvers,
                   set_torch_defaults)
from omegaconf import DictConfig, OmegaConf, open_dict, read_write

os.environ["HYDRA_FULL_ERROR"] = "1"

set_global_breakpoint()  # Overrides breakpoint() to use ipdb.set_trace() instead and handle distributed training
set_global_exists()
set_omega_conf_resolvers()
set_torch_defaults()

def get_batch_size(config):
    with open_dict(config):
        if any(x.lower() in torch.cuda.get_device_name().lower() for x in ["v100", "1080", "2080", "quadro", "titan"]) or torch.cuda.get_device_capability()[0] <= 7:
            config.trainer.precision = "no"
            config.model.force_optimized_native_attn = False
            config.trainer.compile = False
            config.loader.batch_size = config.loader.batch_size // 3
            print(f"Found {torch.cuda.get_device_name().lower()}, set batch size to {config.loader.batch_size}")
    return config

def enc(data, idx, encode_images, config, vae, batch, accelerator, mixed_precision, tokenizer, vgg_data, existing_ids=None, device=None, mapping=None):
    
    if isinstance(batch, list):
        bs = len(batch)
    elif "img" in batch:
        bs = batch["img"].shape[0]
    else:
        bs = batch["attention_mask"].shape[0]
    
    sl = slice(idx * bs, (idx + 1) * bs)
    if not isinstance(batch, list) and "idx" in batch:
        if set(data[sl]["idx"].flatten().tolist()) == set(batch["idx"].tolist()):
            rprint(f"Skipping {idx} as all samples have already been processed 1")
            return
        if existing_ids is not None:
            set_inter = set(batch["idx"].tolist()) & existing_ids
            if len(set_inter) == bs:
                rprint(f"Skipping {idx} as all samples have already been processed 2")
                return
            elif len(set_inter) > 0:
                rprint(f"Running {idx} as some samples have already been processed: {len(set_inter)}")
    else:
        if (data[sl]["idx"] != -1).all():
            rprint(f"Skipping {idx} as all samples have already been processed")
            return
    
    if not isinstance(batch, list) and "img" in batch:
        batch["img"] = batch["img"].to(device=device, dtype=torch.bfloat16 if mixed_precision else None)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
            use_chameleon = getattr(config.data, "use_chameleon", False)
            use_mmc4 = config.data.train == "mmc4"
            use_cambrian = config.data.train == "cambrian"
            if not use_chameleon and not use_mmc4 and not use_cambrian:
                if tokenizer is not None and getattr(config.model, "unified_model", False):
                    if "input_ids" in batch and "attention_mask" in batch:
                        tokens = batch
                    else:
                        tokens = tokenize_text(tokenizer, config.data.block_size, batch[".txt"])

                    batch["txt_input_ids"] = tokens["input_ids"]
                    batch["txt_attention_mask"] = tokens["attention_mask"].float()
                elif getattr(config.data, "add_vggface_v2_attributes", False) and "vggface" not in config.data.train:
                    txt_input_ids, txt_attention_mask = get_output(batch, **vgg_data)
                    batch["txt_input_ids"] = txt_input_ids
                    batch["txt_attention_mask"] = txt_attention_mask
                elif getattr(config.data, "txt_only", False):
                    batch["txt_input_ids"] = batch["input_ids"]
                    batch["txt_attention_mask"] = batch["attention_mask"]

            if getattr(config.model, "unified_model", False) is False:
                if getattr(config.data, "txt_only", False):
                    batch["modality"] = torch.full((bs, 1), fill_value=0, dtype=torch.int16)
                else:
                    batch["modality"] = torch.full((bs, 1), fill_value=1, dtype=torch.int16)

            if isinstance(batch, list) and batch[0].get("idx", None) is not None:
                _idx = torch.tensor([x["idx"] for x in batch], dtype=torch.int32).unsqueeze(-1)
            elif "idx" in batch:
                _idx = batch["idx"].to(torch.int32).unsqueeze(-1)
            else:
                _idx = torch.full((bs, 1), fill_value=0, dtype=torch.int32)

            if "is_valid" in batch:
                _idx[~batch["is_valid"]] = -1
                if (_idx == -1).all():
                    gprint(f"WARNING: All samples are invalid")

            sl = slice(idx * bs, (idx + 1) * bs)
            assert (idx + 1) * bs <= len(data), f"Index {idx} + batch size {bs} is greater than the data length {len(data)}"

            if encode_images:
                if use_chameleon:
                    if isinstance(batch, list):
                        all_input_ids, all_attention_masks = tokenize_chameleon_mmc4(config, tokenizer, vae, batch, device, mapping)
                    else:
                        all_input_ids, all_attention_masks = tokenize_chameleon_fast(config, tokenizer, vae, batch)

                    # all_input_ids_gt, all_attention_masks_gt = tokenize_chameleon(config, tokenizer, vae, batch)
                    # txt_tokens, img_tokens = decode_ids_batched(_vae, all_input_ids[:4], return_tokens=True)
                    # img = decode_latents(config, _vae, img_tokens)
                    # from image_utils import Im;  Im(img).save()

                elif use_mmc4 or use_cambrian:
                    all_input_ids, all_attention_masks, all_modality = tokenize_regular_cambrian_mmc4(config, tokenizer, vae, batch, device, mapping)
                    if all_input_ids is None:
                        return
                else:
                    image_ids = get_image_batch(config, vae, batch, device)

                if use_chameleon or use_mmc4 or use_cambrian:
                    if not use_chameleon:
                        assert (all_input_ids < torch.iinfo(torch.int16).max).all()
                    
                    _kwargs = {}
                    if use_mmc4 or use_cambrian:
                        _kwargs["modality"] = all_modality.to(torch.int8)

                    data[sl] = TensorDict(
                        {
                            "input_ids": all_input_ids.to(torch.int32 if use_chameleon else torch.int16),
                            "attention_mask": all_attention_masks.to(torch.bool),
                            "idx": _idx,
                            "write_flag": torch.ones((bs, 1), dtype=torch.bool),
                            **_kwargs,
                        },
                        batch_size=[bs],
                    )
                elif getattr(config.model, "cond_label", False):
                    data[sl] = TensorDict(
                        {
                            "img_input_ids": image_ids.to(torch.int16),
                            "img_label": batch["label"].to(torch.int32).unsqueeze(-1),
                            "idx": _idx,
                            "write_flag": torch.ones((bs, 1), dtype=torch.bool),
                        },
                        batch_size=[bs],
                    )
                elif getattr(config.model, "unified_model", False) or getattr(config.data, "add_vggface_v2_attributes", False):
                    data[sl] = TensorDict(
                        {
                            "img_input_ids": image_ids.to(torch.int16),
                            "txt_input_ids": (batch.get("txt_input_ids") if batch.get("txt_input_ids") is not None else batch["input_ids"]).to(
                                torch.int32
                            ),
                            "txt_attention_mask": (
                                batch.get("txt_attention_mask") if batch.get("txt_attention_mask") is not None else batch["attention_mask"]
                            ).to(torch.bool),
                            "idx": _idx,
                            "write_flag": torch.ones((bs, 1), dtype=torch.bool),
                        },
                        batch_size=[bs],
                    )
                else:
                    data[sl] = TensorDict(
                        {"input_ids": image_ids.to(torch.int32), "attention_mask": torch.ones((image_ids.shape[0], image_ids.shape[1]), dtype=torch.bool), "idx": _idx, "write_flag": torch.ones((bs, 1), dtype=torch.bool), "modality": batch["modality"].to(torch.int16)},
                        batch_size=[bs],
                    )

            elif getattr(config.data, "txt_only", False):
                data[sl] = TensorDict(
                    {"input_ids": batch['input_ids'].to(torch.int32), "attention_mask": batch['attention_mask'].to(torch.bool), "idx": _idx, "write_flag": torch.ones((bs, 1), dtype=torch.bool), "modality": batch["modality"].to(torch.int16)},
                    batch_size=[bs],
                )
            else:
                real_image = batch["img"]
                if (config.data.resolution == 512 and batch["img"].shape[0] > 16) or (config.model.downscale_ratio <= 8):
                    chunk_size = 8 if (config.model.image_vocab_size > 64000 or config.model.downscale_ratio <= 8) else 16
                    chunks = [batch["img"][i : i + chunk_size] for i in range(0, batch["img"].shape[0], chunk_size)]
                    rec_img_list = []
                    for chunk in chunks:
                        batch_chunk = {"img": chunk}
                        image_ids = get_image_batch(config, vae, batch_chunk, device)
                        rec_img = decode_latents(config, vae, image_ids)
                        rec_img_list.append(rec_img)
                    rec_img = torch.cat(rec_img_list, dim=0)
                else:
                    image_ids = get_image_batch(config, vae, batch, device)
                    rec_img = decode_latents(config, vae, image_ids)

                viz_img = torch.cat([real_image, rec_img], dim=-1)
                from image_utils import Im
                
                if getattr(config.model, 'custom_vae_name', None) is not None:
                    custom_str = getattr(config.model, 'custom_vae_name')
                else:
                    custom_str = f"{'_custom' if getattr(config.model, 'use_custom_vae_ckpt', False) else ''}"
                (Path(__file__).parent.parent.parent / "output").mkdir(parents=True, exist_ok=True)
                Im(viz_img).save(
                    Path(__file__).parent.parent.parent / f"output/{config.data.train.replace('/', '')}_seq{image_ids.shape[1]}_res{config.data.resolution}_{config.model.vae_type}{custom_str}_voc{config.model.image_vocab_size}.png"
                )
                
                # Create directories for saving images
                dataset_name = config.data.train.replace('/', '')
                vae_name = f"seq{image_ids.shape[1]}_res{config.data.resolution}_{config.model.vae_type}{custom_str}_voc{config.model.image_vocab_size}"
                output_dir = Path(__file__).parent.parent.parent / "output" / dataset_name / vae_name
                gt_output_dir = Path(__file__).parent.parent.parent / "output" / dataset_name / f"GT_{config.data.resolution}"
                output_dir.mkdir(parents=True, exist_ok=True)
                gt_output_dir.mkdir(parents=True, exist_ok=True)

                # Save each image separately
                for i, (real, rec) in enumerate(zip(real_image, rec_img)):
                    print(Im(rec).save(output_dir / f"{i}.png"))
                    if (gt_output_dir / f"{i}.png").exists() is False:
                        print(Im(real).save(gt_output_dir / f"{i}.png"))

                gprint(f"Exiting")
                exit()


def get_dict(config, dataset_size):
    if getattr(config.data, "use_chameleon", False) or config.data.train == "cambrian" or config.data.train == "mmc4":
        input_ids_dtype = torch.int32 if getattr(config.data, "use_chameleon", False) else torch.int16
        data = TensorDict(
            {
                "input_ids": torch.zeros(dataset_size, config.model.length, dtype=input_ids_dtype),
                "attention_mask": torch.zeros(dataset_size, config.model.length, dtype=torch.bool),
                "modality": torch.full((dataset_size, config.model.length), fill_value=-1, dtype=torch.int8),
                "idx": torch.full((dataset_size, 1), fill_value=-1, dtype=torch.int32),
                "write_flag": torch.zeros(dataset_size, 1, dtype=torch.bool),
            },
            batch_size=[dataset_size],
        )
    elif getattr(config.model, "cond_label", False):
        data = TensorDict(
            {
                "img_input_ids": torch.zeros(dataset_size, config.model.img_length, dtype=torch.int16),
                "img_label": torch.zeros(dataset_size, 1, dtype=torch.int32),
                "idx": torch.full((dataset_size,), fill_value=-1, dtype=torch.int32),
                "write_flag": torch.zeros(dataset_size, 1, dtype=torch.bool),
            },
            batch_size=[dataset_size],
        )
    elif getattr(config.model, "unified_model", False) or getattr(config.data, "add_vggface_v2_attributes", False):
        data = TensorDict(
            {
                "img_input_ids": torch.zeros(dataset_size, config.model.img_length, dtype=torch.int16),
                "txt_input_ids": torch.zeros(dataset_size, config.model.txt_length, dtype=torch.int32),
                "txt_attention_mask": torch.zeros(dataset_size, config.model.txt_length, dtype=torch.bool),
                "idx": torch.full((dataset_size, 1), fill_value=-1, dtype=torch.int32),
                "write_flag": torch.zeros(dataset_size, 1, dtype=torch.bool),
            },
            batch_size=[dataset_size],
        )
    else:
        data = TensorDict(
            {
                "input_ids": torch.zeros(dataset_size, config.model.txt_length if config.data.txt_only else config.model.img_length, dtype=torch.int16),
                "idx": torch.full((dataset_size, 1), fill_value=-1, dtype=torch.int32),
                "write_flag": torch.zeros(dataset_size, 1, dtype=torch.bool),
                "modality": torch.full((dataset_size, 1), fill_value=-1, dtype=torch.int16),
            },
            batch_size=[dataset_size],
        )
    return data

def signal_handler(signum, frame, train_data, tmp_path):
    """Handle signals to save temporary train data."""
    rprint(f"Received signal {signum}, saving temporary train data.")
    print(f"[PRINT] Received signal {signum}, saving temporary train data.")
    save_tmp_data(train_data, tmp_path)
    sys.exit

def save_tmp_data(data, tmp_path):
    """Save data to a temporary path."""
    if tmp_path.exists() and tmp_path.is_dir():
        rprint(f"Deleting {tmp_path}")
        shutil.rmtree(tmp_path)  # Delete old tmp directory if it exists
    rprint(f"Saving tmp data to {tmp_path}")
    data.memmap(tmp_path, copy_existing=True)

def periodic_save(data, tmp_path, start_time, interval=2 * 60 * 60):
    """Periodically save data to a temporary path."""
    current_time = time.time()
    if current_time - start_time >= interval:
        rprint(f"Hit periodic save interval, saving tmp data to {tmp_path}")
        save_tmp_data(data, tmp_path)
        return current_time  # Reset start time
    return start_time

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config):
    """Main entry point for training."""
    
    try:
        import resource
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit)) # Set the soft limit to the hard limit
        rprint(f"Successfully set RLIMIT_NOFILE to {hard_limit}")
    except Exception as e:
        rprint(f"Failed to set RLIMIT_NOFILE: {e}")

    mixed_precision = False
    train_start_time = time.time()

    from datetime import timedelta
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.utils import InitProcessGroupKwargs
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    prepare_kwargs = {}
    if config.data.train == "mmc4":
        prepare_kwargs["dispatch_batches"] = False

    accelerator = Accelerator(mixed_precision="bf16" if mixed_precision else None, kwargs_handlers=[kwargs], dataloader_config=DataLoaderConfiguration(**prepare_kwargs))
    device = torch.device(f"cuda:{accelerator.local_process_index}")

    import socket
    hostname = socket.gethostname()
    print(f"Hostname: {hostname}, Process index: {accelerator.process_index}, {device}, local_process_index: {accelerator.local_process_index}, get_local_process_index: {get_local_rank()}, device: {device}")
    _print_config(config, resolve=True, save_cfg=True)

    config = get_batch_size(config)

    # with omegaconf.open_dict(config):
    #     batch_sizes = gather_object([config.loader.batch_size])
    #     rprint(f"Batch sizes: {batch_sizes}")
    #     smallest_batch_size = min(batch_sizes)
    #     config.loader.batch_size = smallest_batch_size
    #     rprint(f"New config batch size: {config.loader.batch_size}")

    prefix = f"[Rank {accelerator.process_index}/{accelerator.num_processes}, Node: {os.environ.get('SLURM_NODEID', 'N/A')}, Hostname: {os.environ.get('SLURM_JOB_NODELIST', 'N/A')}, {config.data.train}]"
    print(f"{prefix} Starting precomputing tokens")
    save_validation_dataloader = getattr(config.data, "save_validation_dataloader", False)
    save_train_dataloader = getattr(config.data, "save_train_dataloader", False)

    tokenizer = get_tokenizer(config)
    train_dataloader, val_dataloader = get_dataloaders(
        config, tokenizer=tokenizer, allow_aug=False, force_aug=getattr(config.data, "force_aug", False), skip_valid=not save_validation_dataloader
    )

    train_dataloader = accelerator.prepare(train_dataloader)
    if save_validation_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)
    encode_images = getattr(config.model, "encode_images", False)

    use_chameleon = getattr(config.data, "use_chameleon", False)
    use_mmc4 = config.data.train == "mmc4"
    use_cambrian = config.data.train == "cambrian"
    mapping = None

    if use_chameleon:
        from unidisc.tokenizers.chameleon_tokenizers import ItemProcessor
        vae = ItemProcessor(target_size=config.data.resolution)
    else:
        vae = get_vae(config, device)

    if use_mmc4:
        import pandas as pd
        mapping = pd.read_parquet(config.data.mmc4_mapping_parquet)
        # Keep tar_filepath if it exists, otherwise use shard_path or map img2dataset_shard_id
        if "tar_filepath" in mapping.columns:
            pass
        elif "shard_path" in mapping.columns:
            mapping = mapping.rename(columns={"shard_path": "tar_filepath"})
            mapping["tar_filepath"] = mapping["tar_filepath"].str.replace(".parquet", ".tar")
        else:
            tar_path = Path(config.data.mmc4_tar_path)
            mapping["tar_filepath"] = mapping["img2dataset_shard_id"].apply(lambda x: tar_path / f"{x}.tar")
        
        mapping = mapping[['url', 'tar_filepath', 'key']]
        mapping = mapping.set_index("url").sort_index()

    if use_mmc4 or use_cambrian:
        assert config.data.use_slow_tokenizer and config.data.add_image_token

    if config.data.iterable:
        train_dataset_size = getattr(config.data, "train_dataset_size", None)
    else:
        print(f"{prefix} Train dataloader: {len(train_dataloader)} batches")
        print(f"{prefix} Train underlying dataset: {len(train_dataloader.dataset)} samples")
        train_dataset_size = (len(train_dataloader.dataset) // accelerator.num_processes) + config.loader.batch_size
        if save_validation_dataloader:
            print(f"{prefix} Val dataloader: {len(val_dataloader)} batches")
            print(f"Val underlying dataset: {len(val_dataloader.dataset)} samples")
            val_dataset_size = (len(val_dataloader.dataset) // accelerator.num_processes) + config.loader.batch_size

    print(f"{prefix} Train dataset size: {train_dataset_size} for 1 GPU")
    if save_validation_dataloader:
        print(f"{prefix} Val dataset size: {val_dataset_size} for 1 GPU")

    rank = accelerator.process_index
    output_dir = config.data.token_output_dir
    output_dir = Path(f"{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    assert config.data.force_disable_shuffle

    debug = getattr(config.data, "debug", False)
    print(f"{prefix} Output dir: {output_dir}")

    vgg_data = None
    if getattr(config.data, "add_vggface_v2_attributes", False):
        print(f"{prefix} Adding VGGFace V2 attributes")
        vgg_data = get_inference_func()
        vgg_data["model"] = accelerator.prepare(vgg_data["model"])
    
    if not config.data.split_dataset and is_main_process() and any(output_dir.iterdir()):
        rprint(f"Found temporary directories in output dir, combining them")
        combine_token_dicts(output_dir, use_tmp=False, use_timestamp=True, delete_after_combining=True)
        for item in output_dir.iterdir():
            if item.is_dir() and "tmp" in item.name:
                rprint(f"Removing temporary directory: {item}")
                shutil.rmtree(item)

    # barrier() # TODO: Should be a barrier here
    if not config.data.split_dataset:
        existing_folders = sorted([folder for folder in output_dir.iterdir() if folder.is_dir() and "existing" in folder.name])
        if existing_folders:
            rprint(f"Found existing folders: {existing_folders}")
            existing_data = torch.cat([TensorDict.load_memmap(folder) for folder in existing_folders], dim=0)
            rprint(f"Concatenated existing data with shape: {existing_data.shape}")
            existing_ids = set(existing_data["idx"].to(torch.int32).flatten().tolist())
        else:
            rprint("No existing folders found")
            existing_ids = None
    else:
        existing_ids = None
    
    if save_train_dataloader:
        if not config.data.split_dataset and getattr(config.data, "allow_load_from_tmp", True) and Path(output_dir / f"tmp_train_{rank}").exists():
            rprint("Found tmp_train_{rank} in output dir, loading from it")
            train_data = TensorDict.load_memmap(output_dir / f"tmp_train_{rank}")
            train_data = train_data.clone()
        else:
            train_data = get_dict(config, train_dataset_size)

        print(f"{prefix} Starting train dataloader")
        if config.data.split_dataset:
            rank = int(os.getenv("SLURM_ARRAY_TASK_ID"))
            print(f"Using task id: {rank}")

        split_path = output_dir / f"train_{rank}"
        tmp_train_path = output_dir / f"tmp_train_{rank}"

        signal.signal(signal.SIGUSR1, partial(signal_handler, train_data=train_data, tmp_path=tmp_train_path))
        signal.signal(signal.SIGUSR2, partial(signal_handler, train_data=train_data, tmp_path=tmp_train_path))

        try:
            signal.signal(signal.SIGKILL, partial(signal_handler, train_data=train_data, tmp_path=tmp_train_path))
        except:
            rprint(f"Failed to set SIGKILL handler")

        start_time = time.time()
        with VizTracer(output_file="optional.json", tracer_entries=5000000) if debug else nullcontext():
            for i, batch in tqdm(enumerate(train_dataloader), leave=False, disable=not is_local_main_process()):
                if i == 0 and "img" in batch:
                    print(f"Batch shape: {batch['img'].shape}")
                if debug and i >= 1:
                    break
                enc(train_data, i, encode_images, config, vae, batch, accelerator, mixed_precision, tokenizer, vgg_data=vgg_data, existing_ids=existing_ids, device=device, mapping=mapping)
                try:
                    if not config.data.split_dataset or True:
                        start_time = periodic_save(train_data, tmp_train_path, start_time, getattr(config.data, "periodic_save", 2 * 60 * 60))
                except Exception as e:
                    gprint(f"Failed to save train data: {e}")
                    start_time = time.time()

        if debug:
            exit()

        del train_dataloader
        print(f"{prefix} Saving train data")
        if split_path.exists() and split_path.is_dir():
            rprint(f"Removing {split_path}")
            shutil.rmtree(split_path)

        split_path.mkdir(parents=True, exist_ok=True)
        gprint(f"Saving train data to {split_path}: {train_data.shape}")
        train_data.memmap(split_path, copy_existing=True)

        if tmp_train_path.exists() and tmp_train_path.is_dir():
            rprint(f"Removing {tmp_train_path}")
            shutil.rmtree(tmp_train_path)

        if not config.data.split_dataset:
            with open(output_dir / f"train_{rank}.completed", 'w') as f:
                f.write(f"Processing done for rank {rank}\n")

        print(f"{prefix} Finished train dataloader")

    if save_validation_dataloader:
        val_data = get_dict(config, val_dataset_size)
        split_path = output_dir / f"val_{rank}"
        split_path.mkdir(parents=True, exist_ok=True)
        tmp_val_path = output_dir / f"tmp_val_{rank}"
        print(f"Starting val dataloader")
        start_time = time.time()  # Track start time for periodic saving
        for i, batch in tqdm(enumerate(val_dataloader), leave=False):
            if debug and i >= 10:
                break
            enc(val_data, i, encode_images, config, vae, batch, accelerator, mixed_precision, tokenizer, vgg_data=vgg_data, device=device)
            
            # Periodically save data
            start_time = periodic_save(val_data, tmp_val_path, start_time)

        print(f"{prefix} Saving val data")
        if split_path.exists() and split_path.is_dir():
            rprint(f"Removing {split_path}")
            shutil.rmtree(split_path)
        split_path.mkdir(parents=True, exist_ok=True)
        rprint(f"Saving val data to {split_path}")
        val_data.memmap(split_path, copy_existing=True)
        if tmp_val_path.exists() and tmp_val_path.is_dir():
            shutil.rmtree(tmp_val_path)  # Delete tmp directory after final save
        print(f"{prefix} Finished val dataloader")

    rprint(f"{prefix} Finished precomputing tokens")

    if config.data.split_dataset:
        rprint(f"We are splitting the dataset and thus exiting.")
        exit()

    if get_world_size() > 1 and (time.time() - train_start_time) > 60 * 60:
        time.sleep(60 * 60)
        barrier()

    rprint('after barrier')
    if is_main_process():
        combine_token_dicts(data_dir=output_dir, allow_zero_idx=True, move_files=True, delete_after_combining=True)

    barrier()
    rprint(f"Finished concating tokens")


if __name__ == "__main__":
    with breakpoint_on_error():
        main()
