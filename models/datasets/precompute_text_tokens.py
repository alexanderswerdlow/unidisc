import os
import shutil
import signal
import sys
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path

from accelerate.utils import gather_object
from torchinfo import summary

from unidisc.tokenizers.chameleon_tokenizers import tokenize_chameleon
from utils import _print_config, set_numa_affinity, set_omega_conf_resolvers

sys.path.append(str(Path(__file__).parent.parent.parent / "unidisc/misc/hydra_submitit_launcher"))
import itertools
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
from viztracer import VizTracer

from dataloader import get_dataloaders, get_tokenizer, tokenize_text
from decoupled_utils import (barrier, breakpoint_on_error, get_world_size,
                             is_local_main_process, is_main_process,
                             rank_zero_fn, rprint, set_global_breakpoint,
                             set_global_exists, gprint)
from model import decode_latents, get_image_batch, get_vae
from models.datasets.combine_token_dicts import main as combine_token_dicts
from models.datasets.vggface_v2_attributes import (get_inference_func,
                                                   get_output)
from utils import (_print_config, set_numa_affinity, set_omega_conf_resolvers,
                   set_torch_defaults)

os.environ["HYDRA_FULL_ERROR"] = "1"

set_global_breakpoint()  # Overrides breakpoint() to use ipdb.set_trace() instead and handle distributed training
set_global_exists()
set_omega_conf_resolvers()
set_torch_defaults()

def get_dict(config, dataset_size):
    data = TensorDict(
        {
            "input_ids": torch.zeros(dataset_size, config.model.img_length, dtype=torch.int16),
            "idx": torch.full((dataset_size, 1), fill_value=-1, dtype=torch.int32),
            "write_flag": torch.zeros(dataset_size, 1, dtype=torch.bool),
            "modality": torch.full((dataset_size, 1), fill_value=-1, dtype=torch.int16),
        },
        batch_size=[dataset_size],
    )
    return data

def _group_texts(examples, block_size, bos, eos):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  # TODO(yair): look into not dropping the remainder but rather padding it.
  # We drop the small remainder, and if the total_length < block_size - 2
  # we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of
  # this drop, you can customize this part to your needs.
  new_block_size = block_size  # [BOS] and [EOS] to be added
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    _data = concatenated_examples[i : i + new_block_size]
    _data[0] = bos
    _data[-1] = eos
    _values.append(_data)

  result['input_ids'] = _values

  # We don't have pad tokens when wrapped so we can ignore these
  # result['attention_mask'] = _attn_masks
  # result['modality'] = [[0] for _ in range(len(result['input_ids']))]

  return result

def preprocess_and_tokenize(example, tokenizer, dataset_name, wrap, block_size, EOS, BOS):
    if dataset_name == 'ptb':
        text = example['sentence']
    elif 'scientific_papers' in dataset_name:
        text = example['article']
    else:
        text = example['text']

    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    if wrap:
        tokens = tokenizer(text,
                            add_special_tokens=True,
                            return_attention_mask=False,
                            return_token_type_ids=False)
        tokens = {'input_ids': tokens['input_ids']}
        # Still missing BOS, but will be added in group_texts
    else:
        tokens = tokenizer(text,
                            max_length=block_size,
                            padding='max_length',
                            truncation=True,
                            add_special_tokens=True,
                            return_attention_mask=True,
                            return_token_type_ids=True)
    return tokens

def add_modality(output_dataset):
    modality_column = torch.zeros(len(output_dataset), 1, dtype=torch.long)
    output_dataset = output_dataset.add_column("modality", modality_column)
  
import datasets
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config):
    """Main entry point for training."""
    _print_config(config, resolve=True, save_cfg=True)
    tokenizer = get_tokenizer(config)
    block_size = config.data.block_size
    
    wrap = True
    streaming = config.data.streaming
    num_proc = config.data.num_proc
    split = getattr(config.data, "split", "train")
    use_cache = False

    assert getattr(config.data, "use_slow_tokenizer", False) is False

    output_dir = config.data.token_output_dir
    output_dir = Path(f"{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tensordict_output_dir = output_dir.parent / f"{output_dir.stem}_tensordict"
    tensordict_output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = config.data.train
    if isinstance(dataset_name, list):
        data = datasets.concatenate_datasets([
            datasets.load_dataset(name, split=split, cache_dir=config.data.cache_dir, streaming=streaming)
            for name in dataset_name
        ])
    else:
        _args = []
        if getattr(config.data, "add_load_dataset_args", None) is not None:
            _args.append(getattr(config.data, "add_load_dataset_args", None))
        data = datasets.load_dataset(dataset_name, *_args, split=split, cache_dir=config.data.cache_dir, streaming=streaming)
    
    
    EOS = tokenizer.eos_token_id
    BOS = tokenizer.bos_token_id

    if config.data.n_train_samples is not None:
        print(f"Selecting {config.data.n_train_samples} samples")
        data = data.select(range(config.data.n_train_samples))

    _preprocess_and_tokenize = partial(preprocess_and_tokenize, tokenizer=tokenizer, dataset_name=dataset_name, wrap=wrap, block_size=block_size, EOS=EOS, BOS=BOS)
    if streaming:
        tokenized_dataset = data.map(
            _preprocess_and_tokenize,
            batched=True
        )
    else:
        rprint(f"Tokenizing with num_proc: {num_proc}")
        tokenized_dataset = data.map(
            _preprocess_and_tokenize,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=use_cache,
            desc='Tokenizing')

    tokenized_dataset = tokenized_dataset.remove_columns('text')
    columns_to_keep = ['input_ids']
    if tokenized_dataset.column_names is not None:
        columns_to_remove = [col for col in tokenized_dataset.column_names if col not in columns_to_keep]
        tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
    
    output_dataset = None
    if wrap:
        group_texts =  partial(_group_texts, block_size=block_size, bos=BOS, eos=EOS)
        if streaming:
            chunked_dataset = tokenized_dataset.map(group_texts, batched=True)
        else:
            chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=use_cache, desc='Grouping')
            chunked_dataset.save_to_disk(output_dir)

        output_dataset = chunked_dataset.with_format('torch')
    else:
        if streaming is False:
            tokenized_dataset.save_to_disk(output_dir)
        output_dataset = tokenized_dataset.with_format('torch')

if __name__ == "__main__":
    with breakpoint_on_error():
        main()
