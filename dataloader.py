import math
import typing
from pathlib import Path

import tokenizers
import torch
import transformers
from unidisc.datasets.sampler import WeightedDatasetSampler

from models.datasets.image_datasets import TensorCollate, get_image_dataset, get_unpaired_dataset
from models.datasets.text_datasets import Text8Tokenizer, get_text_dataset
from torch.utils.data import default_collate
from decoupled_utils import breakpoint_on_error, gprint, rprint, is_torch_xla_available
from datasets import load_dataset


def identity(x):
    return x


def get_dataset(dataset_name, tokenizer, *args, config=None, **kwargs):
    rprint(f"getting dataset {dataset_name}")
    if getattr(config.data, "unpaired", False):
        return get_unpaired_dataset(config=config, tokenizer=tokenizer, **kwargs)
    elif getattr(config.model, "image_model", False) or getattr(config.data, "force_image_dataset", False):
        return get_image_dataset(config=config, tokenizer=tokenizer, **kwargs)
    else:
        rprint(f"getting text dataset")
        return get_text_dataset(dataset_name, tokenizer, *args, **kwargs)

def tokenize_text(tokenizer, block_size, text, return_token_type_ids=True):
    return tokenizer(text, max_length=block_size, padding="max_length", truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=return_token_type_ids).convert_to_tensors("pt")

def get_tokenizer(config):
    if config.data.tokenizer_name_or_path is None or config.data.tokenizer_name_or_path == "None":
        return None
    elif config.data.tokenizer_name_or_path == "text8":
        tokenizer = Text8Tokenizer()
    elif config.data.tokenizer_name_or_path == "bert-base-uncased":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        tokenizer_kwargs = dict()
        if config.data.tokenizer_name_or_path == "NousResearch/Llama-2-7b-hf":
            tokenizer_kwargs["add_eos_token"] = True
            tokenizer_kwargs["padding_side"] = 'right'
            rprint("Using Llama tokenizer, adding add_eos_token and setting padding_side to right")
        if getattr(config.data, "use_slow_tokenizer", False):
            tokenizer_kwargs["use_fast"] = False
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.data.tokenizer_name_or_path, **tokenizer_kwargs)

        if getattr(config.data, "add_image_token", False):
            special_token = '<image>'
            existing_id = 811
            tmp_index = len(tokenizer)
            tokenizer.add_special_tokens({
                    'additional_special_tokens': [special_token]
            }, replace_additional_special_tokens=False)
            tokenizer._added_tokens_decoder[existing_id] = tokenizer._added_tokens_decoder.pop(tmp_index)
            assert len(tokenizer.additional_special_tokens_ids) == 1
            tokenizer.additional_special_tokens_ids = [existing_id]
            tokenizer._added_tokens_encoder['<image>'] = existing_id
            tokenizer.total_vocab_size = tmp_index
            
    if isinstance(tokenizer, transformers.GPT2TokenizerFast) or isinstance(tokenizer, transformers.GPT2Tokenizer):
        tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
            (tokenizer.bos_token, tokenizer.bos_token_id), (tokenizer.eos_token, tokenizer.eos_token_id)
        )

    # For wrapped batches:
    #  [BOS] sent1 [EOS] sent2-fragment [EOS]
    #  [BOS] sent2-fragment [EOS] sent3 [EOS]
    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError("Tokenizer must have a bos_token or " f"cls_token: {tokenizer}")
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError("Tokenizer must have a eos_token " f"or sep_token: {tokenizer}")
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        if config.data.tokenizer_name_or_path != "gpt2":
            rprint(f"Adding pad token to tokenizer")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    assert tokenizer.padding_side == 'right'
    assert tokenizer.truncation_side == 'right'

    return tokenizer


class SimpleDataLoader:
    def __init__(self, dataset, batch_size=1, collate_fn=default_collate, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < len(self.dataset):
            batch = []
            for _ in range(self.batch_size):
                if self.idx >= len(self.dataset):
                    break
                batch.append(self.dataset[self.idx])
                self.idx += 1
            return self.collate_fn(batch)
        else:
            raise StopIteration

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
def get_zero_shot_dataloader(config, tokenizer, device=None, **kwargs):
    if config.data.zero_shot_eval_dataset is None:
        rprint("No zero shot eval dataset provided")
        return None, None

    dataset_name = config.data.zero_shot_eval_dataset
    dataloader_seed = config.seed if config.mode == "eval" else 42
    if dataset_name == "nlphuji/flickr30k":
        data = load_dataset(dataset_name, num_proc=config.data.num_proc, cache_dir=config.data.cache_dir, streaming=config.data.streaming)
        dataset = data["test"]
    elif dataset_name == "facebook/winoground":
        data = load_dataset(dataset_name, num_proc=config.data.num_proc, cache_dir=config.data.cache_dir, streaming=config.data.streaming)
        dataset = data["test"]
    breakpoint()
    dl_cls = torch.utils.data.DataLoader
    valid_loader = dl_cls(
        dataset,
        batch_size=config.loader.eval_batch_size,
        num_workers=config.loader.num_eval_workers,
        pin_memory=config.loader.pin_memory,
        generator=torch.Generator().manual_seed(dataloader_seed),
        persistent_workers=False,
        **kwargs,
    )
    valid_loader.tokenizer = tokenizer
    return valid_loader


def get_dataloaders(config, tokenizer, skip_train=False, skip_valid=False, valid_seed=None, device=None, **kwargs):
    if skip_train:
        train_set = None
    else:
        _mode = getattr(config.data, "force_train_mode", "train")
        if _mode != "train":
            rprint(f"Forcing train mode to {_mode}")
        train_set = get_dataset(
            config.data.train,
            tokenizer,
            mode=_mode,
            wrap=config.data.wrap,
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            num_proc=config.data.num_proc,
            streaming=config.data.streaming,
            config=config,
            **kwargs,
        )
        if hasattr(train_set, '__len__'):
            rprint(f"Training set len: {len(train_set)}")

    if config.data.valid in ["text8", "lm1b", "ag_news"]:
        validation_split = "test"
    else:
        validation_split = "validation"
        
    if skip_valid:
        valid_set = None
    else:
        valid_set = get_dataset(
            config.data.valid,
            tokenizer,
            wrap=config.data.wrap,
            mode=validation_split,
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            streaming=False,
            num_proc=config.data.num_proc,
            config=config,
            **kwargs,
        )
        if hasattr(valid_set, '__len__'):
            rprint(f"Validation set len: {len(valid_set)}")

    dataloader_seed = config.seed if (config.mode == "eval" or is_torch_xla_available() or getattr(config.data, "force_seed", False)) else 42
    gprint(f"Dataloader seed: {dataloader_seed}")

    if skip_train:
        train_loader = None
    else:
        train_kwargs = dict(drop_last=True)
        train_dataloader_generator = torch.Generator().manual_seed(dataloader_seed)
        dl_cls = torch.utils.data.DataLoader
        if getattr(config.data, "webdataset_iterable", False) or getattr(config.data, "webdataset_indexed", False):
            train_kwargs.pop("drop_last", None)

        if getattr(config.loader, "disable_prefetch", False):
            train_kwargs["prefetch_factor"] = 1

        if getattr(config.data, "force_disable_shuffle", False) is False:
            if getattr(config.data, "webdataset_iterable", False):
                import webdataset
                dl_cls = webdataset.WebLoader
                train_kwargs["shuffle"] = False
                train_kwargs["prefetch_factor"] = 8
            elif getattr(config.data, "webdataset_indexed", False):
                import wids
                train_kwargs["sampler"] = wids.DistributedChunkedSampler(train_set, shuffle=True)
            elif isinstance(train_set, torch.utils.data.IterableDataset) is False:
                train_kwargs["shuffle"] = True

        if "tokens" in config.data.train and config.data.pin_dataset_to_gpu:
            if config.backend == 'cuda':
                cur_mb = torch.cuda.memory_reserved() / 1e9
                rprint(f"Moving dataloader to device {device} with: {cur_mb} GB of memory reserved")
            train_set = train_set.to(device=device)
            if config.backend == 'cuda':
                cur_mb = torch.cuda.memory_reserved() / 1e9
                rprint(f"Moved dataloader to device {device} with: {cur_mb} GB of memory reserved")

        if "tokens" in config.data.train:
            if getattr(config.data, "use_custom_tensordict_collate", False):
                train_kwargs["collate_fn"] = TensorCollate(device=device, enable_cuda_in_tensordict_collate=config.data.enable_cuda_in_tensordict_collate)
            else:
                train_kwargs["collate_fn"] = identity

            if getattr(config.data, "use_packing_collate", False):
                generator = torch.Generator().manual_seed(dataloader_seed)
                token_collate = train_kwargs["collate_fn"] if getattr(config.data, "use_custom_tensordict_collate", False) else None
                train_kwargs["collate_fn"] = PackingCollate(config, train_set, config.model.length, generator, tensor_collate=token_collate, tokenizer=tokenizer)

            if getattr(config.data, "use_weighted_tensordict_sampler", False):
                generator = torch.Generator().manual_seed(dataloader_seed)
                train_kwargs['sampler'] = WeightedDatasetSampler(train_set, generator=generator)
                train_kwargs["shuffle"] = False
            else:
                train_kwargs["shuffle"] = True

        if getattr(config.data, "use_list_collate", False):
            train_kwargs["collate_fn"] = lambda x: x

        if getattr(config.data, "force_shuffle_train", False):
            rprint("Forcing shuffle on train dataloader")
            train_kwargs["shuffle"] = True
        
        if getattr(config.data, "force_disable_shuffle_train", False):
            rprint("Forcing disable shuffle on train dataloader")
            train_kwargs["shuffle"] = False

        if getattr(config.data, "force_distributed_sampler", False):
            import torch_xla.runtime as xr
            train_kwargs["sampler"] = torch.utils.data.distributed.DistributedSampler(
                train_set,
                num_replicas=xr.world_size(),
                rank=xr.global_ordinal(),
                shuffle=True
            )

        if getattr(config.data, "use_identity_collate", False):
            train_kwargs["collate_fn"] = lambda x: x

        if train_set.__class__.__name__ == "WebLoader":
            train_loader = train_set
        else:
            rprint(f"Train dataloader kwargs: {train_kwargs}")
            train_loader = dl_cls(
                train_set,
                batch_size=None if getattr(config.data, "webdataset_iterable", False) else config.loader.batch_size,
                num_workers=config.loader.num_workers,
                pin_memory=config.loader.pin_memory,
                persistent_workers=config.loader.num_workers > 0 and getattr(config.loader, "persistent_workers", True),
                generator=train_dataloader_generator,
                **train_kwargs,
            )
        train_loader.tokenizer = tokenizer

    if skip_valid:
        valid_loader = None
    else:
        shuffle_valid = True
        valid_dataloader_generator = torch.Generator().manual_seed(dataloader_seed)
        valid_kwargs = dict(drop_last=True)

        dl_cls = torch.utils.data.DataLoader
        if getattr(config.data, "webdataset_iterable", False) or getattr(config.data, "webdataset_indexed", False):
            valid_kwargs.pop("drop_last", None)

        if getattr(config.data, "force_disable_shuffle", False) is False:
            if getattr(config.data, "webdataset_iterable", False):
                valid_kwargs["shuffle"] = False
                import webdataset
                dl_cls = webdataset.WebLoader
            elif getattr(config.data, "webdataset_indexed", False):
                import wids
                valid_kwargs["sampler"] = wids.DistributedChunkedSampler(valid_set, shuffle=True)
            elif isinstance(valid_set, torch.utils.data.IterableDataset) is False and shuffle_valid:
                valid_kwargs["shuffle"] = shuffle_valid

        if "tokens" in config.data.valid:
            if getattr(config.data, "use_custom_tensordict_collate", False):
                valid_kwargs["collate_fn"] = TensorCollate(device=device, enable_cuda_in_tensordict_collate=config.data.enable_cuda_in_tensordict_collate)
            else:
                valid_kwargs["collate_fn"] = identity

            if getattr(config.data, "use_packing_collate", False):
                generator = torch.Generator().manual_seed(dataloader_seed)
                token_collate = valid_kwargs["collate_fn"] if getattr(config.data, "use_custom_tensordict_collate", False) else None
                valid_kwargs["collate_fn"] = PackingCollate(config, valid_set, config.model.length, generator, tensor_collate=token_collate, tokenizer=tokenizer)

            if getattr(config.data, "use_weighted_tensordict_sampler", False):
                generator = torch.Generator().manual_seed(dataloader_seed)
                valid_kwargs['sampler'] = WeightedDatasetSampler(valid_set, generator=generator)
                
            if getattr(config.data, "shuffle_valid", False):
                torch.manual_seed(config.seed)

            valid_kwargs["shuffle"] = getattr(config.data, "shuffle_valid", False)

        if getattr(config.data, "force_distributed_sampler", False):
            import torch_xla.runtime as xr
            valid_kwargs["sampler"] = torch.utils.data.distributed.DistributedSampler(
                valid_set,
                num_replicas=xr.world_size(),
                rank=xr.global_ordinal(),
                shuffle=True
            )
            
        if valid_set.__class__.__name__ == "WebLoader":
            valid_loader = valid_set
        else:
            rprint(f"Valid dataloader kwargs: {valid_kwargs}")
            valid_loader = dl_cls(
                valid_set,
                batch_size=None if getattr(config.data, "webdataset_iterable", False) else config.loader.eval_batch_size,
                num_workers=getattr(config.loader, "num_eval_workers", config.loader.num_workers),
                pin_memory=config.loader.pin_memory,
                generator=valid_dataloader_generator,
                persistent_workers=False,
                **valid_kwargs,
            )
        # Will be used in generative perplexity calculation
        valid_loader.tokenizer = tokenizer

    return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

    def __init__(self, *args, generator=None, **kwargs):
        # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
        # which should be reproducible if pl.seed_everything was called beforehand.
        # This means that changing the seed of the experiment will also change the
        # sampling order.
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        kwargs.pop("shuffle", None)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {"random_state": self.generator.get_state(), "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get("random_state"))
        self.counter = state_dict["counter"]
        # self.start_counter = self.counter
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.

    def __iter__(self) -> typing.Iterator[int]:
        n = len(self.data_source)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter :]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {"epoch": self.epoch, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.counter = state_dict["counter"]
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter :]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


if __name__ == "__main__":
    import os

    with breakpoint_on_error():
        from omegaconf import OmegaConf

        cc12m_config = OmegaConf.create(
            {
                "model": {
                    "image_model": True,
                    "unified_model": True,
                },
                "data": {
                    "tokenizers_parallelism": False,
                    "resolution": 128,
                    "train": "pixparse/cc12m-wds",
                    "val": "pixparse/cc12m-wds",
                    "streaming": False,
                    "precache": True,
                    "tokenizer_name_or_path": "gpt2",
                    "n_val_samples": None,
                    "n_train_samples": None,
                    "block_size": 32,
                    "data_dir": "/path/to/cc12m",
                },
            }
        )

        imagenet_config = OmegaConf.create(
            {
                "model": {
                    "image_model": True,
                },
                "data": {
                    "resolution": 128,
                    "train": "ILSVRC/imagenet-1k",
                    "val": "ILSVRC/imagenet-1k",
                    "streaming": False,
                    "precache": True,
                    "tokenizer_name_or_path": "gpt2",
                },
            }
        )

        facecaption_config = OmegaConf.create(
            {
                "seed": 12345,
                "model": {
                    "image_model": True,
                },
                "data": {
                    "resolution": 256,
                    "train": "facecaption",
                    "val": "facecaption",
                    "streaming": False,
                    "precache": False,
                    "tokenizer_name_or_path": "gpt2",
                    "cache_dir": os.environ["HF_DATASETS_CACHE"],
                    "raw_data_dir": "/grogu/user/mprabhud/data/diffusion/facecaption",
                    "block_size": 32,
                },
                "loader": {
                    "num_workers": 0,
                    "batch_size": 1,
                    "eval_batch_size": 1,
                },
                "trainer": {
                    "devices": 1,
                    "num_nodes": 1,
                    "accumulate_grad_batches": 1,
                },
            }
        )

        tokenizer = get_tokenizer(facecaption_config)
        dataset = get_dataset(
            dataset_name=facecaption_config.data.train,
            mode="train",
            config=facecaption_config,
            tokenizer=tokenizer,
        )
        test = next(iter(dataset))
        breakpoint()



from typing import List, Dict
import torch
from tensordict import TensorDict
def process_batch(batch: TensorDict):
    if isinstance(batch, list):
        return [process_batch(b) for b in batch]
    else:
        if "write_flag" in batch:
            del batch["write_flag"]
        if "dataset_idx" in batch:
            del batch["dataset_idx"]
        batch.auto_batch_size_()
        return batch

def ignore_slice(tensor, slice, padding_token_id):
    tensor["modality"][slice] = -1
    tensor["attention_mask"][slice] = 0
    tensor["input_ids"][slice] = padding_token_id
    if "sample_ids" in tensor:
        tensor["sample_ids"][slice] = -1
    else:
        tensor["sample_ids"] = torch.full(tensor["input_ids"].shape, fill_value=-1, dtype=tensor["input_ids"].dtype, device=tensor["input_ids"].device)

class PackingCollate:
    def __init__(self, config, dataset, seq_length, generator, tensor_collate=None, tokenizer=None):
        self.dataset = dataset
        self.seq_length = seq_length
        self.tensor_collate = tensor_collate
        self.generator = generator
        self.tokenizer = tokenizer
        self.padding_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.disable_packing = getattr(config.data, "disable_packing", False)
        img_special_tokens = tokenizer("<image>", add_special_tokens=False)['input_ids']
        assert len(img_special_tokens) == 1
        self.image_token_id = img_special_tokens[0]

    def __call__(self, batch: TensorDict):
        if self.tensor_collate is not None:
            if isinstance(batch, list):
                batch = [self.tensor_collate(b) for b in batch]
            else:
                batch = self.tensor_collate(batch)

        B = len(batch)
        seq_length = self.seq_length

        batch = process_batch(batch)
        assert batch[0].batch_size is None or len(batch[0].batch_size) == 1

        new_batch = batch[0].new_zeros((B, seq_length))
        ignore_slice(new_batch, slice(None, None), self.padding_token_id)

        for i in range(B):
            total_length = 0
            sample_idx = 0
            sample_queue = [batch[i]]

            # We originally get bs number of samples but since we're packing, we probably need more so we randomly select.
            while total_length < seq_length:
                if self.disable_packing and sample_idx > 0:
                    break
                if not sample_queue:
                    dataset_idx = torch.randint(len(self.dataset.datasets), (1,), generator=self.generator).item()
                    element_idx = torch.randint(len(self.dataset.datasets[dataset_idx]), (1,), generator=self.generator).item()
                    sample = self.dataset[(dataset_idx, element_idx)]
                    sample = process_batch(sample)
                else:
                    sample = sample_queue.pop(0)

                available_length = seq_length - total_length
                if available_length < sample.shape[0] // 4:
                    if total_length > 0:
                        break
                    else:
                        continue

                if "sample_ids" not in sample:
                    sequence_starts = (sample['input_ids'] == self.padding_token_id).long()
                    sample["sample_ids"] = torch.cumsum(sequence_starts, dim=0) - 1
                    processed_ids = torch.where(sample["sample_ids"] < 0, torch.zeros_like(sample["sample_ids"]), -1)
                    sample["sample_ids"] = processed_ids

                if not ((sample["sample_ids"] == 0) | (sample["sample_ids"] == -1)).all():
                    assert (sample["modality"] == 0).all()

                first_neg_one = (sample["sample_ids"] == -1).nonzero(as_tuple=True)[0]

                if first_neg_one.numel() > 0:
                    first_neg_one = first_neg_one[0].item()
                else:
                    assert sample["attention_mask"].all()
                    first_neg_one = len(sample["attention_mask"])
                
                valid_slice = slice(None, min(first_neg_one, available_length))
                new_length = min(first_neg_one, available_length)
                
                sample["sample_ids"][valid_slice] = sample_idx
                new_batch[i, total_length:total_length+new_length] = sample[valid_slice]

                total_length += new_length
                sample_idx += 1

            if (new_batch["sample_ids"] == -1).all():
                gprint(f"WARNING!!!! All sample ids are -1 in packing collate before ignore")

            if new_batch["modality"][i, -1] == 1:
                # Find contiguous sequence of image tokens from the end
                modality_slice = new_batch["modality"][i]
                is_image = modality_slice == 1
                
                # Get indices where modality changes
                change_points = torch.where(is_image[:-1] != is_image[1:])[0] + 1
                
                if change_points.numel() > 0 and is_image[-1]:
                    # Get start of last contiguous image sequence
                    start_pos = change_points[-1].item()
                    assert (new_batch["modality"][i, start_pos:] == 1).all()
                    try:
                        if start_pos > 0 and new_batch["input_ids"][i, start_pos - 1] == self.image_token_id:
                            start_pos -= 1
                    
                        if start_pos > 0 and new_batch["input_ids"][i, start_pos - 1] != self.eos_token_id:
                            new_batch["input_ids"][i, start_pos] = self.eos_token_id
                            new_batch["attention_mask"][i, start_pos] = 1
                            new_batch["modality"][i, start_pos] = 0
                            start_pos += 1

                    except IndexError:
                        print(f"WARNING!!!! ERROR IN PACKING COLLATE")

                    ignore_slice(new_batch[i], slice(start_pos, None), self.padding_token_id)

                if (new_batch["sample_ids"] == -1).all():
                    gprint(f"WARNING!!!! All sample ids are -1 in packing collate after ignore")

        return new_batch

