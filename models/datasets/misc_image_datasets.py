import os
import random
import typing
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import pandas as pd
from unidisc.tokenizers.conversation import get_image_gen_tokens, get_image_suffix
import torch
import torch.nn as nn
from numpy import pad
from PIL import Image, ImageFile
from tensordict import TensorDict
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
import re
import shutil
from constants import LIB_DIR
from decoupled_utils import barrier, gprint, is_main_process, is_torch_cuda_available, rprint
from models.datasets.webdataset_utils import get_data
from unidisc.utils.tensor_utils import get_interleaved_indices, get_contiguous_blocks, packbits, unpackbits
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ResilientIterableDatasetWrapper(torch.utils.data.IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        iterator = iter(self.dataset)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                raise StopIteration
            except Exception as e:
                gprint(e)
                iterator = iter(self.dataset)


class ResilientDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        while True:
            try:
                return self.dataset[idx]
            except Exception as e:
                gprint(e)
                import traceback
                traceback.print_exc()
                idx = (idx + 1) % len(self.dataset)


class CustomTransformDataset(Dataset):
    def __init__(self, original_dataset, transform):
        self.original_dataset = original_dataset
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        for i in range(10):
            try:
                data = self.original_dataset[idx]
                if i > 0:
                    rprint(f"Took {i} times")
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                gprint(e)
            
        transformed_data = self.transform(data, idx=idx)
        return transformed_data

class TensorCollate(nn.Module):
    def __init__(self, device=None, transform=None, enable_cuda_in_tensordict_collate=True):
        super().__init__()
        self.device = torch.device(device) if device is not None else None
        self.transform = transform
        self.enable_cuda_in_tensordict_collate = enable_cuda_in_tensordict_collate

    def __call__(self, x: TensorDict):
        if self.device is not None and self.device.type == "cuda" and self.enable_cuda_in_tensordict_collate:
            out = x.pin_memory() # move data to RAM
        else:
            out = x

        if self.device and self.enable_cuda_in_tensordict_collate:
            out = out.to(self.device)

        if self.transform:
            out = self.transform(out)

        return out

def clean_identity(value):
    cleaned_value = "".join(filter(str.isdigit, str(value)))
    return int(cleaned_value) if cleaned_value else None


class VGGFace(Dataset):
    def __init__(self, path, is_train, filter_resolution: int = 196, transform=None, cond_transform=None, v2=False):
        self.path = Path(path)
        self.is_train = is_train

        self.train_folders = self.get_folders("train")
        self.test_folders = self.get_folders("test")
        self.prefix = "train" if self.is_train else "test"
        self.gender_meta = pd.read_csv(self.path / 'meta' / 'identity_meta.csv', on_bad_lines='skip')
        self.v2 = v2
        self.transform = transform
        self.cond_transform = cond_transform
        self.filter_resolution = filter_resolution

        cache_file = self.path / f"{self.prefix}_{'filtered' if filter_resolution == 196 else ('unfiltered' if filter_resolution is None else 'filtered_' + str(filter_resolution))}.pkl"
        if cache_file.exists():
            self.data = pd.read_pickle(cache_file)
        else:
            self.data = pd.read_csv(self.path / "MAAD_Face.csv")
            self.data["Identity"] = self.data["Identity"].apply(clean_identity)
            self.data = self.data[self.data["Identity"].isin(self.train_folders if self.is_train else self.test_folders)]
            def get_image_size(file_path):
                with Image.open(file_path) as img:
                    return img.size

            self.data['Resolution'] = self.data.apply(lambda row: get_image_size(self.path / "data" / self.prefix / f"{row['Filename']}"), axis=1)
            if filter_resolution:
                self.data = self.data[self.data['Resolution'].apply(lambda x: x[0] >= filter_resolution and x[1] >= filter_resolution)]

            self.data = self.data.drop('Resolution', axis=1)
            self.data.to_pickle(cache_file)

    def get_folders(self, split):
        train_path = Path(self.path) / "data" / split
        folders = [int(folder.name[1:]) for folder in train_path.iterdir() if folder.is_dir()]
        return folders

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.path / "data" / self.prefix / f"{row['Filename']}"
        attr = row.to_numpy()[2:].astype(int)
        tokens = attr.copy() + 1
        non_zero_mask = attr > 0
        non_zero_idx = np.where(non_zero_mask)[0]

        if self.v2:
            attn_mask = np.ones(48)
            matched_ = self.gender_meta[self.gender_meta["Class_ID"] == row.Filename.split("/")[0]]
            assert len(matched_) <= 1, f"idx: {idx}, filename: {row}"
            if len(matched_) == 1:
                matched_row = matched_.iloc[0]
                is_female = matched_row[" Gender"] == " f"
            else:
                is_female = False
                attn_mask[0] = 0
                
            tokens[non_zero_idx] = non_zero_idx + 3
            tokens = np.concatenate([np.array([2 if is_female else 0]), tokens])
        else:
            attn_mask = np.zeros(len(tokens))
            tokens[non_zero_idx] = non_zero_idx + 2

        img = Image.open(img_path)
        ret_dict = {"img": img, "input_ids": tokens, "attention_mask": attn_mask, "idx": idx}

        if self.transform:
            ret_dict["img"] = self.transform(img)

        if self.cond_transform is not None:
            ret_dict["cond_img"] = self.cond_transform(img)

        return ret_dict

class Cub2011(VisionDataset):
    def __init__(
        self,
        root: Path,
        train=True,
        transform=None,
        target_transform=None,
        transforms=None,
        shuffle_attributes=False,
        n_duplicate=None,
        n_samples=None,
        **kwargs,
    ):
        super(Cub2011, self).__init__(root, transform=transform, target_transform=target_transform, transforms=transforms)
        self.train = train
        self.shuffle_attributes = shuffle_attributes
        self.n_duplicate = n_duplicate
        self.n_samples = n_samples
        self.loader = default_loader
        self._load_metadata()

    def _load_metadata(self):
        images = pd.read_csv(self.root / "images.txt", sep=" ", names=["img_id", "filepath"])
        image_class_labels = pd.read_csv(self.root / "image_class_labels.txt", sep=" ", names=["img_id", "target"])
        train_test_split = pd.read_csv(self.root / "train_test_split.txt", sep=" ", names=["img_id", "is_training_img"])

        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")
        class_names = pd.read_csv(self.root / "classes.txt", sep=" ", names=["class_name"], usecols=[1])
        self.class_names = class_names["class_name"].to_list()

        if self.train:
            self.data = self.data[(self.data.is_training_img == 1) | (self.data.index < 10000)]
        else:
            self.data = self.data[(self.data.is_training_img == 0) & (self.data.index >= 10000)]

        df_images = pd.read_csv(self.root / "images.txt", sep="\s+", names=["img_id", "img_path"])
        df_labels = pd.read_csv(self.root / "classes.txt", sep="\s+", names=["cls_id", "cls_name"])
        df_is_train = pd.read_csv(self.root / "train_test_split.txt", sep="\s+", names=["img_id", "is_train"])

        df_att = pd.read_csv(self.root / "attributes.txt", sep="\s+", names=["att_id", "att_name"])
        df_att_ant = pd.read_csv(
            self.root / "attributes/image_attribute_labels_filtered.txt", names=["img_id", "att_id", "is_pres", "cert_id", "time"], sep="\s+"
        )

        image_ids = df_att_ant["img_id"].unique()
        df_images = df_images[df_images["img_id"].isin(image_ids)]
        df_is_train = df_is_train[df_is_train["img_id"].isin(image_ids)]

        df_data_att = pd.merge(df_att_ant, df_att, on="att_id", how="left")
        df_data_att = df_data_att.loc[(df_data_att["is_pres"] == 1) & (df_data_att["cert_id"] > 2)]

        self.df_data_att = df_data_att

    def __len__(self):
        orig_size = len(self.data)
        if self.n_samples is not None:
            orig_size = self.n_samples
        if self.n_duplicate is not None:
            orig_size = orig_size * self.n_duplicate
        return orig_size

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        if self.n_samples is not None:
            idx = idx % self.n_samples

        idx = idx % len(self.data)
        sample = self.data.iloc[idx]
        img_id = sample["img_id"]
        path = self.root / "images" / sample.filepath
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        data = {"img": img}
        data["text"] = ", ".join(list(self.df_data_att.loc[(self.df_data_att["img_id"] == img_id)]["att_name"].values))
        tokens = torch.full((312,), dtype=torch.int64, fill_value=0)  # 40 is our pad token
        _atts = self.df_data_att.loc[(self.df_data_att["img_id"] == img_id)]["att_id"].values
        _atts = _atts.tolist()
        if self.shuffle_attributes:
            random.shuffle(_atts)
        tokens[: len(_atts)] = torch.tensor(_atts)
        data["input_ids"] = tokens
        data["attention_mask"] = tokens > 0
        return data


class TokenDataset(Dataset):
    def __init__(self, path, n_samples: typing.Optional[int] = None, n_duplicate: Optional[int] = None, should_aug: bool = False):
        self.path = path
        self.data = TensorDict.load_memmap(path)
        self.n_samples = n_samples
        self.n_duplicate = n_duplicate
        self.device = None

    def to_gpu(self, device):
        self.device = device
        self.data = self.data.to(self.device)

    def __len__(self):
        if self.n_duplicate is None and self.n_samples is None:
            return len(self.data)
        else:
            n_duplicate = 1 if self.n_duplicate is None else self.n_duplicate
            n_samples = 1 if self.n_samples is None else self.n_samples
            return n_samples * n_duplicate

    def __getitem__(self, idx):
        n_samples = self.n_samples if self.n_samples is not None else len(self.data)
        n_duplicate = self.n_duplicate if self.n_duplicate is not None else 1
        idx = idx % (n_samples * n_duplicate)
        element = self.data[idx]

        index_keys = ["img_input_ids", "txt_input_ids"]
        for key in index_keys:
            if key in element:
                element[key] = element[key].to(torch.int64)

        index_keys = ["img_label"]
        for key in index_keys:
            if key in element:
                element[key] = element[key].squeeze(-1)

        return element.to_dict()


def get_sora_dataset(mode, config, tokenizer, should_aug=True, **kwargs):
    assert (LIB_DIR / "Open-Sora-Plan").exists()
    __import__("sys").path.append(str(LIB_DIR / "Open-Sora-Plan"))
    from opensora.dataset.transform import (CenterCropResizeVideo,
                                            RandomHorizontalFlipVideo,
                                            TemporalRandomCropGlobal,
                                            ToTensorVideo)

    from models.datasets.t2v_datasets import T2V_dataset

    is_train = mode == "train"
    n_duplicate_train = getattr(config.data, "n_duplicate_train", None)
    n_duplicate_val = getattr(config.data, "n_duplicate_val", None)
    n_duplicate = n_duplicate_train if is_train else n_duplicate_val

    n_val_samples = getattr(config.data, "n_val_samples", None)
    n_train_samples = getattr(config.data, "n_train_samples", None)
    n_samples = n_train_samples if is_train else n_val_samples

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    is_celeb = getattr(config.data, "celeb", False)
    temporal_sample = TemporalRandomCropGlobal()  # DynamicSampleDuration

    transform = transforms.Compose(
        [
            ToTensorVideo(),
            CenterCropResizeVideo(config.data.resolution),
            *([RandomHorizontalFlipVideo(p=0.5)] if is_train and should_aug else []),  # This may mess up certain captions.
        ]
    )

    dataset = T2V_dataset(
        num_frames=config.data.num_frames,
        transform=transform,
        temporal_sample=temporal_sample,
        tokenizer=tokenizer,
        hf_format=True,
        unified_model=config.model.unified_model,
        specified_keywords_only=getattr(config.data, "specified_keywords_only", None),
        ignore_clips=True,
        celeb_only=is_celeb,
        model_max_length=128,
        is_train=is_train,
        n_duplicate=n_duplicate,
        n_samples=n_samples,
        **kwargs,
    )

    return dataset


def get_sample_ids_from_attention_mask(attention_mask):
    if attention_mask.all():
        return torch.zeros_like(attention_mask, dtype=torch.int)

    # Convert boolean tensor to integer for easy manipulation (True -> 1, False -> 0)
    inverted = (~attention_mask).to(torch.int)
    
    # Find the last position where the False sequence starts
    diff = inverted.diff(dim=0, prepend=torch.tensor([0], dtype=inverted.dtype))
    
    # Find the starting position of the last contiguous False sequence
    nonzero_indices = (diff == 1).nonzero(as_tuple=True)[0]
    if nonzero_indices.numel() == 0: assert False
    last_false_start = nonzero_indices.max(dim=0)[0] if nonzero_indices.numel() > 0 else torch.tensor(0)
    
    # Mark all elements in the last contiguous False sequence as -1
    output = torch.zeros_like(attention_mask, dtype=torch.int)
    output[last_false_start:] = inverted[last_false_start:].cumsum(0).ne(0).to(torch.int) * -1
    
    return output


class MultipleTensorDictDataset(Dataset):
    def __init__(self, datasets, weights, dataset_names, config, tokenizer=None, returns_raw_images=False, returns_tokenized_text=False, returns_parquet=False, returns_tokenize_vqvae_in_dataloader=False, allow_label=False):
        self.datasets = [x.to("cpu") if isinstance(x, TensorDict) else x for x in datasets]
        self.weights = weights
        self.dataset_names = dataset_names
        self.add_dataset_idx = True
        self.tokenizer = tokenizer  # this is for text only
        self.text_vocab_size = getattr(config.model, "text_vocab_size")

        self.config = config        
        self.returns_raw_images = returns_raw_images
        self.returns_tokenized_text = returns_tokenized_text
        self.returns_parquet = returns_parquet
        self.returns_tokenize_vqvae_in_dataloader = returns_tokenize_vqvae_in_dataloader
        self.seq_len = config.model.length
        self.allow_label = allow_label
        self.require_sample_ids = getattr(config.data, "require_sample_ids", False)
        self.remove_txt_img_padding = getattr(config.data, "remove_txt_img_padding", False)
        self.add_image_gen_tokens = getattr(config.data, "add_image_gen_tokens", False)
        self.dynamic_packing_lengths = getattr(config.data, "dynamic_packing_lengths", False)

        if self.dynamic_packing_lengths:
            # We can't directly stack here, we first need to pack/pad in the packing collate
            rprint(f"Removing __getitems__ from {self.__class__.__name__} as we are using dynamic packing lengths")
            if hasattr(self, '__getitems__'):
                delattr(self.__class__, '__getitems__')

        if self.allow_label and not self.returns_raw_images:
            self.raw_images_keys_supported = ["input_ids", "attention_mask", "modality", "label", "sample_ids"]
        else:
            self.raw_images_keys_supported = ["img", "input_ids", "attention_mask", "modality", "idx", "label", "sample_ids"]

        assert not getattr(config.trainer, "force_shift_image_batches", False)

    def __len__(self):
        return sum(10 if isinstance(dataset, torch.utils.data.IterableDataset) else len(dataset) for dataset in self.datasets)

    def __getitem__(self, index_data):
        dataset_idx, idx = index_data
        dataset = self.datasets[dataset_idx]
        if isinstance(dataset, TensorDict):
            data = dataset[idx]
            txt_len = None
            
            if "attention_mask" in data and (data["attention_mask"] == False).all():
                is_pad = data["input_ids"] == self.tokenizer.pad_token_id
                change_points = torch.where(is_pad[:-1] != is_pad[1:])[0] + 1
                if change_points.numel() > 0 and is_pad[-1]:
                    start_pos = change_points[-1].item()
                    data["attention_mask"][:start_pos] = True

            if "input_ids" not in data:
                if self.remove_txt_img_padding:
                    image_gen_tokens = get_image_gen_tokens(self.tokenizer)
                    new_txt_input_ids = data["txt_input_ids"].to(torch.int64)[data["txt_attention_mask"].to(torch.bool)]
                    new_txt_attention_mask = data["txt_attention_mask"].to(torch.bool)[data["txt_attention_mask"].to(torch.bool)]
                    new_txt_input_ids = torch.cat([image_gen_tokens["input_ids"][0], new_txt_input_ids], dim=-1)

                    if new_txt_input_ids[-1] == self.tokenizer.eos_token_id:
                        new_txt_input_ids = new_txt_input_ids[:-1]
                        new_txt_attention_mask = new_txt_attention_mask[:-1]

                    new_txt_input_ids = torch.cat([new_txt_input_ids, torch.tensor(get_image_suffix(self.tokenizer), dtype=torch.int64)], dim=-1)
                    new_txt_attention_mask = torch.cat([new_txt_attention_mask, torch.ones_like(new_txt_attention_mask[:1])], dim=-1)
                    new_txt_input_modality = torch.zeros((new_txt_input_ids.shape[0],), dtype=torch.int64)
                    img_modality = torch.ones((data["img_input_ids"].shape[0],), dtype=torch.int64)

                    new_input_ids = torch.cat([new_txt_input_ids, data["img_input_ids"].to(torch.int64), torch.tensor([self.tokenizer.eos_token_id], dtype=torch.int64)], dim=-1)
                    new_attention_mask = torch.ones_like(new_input_ids, dtype=torch.bool)
                    new_modality = torch.cat([new_txt_input_modality, img_modality, torch.zeros_like(new_txt_input_modality[:1])], dim=-1)

                    txt_len = None
                    data = TensorDict.from_dict(
                        {
                            "input_ids": new_input_ids,
                            "attention_mask": new_attention_mask,
                            "modality": new_modality
                        },
                        batch_size=[],
                    )
                else:
                    txt_len = data["txt_input_ids"].shape[0]
                    data = TensorDict.from_dict(
                        {
                            "input_ids": torch.cat(
                                [data["txt_input_ids"].to(torch.int64), data["img_input_ids"].to(torch.int64)], dim=-1
                            ),
                            "attention_mask": torch.cat(
                                [data["txt_attention_mask"].to(torch.bool), torch.ones_like(data["img_input_ids"]).to(torch.bool)], dim=-1
                            ),
                        },
                        batch_size=[],
                    )

                if self.require_sample_ids and "sample_ids" not in data:
                    data["sample_ids"] = get_sample_ids_from_attention_mask(data["attention_mask"])

            else:
                if "modality" in data and data["modality"].shape[-1] != data["input_ids"].shape[-1]:
                    data["modality"] = unpackbits(data["modality"]).to(torch.int64)

                if "attention_mask" in data and data["attention_mask"].shape[-1] != data["input_ids"].shape[-1]:
                    data["attention_mask"] = unpackbits(data["attention_mask"]).to(torch.bool)
            
            if "modality" not in data:
                data["modality"] = torch.zeros((data["input_ids"].shape[0],), dtype=torch.int64)

            elif data["modality"].shape[0] == 1:
                data["modality"] = data["modality"].expand(data["input_ids"].shape[0])

            if txt_len is not None:
                data["modality"][txt_len:] = 1

            if "idx" in data:
                data.pop("idx")
        else:
            if isinstance(dataset, torch.utils.data.IterableDataset):
                data = next(iter(dataset))
            else:
                data = dataset[idx]

            if self.returns_raw_images:
                if not isinstance(data, TensorDict):
                    data = TensorDict.from_dict(data, batch_size=[])
                
                if "idx" in data and len(data["idx"].shape) == 0:
                    data["idx"] = data["idx"].unsqueeze(-1)

                if "input_ids" not in data:
                    data["input_ids"] = torch.full((self.seq_len,), dtype=torch.int64, fill_value=-1)
                    data["attention_mask"] = torch.full((self.seq_len,), dtype=torch.bool, fill_value=True)
                    data["modality"] = torch.full((self.seq_len,), dtype=torch.int64, fill_value=1)
                
                elif "modality" not in data:
                    data["modality"] = torch.full((self.seq_len,), dtype=torch.int64, fill_value=1)  # assuming images
                    data["modality"][:data["input_ids"].shape[0]] = 0
                    data["input_ids"] = torch.cat([data["input_ids"], torch.full((self.seq_len - data["input_ids"].shape[0],), dtype=torch.int64, fill_value=-1)])
                    data["attention_mask"] = torch.cat([data["attention_mask"], torch.full((self.seq_len - data["attention_mask"].shape[0],), dtype=torch.bool, fill_value=True)]).bool()

            elif self.returns_tokenized_text:
                from dataloader import tokenize_text
                _txt = data["content"] if "content" in data else data["text"]
                data = tokenize_text(self.tokenizer, self.text_length, _txt)
                data = TensorDict.from_dict({
                    "input_ids": data["input_ids"].to(torch.int64),
                    "attention_mask": data["attention_mask"].to(torch.bool)},
                batch_size=[])
                if "modality" not in data:
                    data["modality"] = torch.full((data["input_ids"].shape[0], ), dtype=torch.int64, fill_value=0)
            elif self.returns_parquet:
                if "attention_mask" not in data:
                    data["attention_mask"] = torch.ones((len(data["input_ids"])), dtype=torch.bool)
                data = TensorDict.from_dict({
                    "input_ids": data["input_ids"],
                    "attention_mask": data["attention_mask"].bool() if isinstance(data["attention_mask"], torch.Tensor) else torch.tensor(data["attention_mask"], dtype=torch.bool)
                }, batch_size=[])

                if "modality" not in data:
                    data["modality"] = torch.full((data["input_ids"].shape[0],), dtype=torch.int64, fill_value=0)
                
                if self.require_sample_ids and "sample_id" not in data:
                    sequence_starts = (data["input_ids"] == self.tokenizer.bos_token_id).long()
                    assert sequence_starts[0] == 1
                    sample_ids = torch.cumsum(sequence_starts, dim=0) - 1
                    unique_ids, counts = torch.unique(sample_ids, return_counts=True)
                    occurrence_mask = torch.isin(sample_ids, unique_ids[counts < 10]) # Require at least 10 tokens to be presen
                    data["sample_ids"] = torch.where(occurrence_mask, -1, sample_ids)

            elif self.returns_tokenize_vqvae_in_dataloader:
                if "txt_input_ids" in data and "txt_attention_mask" in data:
                    modality = torch.zeros(data["txt_input_ids"].shape[0] + data["img_input_ids"].shape[0], dtype=torch.int64)
                    modality[data["txt_input_ids"].shape[0]:] = 1
                    data = TensorDict.from_dict({
                        "input_ids": torch.cat([data["txt_input_ids"], data["img_input_ids"]], dim=-1), 
                        "attention_mask": torch.cat([data["txt_attention_mask"], torch.ones_like(data["img_input_ids"], dtype=torch.bool)], dim=-1).bool(),
                        "modality": modality
                    }, batch_size=[])
                else:
                    data = TensorDict.from_dict({
                        "input_ids": data["img_input_ids"], 
                        "attention_mask": torch.ones_like(data["img_input_ids"], dtype=torch.bool), 
                        "modality": torch.full((data["img_input_ids"].shape[0],), dtype=torch.int64, fill_value=1)
                    }, batch_size=[])
            else:
                raise ValueError(f"Unsupported return type")

        data["input_ids"] = data["input_ids"].to(torch.int64)
        data["input_ids"] = torch.where(
            (data["modality"] == 1) & (data["input_ids"] != -1),
            data["input_ids"] + self.config.data.img_token_shift,
            data["input_ids"]
        )

        if not self.allow_label and "label" in data:
            data.pop("label")

        if self.returns_raw_images or self.allow_label:
            # fill in the missing keys in tensor dict for both text and image batches
            for key in self.raw_images_keys_supported:
                if key not in data:
                    if key == "img":
                        data[key] = torch.zeros((3, self.config.data.resolution, self.config.data.resolution), dtype=torch.float32)
                    elif key == "label":
                        data[key] = torch.full((1,), dtype=torch.int64, fill_value=0)
                    else:
                        data[key] = torch.full((self.config.model.length,), dtype=torch.int64, fill_value=self.tokenizer.pad_token_id)

        if "attention_mask" in data and (data["attention_mask"] == 0).all():
            breakpoint()

        return data.clone()

    def __getitems__(self, index_data_list):
        return torch.stack([self.__getitem__(index_data) for index_data in index_data_list]).clone()
