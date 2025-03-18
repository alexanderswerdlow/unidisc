import base64
import io
import json
import random
import sys
import tarfile
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import random
import io
import socket
from collections import defaultdict
import pickle
import torchvision
from constants import LIB_DIR
from dataloader import tokenize_text
from decoupled_utils import gprint, rprint
import pandas as pd
import copy
from pathlib import Path
chameleon_path = LIB_DIR / "Lumina-mGPT/lumina_mgpt"
sys.path.append(str(chameleon_path))

try:
    from data.convertsation import Conversation
    from data.item_processor import FlexARItemProcessor
    class ItemProcessor(FlexARItemProcessor):
        def __init__(
            self,
            tokenizer="Alpha-VLLM/Lumina-mGPT-7B-768",
            conv_template=Conversation,
            target_size=512,
        ):
            super().__init__(tokenizer, conv_template, target_size)
            print(self.crop_size_list)

        def process_item(self, img, txt, training_mode=True, out_flatten=True, w=None, h=None):
            # Add custom codes here to convert raw_item to the standard format
            # The standard format contains the "conversations" and "image" keys

            _prompt = f"Generate an image of {w}x{h} according to the following prompt:\n{txt}" if w is not None and h is not None else f"Generate an image according to the following prompt:\n{txt}"
            item = {
                "conversations": [
                    {
                        "from": "human",
                        "value": _prompt
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>"
                    },
                ],
                "image": [img],
            }

            return super(ItemProcessor, self).process_item(item, training_mode, out_flatten)
        
        def process_item_json(self, item, training_mode=True, out_flatten=True):
            # Add custom codes here to convert raw_item to the standard format
            # The standard format contains the "conversations" and "image" keys

            # _prompt = f"Generate an image of {w}x{h} according to the following prompt:\n{txt}" if w is not None and h is not None else f"Generate an image according to the following prompt:\n{txt}"
            # item = {
            #     "conversations": [
            #         {
            #             "from": "human",
            #             "value": _prompt
            #         },
            #         {
            #             "from": "gpt",
            #             "value": "<|image|>"
            #         },
            #     ],
            #     "image": [img],
            # }

            return super(ItemProcessor, self).process_item(item, training_mode, out_flatten)
except Exception as e:
    if chameleon_path.exists():
        rprint(f"Failed to import Chameleon tokenizers from {chameleon_path}: {e}")



def tensor_center_crop(tensor_image, crop_size):
    _, _, h, w = tensor_image.shape

    while h >= 2 * crop_size[0] and w >= 2 * crop_size[1]:
        tensor_image = F.interpolate(tensor_image, size=(h // 2, w // 2), mode='area')
        _, _, h, w = tensor_image.shape

    scale = max(crop_size[0] / h, crop_size[1] / w)
    new_h, new_w = round(h * scale), round(w * scale)
    tensor_image = F.interpolate(tensor_image, size=(new_h, new_w), mode='bilinear')

    crop_top = random.randint(0, new_h - crop_size[0])
    crop_left = random.randint(0, new_w - crop_size[1])
    crop_bottom = crop_top + crop_size[0]
    crop_right = crop_left + crop_size[1]
    return tensor_image[:, :, crop_top:crop_bottom, crop_left:crop_right]

def var_center_crop(tensor_image, crop_size_list, random_top_k=1):
    _, _, h, w = tensor_image.shape
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k]
    )[1]
    # alternates = sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)
    # for i, (a, (x, y)) in enumerate(alternates):
    #     print(f"{i}: {(x // 16) * (y // 16)}")
    return tensor_center_crop(tensor_image, crop_size)

def tokenize_chameleon_fast(config, tokenizer=None, vae=None, batch=None, txt_decoded=None, **kwargs):
    assert "idx" in batch
    
    if txt_decoded is None:
        txt_decoded = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    all_attention_masks = []
    all_input_ids = []

    bs = batch['img'].shape[0]
    _img = var_center_crop(batch['img'], vae.crop_size_list, random_top_k=5)
    image_toks = vae.chameleon_ori_image_tokenizer.img_tokens_from_tensor(_img)
    image_toks = vae.chameleon_ori_translation.img2bpe_mapping_tensor[image_toks]
    h, w = _img.shape[-2:]
    h_grids, w_grids = h // vae.patch_size, w // vae.patch_size
    full_image_toks = image_toks.reshape(bs, h // 16, w // 16)
    new_line_id = vae.token2id(vae.new_line_token)

    full_image_toks = torch.cat(
        (
            full_image_toks,
            torch.full((bs, h // 16, 1), fill_value=new_line_id,  device=full_image_toks.device, dtype=full_image_toks.dtype),
        ),
        dim=-1,
    ).flatten(start_dim=1, end_dim=-1)

    result_toks = torch.cat([
        torch.tensor([
            vae.token2id(vae.image_start_token),
            vae.token2id(vae.get_n_grids_token(h_grids)),
            vae.token2id(vae.get_n_grids_token(w_grids))
        ], device=full_image_toks.device, dtype=full_image_toks.dtype).unsqueeze(0).expand(bs, -1),
        full_image_toks,
        torch.tensor([
            vae.token2id(vae.image_end_token)
        ], device=full_image_toks.device, dtype=full_image_toks.dtype).unsqueeze(0).expand(bs, -1)
    ], dim=1)

    input_ids = torch.full((bs, config.model.length,), fill_value=-100, dtype=torch.int64)
    attention_mask = torch.full((bs, config.model.length,), fill_value=False, dtype=torch.bool)

    for i in range(batch['input_ids'].shape[0]):
        _img = (result_toks[i], config.data.resolution, config.data.resolution)
        _txt = txt_decoded[i][:200]
        tokens, labels = vae.process_item(_img, _txt, out_flatten=False, h=h, w=w)
        idx = 0
        for j, token_or_media in enumerate(tokens):
            if isinstance(token_or_media, int):
                input_ids[i, idx:idx+1] = token_or_media
                idx += 1
            else:
                media_len = len(token_or_media["input_ids"])
                input_ids[i, idx:idx+media_len] = token_or_media["input_ids"]
                idx += media_len
        
        attention_mask[i, :idx] = True
        if idx >= config.model.length:
            gprint("WARNING!!!! Truncating input ids")
    
    all_attention_masks = attention_mask
    all_input_ids = input_ids

    return all_input_ids, all_attention_masks


def tokenize_chameleon_mmc4(config, tokenizer, vae, batch, device, mapping, **kwargs):
    all_images, all_content = get_mmc4(config, tokenizer, vae, batch, device, mapping)

    _img = torch.cat([tensor_center_crop(torch.from_numpy(np.array(img))[None, :].permute(0, 3, 1, 2) / 255, (config.data.resolution, config.data.resolution)) for img in all_images])
    
    all_attention_masks = []
    all_input_ids = []

    bs = _img.shape[0]
    # _img = var_center_crop(batch['img'], vae.crop_size_list, random_top_k=5)
    image_toks = vae.chameleon_ori_image_tokenizer.img_tokens_from_tensor(_img)
    image_toks = vae.chameleon_ori_translation.img2bpe_mapping_tensor[image_toks]
    h, w = _img.shape[-2:]
    h_grids, w_grids = h // vae.patch_size, w // vae.patch_size
    full_image_toks = image_toks.reshape(bs, h // 16, w // 16)
    new_line_id = vae.token2id(vae.new_line_token)

    full_image_toks = torch.cat(
        (
            full_image_toks,
            torch.full((bs, h // 16, 1), fill_value=new_line_id,  device=full_image_toks.device, dtype=full_image_toks.dtype),
        ),
        dim=-1,
    ).flatten(start_dim=1, end_dim=-1)

    # TODO: Currently unused
    result_toks = torch.cat([
        torch.tensor([
            vae.token2id(vae.image_start_token),
            vae.token2id(vae.get_n_grids_token(h_grids)),
            vae.token2id(vae.get_n_grids_token(w_grids))
        ], device=full_image_toks.device, dtype=full_image_toks.dtype).unsqueeze(0).expand(bs, -1),
        full_image_toks,
        torch.tensor([
            vae.token2id(vae.image_end_token)
        ], device=full_image_toks.device, dtype=full_image_toks.dtype).unsqueeze(0).expand(bs, -1)
    ], dim=1)

    input_ids = torch.full((bs, config.model.length,), fill_value=-100, dtype=torch.int64)
    attention_mask = torch.full((bs, config.model.length,), fill_value=False, dtype=torch.bool)

    for i in range(len(all_content)):
        w, h = config.data.resolution, config.data.resolution
        _item = all_content[i]
        conversations = []
        first_text = True
        for item in _item:
            if item['type'] == "text":
                if first_text:
                    conversations.append({"from": "human", "value": f"Generate an image of {w}x{h} according to the following prompt:\n{item['text']}"})
                    first_text = False
                else:
                    conversations.append({"from": "human", "value": item["text"]})
            elif item['type'] == "image_url":
                conversations.append({"from": "gpt", "value": "<|image|>"})
            
        item = {
            "conversations": conversations,
            "image": [(full_image_toks[it["image_url"]["url"]], w, h) for it in _item if it["type"] == "image_url"],
        }

        tokens, labels = vae.process_item_json(item, out_flatten=False)
        idx = 0
        for j, token_or_media in enumerate(tokens):
            if isinstance(token_or_media, int):
                input_ids[i, idx:idx+1] = token_or_media
                idx += 1
            else:
                media_len = len(token_or_media["input_ids"])
                input_ids[i, idx:idx+media_len] = token_or_media["input_ids"]
                idx += media_len
        
        attention_mask[i, :idx] = True
        if idx >= config.model.length:
            gprint("WARNING!!!! Truncating input ids")
    
    all_attention_masks = attention_mask
    all_input_ids = input_ids

    return all_input_ids, all_attention_masks

def tokenize_chameleon(config, tokenizer, vae, batch, **kwargs):
    assert "idx" in batch
    txt_decoded = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    all_attention_masks = []
    all_input_ids = []
    for i in range(batch['input_ids'].shape[0]):
        _img = Image.fromarray((batch['img'][i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        _txt = txt_decoded[i][:100]

        tokens, labels = vae.process_item(_img, _txt, out_flatten=False)
        input_ids = []
        first_img_id = None
        for i, token_or_media in enumerate(tokens):
            if isinstance(token_or_media, int):
                input_ids.append(token_or_media)
            else:
                if len(input_ids) > (config.model.length - 1061):
                    gprint("WARNING!!!! Truncating input ids")
                    input_ids = input_ids[:-1061]
                input_ids += token_or_media["input_ids"]
                first_img_id = i
                
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        attention_mask = torch.zeros(config.model.length, dtype=torch.bool)
        attention_mask[:len(input_ids)] = True
        if len(input_ids) > config.model.length:
            gprint("WARNING!!!! Truncating input ids, this should not happen")
            input_ids = input_ids[:config.model.length]
            attention_mask = attention_mask[:config.model.length]

        input_ids = torch.cat([input_ids, -100 * torch.ones(config.model.length - len(input_ids), dtype=torch.int64)])
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)

    all_attention_masks = torch.stack(all_attention_masks, dim=0)
    all_input_ids = torch.stack(all_input_ids, dim=0)
    return all_input_ids, all_attention_masks

_tar_cache = {}
_tar_contents_cache = None

def process_tar_file(tar_filepath):
    try:
        with tarfile.open(tar_filepath) as tar:
            return tar_filepath, set(tar.getnames())
    except:
        return tar_filepath, set()

def get_cache(mapping, split, parent_dir):
    global _tar_contents_cache
    if _tar_contents_cache is not None:
        return _tar_contents_cache

    hostname = socket.gethostname()
    userhome = Path.home()
    cache_path = userhome / ".cache" / "unidisc" / f"{split}_tar_contents_cache.pkl"

    if cache_path.exists():
        print("Loading tar contents cache")
        with open(cache_path, 'rb') as f:
            _tar_contents_cache = pickle.load(f)
            return _tar_contents_cache
            
    unique_tar_filepaths = mapping['tar_filepath'].unique()
    if parent_dir is not None:
        for i in range(len(unique_tar_filepaths)):
            orig_tar_filepath = Path(unique_tar_filepaths[i])
            relative_path = orig_tar_filepath.relative_to(*orig_tar_filepath.parts[:len(Path(parent_dir).parts)])
            unique_tar_filepaths[i] = Path(parent_dir) / relative_path
            if not unique_tar_filepaths[i].exists():
                unique_tar_filepaths[i] = Path(parent_dir) / orig_tar_filepath.relative_to(*orig_tar_filepath.parts[:len(Path(parent_dir).parts) - 1])

    print(f"Building tar contents cache for {len(unique_tar_filepaths)} files. Example: {unique_tar_filepaths[:4]}")
    import multiprocessing as mp
    with mp.Pool() as pool:
        results = list(tqdm(
            pool.imap(process_tar_file, unique_tar_filepaths),
            total=len(unique_tar_filepaths),
            desc="Building tar contents cache"
        ))
    
    _tar_contents_cache = dict(results)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(_tar_contents_cache, f)
        
    return _tar_contents_cache

def load_image(tar_filepath, key, split, parent_dir) -> bytes:
    global _tar_contents_cache, _tar_cache
    image_path = f"{key}.jpg"

    if parent_dir is not None:
        orig_tar_filepath = Path(tar_filepath)
        relative_path = orig_tar_filepath.relative_to(*orig_tar_filepath.parts[:len(Path(parent_dir).parts)])
        tar_filepath = Path(parent_dir) / relative_path
        if not tar_filepath.exists():
            tar_filepath = Path(parent_dir) / orig_tar_filepath.relative_to(*orig_tar_filepath.parts[:len(Path(parent_dir).parts) - 1])

    if image_path not in _tar_contents_cache[tar_filepath]:
        raise ValueError(f"Image {image_path} not found in {tar_filepath}")

    hostname = socket.gethostname()
    if parent_dir is None and "babel" in hostname:
        tar_filepath = tar_filepath.replace("/scratch", "/other_path")

    if tar_filepath not in _tar_cache:
        _tar_cache[tar_filepath] = tarfile.open(tar_filepath)

    tar = _tar_cache[tar_filepath]
    with tar.extractfile(image_path) as f:
        buffered_reader = io.BufferedReader(f)
        return buffered_reader.read()

def cleanup_tar_cache():
    for tar in _tar_cache.values():
        tar.close()
    _tar_cache.clear()
    _tar_contents_cache.clear()

def get_mmc4(config, tokenizer, vae, batch, device, mapping):
    split = "fewer_faces" if "fewer_faces" in getattr(config.data, "raw_data_dir") else "core"
    parent_dir = Path(getattr(config.data, "mmc4_parent_dir", None))
    get_cache(mapping, split, parent_dir)

    all_images = []
    image_idx = 0
    all_content = []
    remove_instances_missing_images = False
    before_ratio = 0.8
    for jsonl_row in batch:
        stat_counter = defaultdict(int)
        text_list = jsonl_row["text_list"]
        images_insert_before_text = [ [] for _ in range(len(text_list)) ]
        images_insert_after_text = [ [] for _ in range(len(text_list)) ]

        for image_info in jsonl_row["image_info"]:
            # randomly decide whether to prepend or append the image to the corresponding text
            insert_before = random.random() < before_ratio
            try:
                mapped_to_ = mapping.loc[image_info["raw_url"]]
                if isinstance(mapped_to_, pd.Series):
                    mapped_to_ = [mapped_to_]
                elif isinstance(mapped_to_, pd.DataFrame):
                    mapped_to_ = [row for _, row in mapped_to_.iterrows()]
                else:
                    mapped_to_ = [mapped_to_]

            except KeyError as e:
                if remove_instances_missing_images:
                    stat_counter["instance_skipped_due_to_missing_image"] += 1
                    break  # skip this instance
                else:
                    stat_counter["n_missing_images"] += 1
                    continue # skip this image
            
            success = False
            for mapped_to in mapped_to_:
                try:
                    tar_filepath = mapped_to["tar_filepath"]
                    key = mapped_to["key"]
                except Exception as e:
                    print(f"V2 Error mapping key to path: {e}")
                    continue
                
                try:
                    image_bytes = load_image(tar_filepath, key, split, parent_dir)
                except Exception as e:
                    # print(f"Failed to read key: {key}, {e}")
                    if remove_instances_missing_images:
                        stat_counter["instance_skipped_due_to_missing_image"] += 1
                        break  # skip this instance
                    else:
                        stat_counter["n_missing_images"] += 1
                        continue # skip this image
                    
                image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                success = True
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": image_idx}
                }
                image_idx += 1
                all_images.append(image_pil)
                stat_counter["n_images_inserted"] += 1

                if insert_before:
                    stat_counter["n_images_inserted_before_text"] += 1
                    images_insert_before_text[image_info["matched_text_index"]].append(image_content)
                else:
                    stat_counter["n_images_inserted_after_text"] += 1
                    images_insert_after_text[image_info["matched_text_index"]].append(image_content)
                
                break

            if not success:
                print(f"Failed find image: {key}")

        # flatten content: list of list of content -> list of content
        content = []
        for i, text in enumerate(text_list):
            content.extend(images_insert_before_text[i])
            content.append({"type": "text", "text": text})
            content.extend(images_insert_after_text[i])
        all_content.append(content)

    reordered_images = []
    old_to_new_idx = {}
    new_idx = 0
    for content_list in all_content:
        for item in content_list:
            if item["type"] == "image_url":
                old_idx = item["image_url"]["url"]
                if old_idx not in old_to_new_idx:
                    old_to_new_idx[old_idx] = new_idx
                    reordered_images.append(all_images[old_idx])
                    new_idx += 1

    batch_size_map = defaultdict(list)
    for i, content_list in enumerate(all_content):
        for item in content_list:
            if item["type"] == "image_url":
                old_idx = item["image_url"]["url"]
                item["image_url"]["url"] = old_to_new_idx[old_idx]
                batch_size_map[i].append(item["image_url"]["url"])

    all_images = reordered_images

    return all_images, all_content, batch_size_map

def tokenize_regular_cambrian_mmc4(config, tokenizer, vae, batch, device, mapping, inference_data=False,**kwargs):
    """Use this for MMC4 and Cambrian. Ignore the other functions below."""
    is_cambrian = config.data.train == "cambrian"
    if inference_data:
        breakpoint()
    elif is_cambrian:
        all_images = []
        all_content = batch
        parent_path = Path(getattr(config.data, "cambrian_path", "/cambrian_base_path"))
        batch_size_map = defaultdict(list)
        for i in range(len(batch)):
            if "image" in batch[i]:
                img = Image.open(parent_path / batch[i]["image"]).convert("RGB")
                all_images.append(img)
                batch_size_map[i].append(i)
    else:
        all_images, all_content, batch_size_map = get_mmc4(config, tokenizer, vae, batch, device, mapping)
        if len(all_images) == 0:
            gprint(f"No images, skipping...")
            return None, None, None
        
        output_list = []
        for input_data in all_content:
            conversations = []
            current_human_text = ""
            images_to_prepend = []
            text_counter = 0
            has_image = False
            for item in input_data:
                if item['type'] == 'image_url':
                    if text_counter == 0:
                        images_to_prepend.append('<image>')
                    else:
                        if current_human_text:
                            current_human_text += '<image>'
                            conversations.append({'from': 'human', 'value': current_human_text})
                            current_human_text = ""
                        else:
                            if conversations and conversations[-1]['from'] == 'human':
                                conversations[-1]['value'] += '<image>'
                            else:
                                images_to_prepend.append('<image>')
                    has_image = True
                elif item['type'] == 'text':
                    text_counter += 1
                    if current_human_text:
                        current_human_text += ' ' + ' '.join(images_to_prepend) + ' ' + item['text']
                        images_to_prepend = []
                    else:
                        current_human_text = ''.join(images_to_prepend) + ' ' + item['text']
                        images_to_prepend = []
                        
            if current_human_text or images_to_prepend:
                if current_human_text:
                    current_human_text += ' ' + ' '.join(images_to_prepend)
                else:
                    current_human_text = ' '.join(images_to_prepend)
                conversations.append({'from': 'human', 'value': current_human_text.strip()})
            
            _kwargs = {}
            if has_image:
                _kwargs['image'] = {}

            output_list.append({
                "id": "1",
                "conversations": conversations,
                **_kwargs
            })

        all_content = output_list
            
    _res = config.data.resolution
    _length = config.model.length
    if is_cambrian and len(all_images) == 0:
        image_ids = None
    else:
        _img = torch.cat([tensor_center_crop(torch.from_numpy(np.array(img))[None, :].permute(0, 3, 1, 2) / 255, (_res, _res)) for img in all_images])
        from model import get_image_batch
        try:
            batch_size = 32
            image_ids = []
            for i in range(0, len(_img), batch_size):
                batch = _img[i:i+batch_size]
                batch_ids = get_image_batch(config, vae, {"img": batch}, device)
                image_ids.append(batch_ids)
            image_ids = torch.cat(image_ids)
        except Exception as e:
            gprint(f"{_img.shape}, {e}")
            import traceback
            traceback.print_exc()

    from unidisc.tokenizers.tokenize_interleaved import preprocess, _has_image
    all_input_ids = []
    all_attention_masks = []
    all_modality = []
    for i, sources in enumerate(all_content):
        has_image = _has_image(sources)
        sources = copy.deepcopy([e["conversations"] for e in [sources]])
        _image_ids = None
        if has_image:
            try:
                _image_ids = image_ids[batch_size_map[i]]
            except Exception as e:
                import traceback
                traceback.print_exc()
                gprint(f"Error in tokenize_regular_cambrian_mmc4: {e}")
                return None, None, None
        try:
            data_dict = preprocess(sources, tokenizer, has_image=has_image, image_ids=_image_ids)
        except Exception as e:
            import traceback
            traceback.print_exc()
            gprint(f"Error in preprocess: {e}")
            return None, None, None
        input_ids = data_dict["input_ids"][0]
        attention_mask = data_dict["attention_mask"][0]
        modality = data_dict["modality"][0]
        if (input_ids[-2:] == tokenizer.eos_token_id).all():
            input_ids = input_ids[:-1]
            attention_mask = attention_mask[:-1]
            modality = modality[:-1]

        if input_ids.shape[0] > _length:
            gprint(f"WARNING!!!! Truncating input ids: {input_ids.shape[0]} vs. {_length}")
            input_ids = input_ids[:_length]
            attention_mask = attention_mask[:_length]
            modality = modality[:_length]
        input_ids = torch.nn.functional.pad(input_ids, (0, _length - input_ids.shape[-1]), value=tokenizer.pad_token_id)
        attention_mask = torch.nn.functional.pad(attention_mask.bool(), (0, _length - attention_mask.shape[-1]), value=False)
        modality = torch.nn.functional.pad(modality, (0, _length - modality.shape[-1]), value=0)

        # We don't want to cut off an image.
        if modality[-1] == 1:
            is_image = modality == 1
            change_points = torch.where(is_image[:-1] != is_image[1:])[0] + 1
            if change_points.numel() > 0:
                # Get start of last contiguous image sequence
                start_pos = change_points[-1].item()
                modality[start_pos:] = 0
                attention_mask[start_pos:] = False
                input_ids[start_pos:] = tokenizer.pad_token_id

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_modality.append(modality)

    all_input_ids = torch.stack(all_input_ids)
    all_attention_masks = torch.stack(all_attention_masks)
    all_modality = torch.stack(all_modality)
    
    return all_input_ids, all_attention_masks, all_modality

def decode_ids_batched(vae, tokens, pad_token_id, **kwargs):
    all_text_ids, all_image_ids = [], []
    num_text_tokens = 0
    for b in tokens:
        try:
            text_ids, image_ids = decode_ids(vae, b.tolist(), **kwargs)
        except Exception as e:
            breakpoint()
        all_image_ids.append(torch.tensor(image_ids))
        all_text_ids.append(torch.tensor(text_ids))
        num_text_tokens = max(num_text_tokens, all_text_ids[-1].shape[-1])
        
    for i in range(len(all_text_ids)):
        all_text_ids[i] = torch.nn.functional.pad(all_text_ids[i], (0, num_text_tokens - all_text_ids[i].shape[-1]), value=pad_token_id) if all_text_ids[i].shape[-1] < num_text_tokens else all_text_ids[i]

    all_text_ids = torch.stack(all_text_ids)
    return all_text_ids.to(tokens.device), all_image_ids

def decode_ids(vae, tokens, return_tokens=False):
    try:
        generated_images = []
        generation_result_processed = []

        i = 0
        while i < len(tokens):
            token_id = tokens[i]
            if token_id == vae.token2id(vae.image_start_token):
                cache = []
                for j in range(i + 1, len(tokens)):
                    if tokens[j] != vae.token2id(vae.image_end_token):
                        cache.append(tokens[j])
                        i = j + 1
                    else:
                        if return_tokens:
                            image = cache
                        else:
                            try:
                                image = vae.decode_image(cache)
                            except Exception as e:
                                rprint(f"Failed to decode image: len: {len(cache)}, E: {e}")
                        
                        generated_images.append(image)
                        generation_result_processed.append(vae.token2id("<|image|>"))
                        i = j + 1
                        break
            else:
                generation_result_processed.append(token_id)
                i += 1

        if return_tokens:
            rprint(f"generation_result_processed: {generation_result_processed[:50]}")
            generated = generation_result_processed
        else:
            try:
                generated = vae.tokenizer.decode(generation_result_processed)
            except:
                generated = None
                rprint("Failed to decode text")
    except Exception as e:
        breakpoint()

    return generated, generated_images

def get_chameleon_images(vae, batch):
    start_img_token = vae.token2id(vae.image_start_token)
    end_img_token = vae.token2id(vae.image_end_token)
    all_images = []
    for i in range(batch["input_ids"].shape[0]):
        start_idx = (batch["input_ids"][i] == start_img_token).nonzero(as_tuple=True)[0].item()
        end_idx = (batch["input_ids"][i] == end_img_token).nonzero(as_tuple=True)[0].item()
        all_images.append(batch["input_ids"][[i], start_idx:end_idx])
    return all_images



def preprocess_image(sample, image_processor):
    """
    Convert images to tensors for training.
    Augmentations: random horizontal flip.
    Normalization handled by wds.
    """
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    return image

def preprocess_interleaved(
    sample,
    tokenizer,
    clip_processor,
    sim_threshold,
    min_num_images,
    max_num_images,
    max_tokens=256,
):
    
    Image.MAX_IMAGE_PIXELS = 1000000000
    N_CHANNELS = 3
    MIN_KB = 10

    """
    Preprocess an interleaved image-text sequence, either by calling preprocess_gpt_interleaved (if the sequence
    is ChatGPT-generated) or by preprocessing in this function (if the sequences is from MMC4).
    """
    info = sample

    sentences = info["text_list"]
    sim_matrix = info["similarity_matrix"]

    # load images first to find which ones are valid
    valid_images, valid_image_indices = [], []
    for i, sample_image in enumerate(info["image_info"]):
        print(i)
        if "image_base64" not in sample_image:
            # print(f"No image_base64 in sample_image")
            continue
        image_base64 = sample_image["image_base64"]
        rawbytes = base64.b64decode(image_base64)

        # filter to images >= 10KB
        if len(rawbytes) // 1000 <= MIN_KB:
            # print(f"Image {i} is too small")
            continue

        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
        valid_images.append(image)
        valid_image_indices.append(i)

    if len(valid_image_indices) == 0:
        raise ValueError("No images in sample")

    sim_matrix = np.array(sim_matrix)  # of shape images x sentences
    sim_matrix = sim_matrix[valid_image_indices]

    # negate the similarities to turn then into costs
    cost_matrix = -sim_matrix
    # find one to one assignements
    from scipy.optimize import linear_sum_assignment
    image_indices, sentence_indices = linear_sum_assignment(cost_matrix)

    images, sentence_ixs = [], []
    for i, sim_ix in zip(image_indices, sentence_indices):
        sim_score = sim_matrix[i][sim_ix]

        if sim_score < sim_threshold:
            continue

        images.append(valid_images[i])
        sentence_ixs.append(sim_ix)

    if len(images) == 0:
        raise ValueError("No images in sample after filtering")

    # preprocess and pad images
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), max_num_images))
    images_tensors = images_tensors[keep_ixs]
    sentence_ixs = [sentence_ixs[ix] for ix in keep_ixs]
    if len(images_tensors) < max_num_images:
        zero_padding = torch.zeros(
            (
                max_num_images - len(images_tensors),
                N_CHANNELS,
                images_tensors[0].shape[1],
                images_tensors[0].shape[2],
            ),
            dtype=torch.float,
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # preprocess and tokenize text
    # add in <image> and <eoc> tokens
    for ix in sentence_ixs:
        sentences[ix] = f"<|endofchunk|><image>{sentences[ix]}"
    text = " ".join(sentences)
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )
    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # reject sequences with too few images (after truncation)
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )
    if num_images < min_num_images:
        raise ValueError(f"Fewer than {min_num_images} images in sample")
    elif (
        num_images == 1 and random.random() <= 0.5
    ):  # 50% chance of keeping single image samples
        raise ValueError("Only one image in sample")

    # avoid the situation where there's one <image> token and it's at the end
    if (
        num_images == 1
        and text_tensor["input_ids"][:, -1]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    ):
        raise ValueError(
            "Only one image at the end of sample, so labels will all be -100"
        )

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )

# clip_processor = get_transform(512, 'train', True, False)
# preprocess_interleaved(x['.json'], tokenizer, clip_processor, 0.5, 1, 100, 20000)

if __name__ == "__main__":
    vae = ItemProcessor(target_size=512)

    from image_utils import Im
    bs = 1
    raw_img = Im.random().torch[None]
    _img = var_center_crop(raw_img, vae.crop_size_list, random_top_k=1)
    image_toks = vae.chameleon_ori_image_tokenizer.img_tokens_from_tensor(_img)
    image_toks = vae.chameleon_ori_translation.img2bpe_mapping_tensor[image_toks]
    h, w = _img.shape[-2:]
    h_grids, w_grids = h // vae.patch_size, w // vae.patch_size
    full_image_toks = image_toks.reshape(bs, h // 16, w // 16)
    new_line_id = vae.token2id(vae.new_line_token)

    full_image_toks = torch.cat(
        (
            full_image_toks,
            torch.full((bs, h // 16, 1), fill_value=new_line_id,  device=full_image_toks.device, dtype=full_image_toks.dtype),
        ),
        dim=-1,
    ).flatten(start_dim=1, end_dim=-1)

    result_toks = torch.cat([
        torch.tensor([
            vae.token2id(vae.image_start_token),
            vae.token2id(vae.get_n_grids_token(h_grids)),
            vae.token2id(vae.get_n_grids_token(w_grids))
        ], device=full_image_toks.device, dtype=full_image_toks.dtype).unsqueeze(0).expand(bs, -1),
        full_image_toks,
        torch.tensor([
            vae.token2id(vae.image_end_token)
        ], device=full_image_toks.device, dtype=full_image_toks.dtype).unsqueeze(0).expand(bs, -1)
    ], dim=1)

    img = (result_toks[0], 512, 512)
    output = vae.process_item(img, "hello", out_flatten=False)
    breakpoint()