from typing import Dict, Sequence
import transformers
import copy
from unidisc.tokenizers import conversation as conversation_lib
import json
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import bisect
import os
import subprocess
from pathlib import Path
import socket

# Model Constants
DEFAULT_IMAGE_TOKEN = "<image>"

def tokenizer_image_token(prompt, tokenizer, return_tensors=None, image_ids=None, start_idx=None):
    prompt_chunks = [tokenizer(chunk, add_special_tokens=False).input_ids for chunk in prompt.split('<image>')]

    input_ids = []
    attention_mask = []
    modality = []

    start_idx = 0
    for i, chunk in enumerate(prompt_chunks):
        input_ids.extend(chunk)
        attention_mask.extend([True] * len(chunk))
        modality.extend([False] * len(chunk))
        
        if i < len(prompt_chunks) - 1:
            if image_ids is not None and start_idx < len(image_ids):
                input_ids.extend([tokenizer.additional_special_tokens_ids[tokenizer.additional_special_tokens.index("<image>")]])
                attention_mask.append(True)
                modality.append(False)

                input_ids.extend(image_ids[start_idx].tolist())
                attention_mask.extend([True] * len(image_ids[start_idx]))
                modality.extend([True] * len(image_ids[start_idx]))
                start_idx += 1

    if not input_ids[0] == tokenizer.bos_token_id:
        input_ids = [tokenizer.bos_token_id] + input_ids
        attention_mask = [True] + attention_mask
        modality = [False] + modality

    if not input_ids[-1] == tokenizer.eos_token_id:
        input_ids = input_ids + [tokenizer.eos_token_id]
        attention_mask = attention_mask + [True]
        modality = modality + [False]

    if return_tensors is not None:
        if return_tensors == 'pt':
            return (torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.bool), torch.tensor(modality, dtype=torch.bool)), start_idx
        raise ValueError(f'Unsupported tensor type: {return_tensors}')

    return (input_ids, attention_mask, modality), start_idx


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    image_ids = None,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]

            # if not role == conv.roles[j % 2]:
            #     print(f"Role mismatch at {i}, {j}: {role} vs {conv.roles[j % 2]}")

            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    print(f"After pre-processing: {conversations}")

    # Tokenize conversations
    if has_image:
        data = []
        start_idx = 0
        for i, prompt in enumerate(conversations):
            new_data, start_idx = tokenizer_image_token(prompt, tokenizer, return_tensors='pt', image_ids=image_ids, start_idx=start_idx)
            data.append(new_data)

        if not start_idx == len(image_ids):
            breakpoint()
            
        assert start_idx == len(image_ids), f"start_idx: {start_idx}, len(image_ids): {len(image_ids)}"

        input_ids = torch.stack([x[0] for x in data], dim=0)
        attention_mask = torch.stack([x[1] for x in data], dim=0)
        modality = torch.stack([x[2] for x in data], dim=0)
    else:
        data = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=8192,
            truncation=True,
        )
        attention_mask = data["attention_mask"]
        input_ids = data["input_ids"]
        modality = torch.zeros_like(attention_mask, dtype=torch.bool)

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        modality=modality
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    image_ids = None
) -> Dict:

    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    return preprocess_llama_2(sources, tokenizer, has_image=has_image, image_ids=image_ids)
  

def _has_image(sample: dict) -> bool:
    return "image" in sample and not str(sample['image']) in ['', 'None', 'none', 'nan']

def preprocess_multimodal(
    sources: Sequence[str],
) -> Dict:
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"
    is_multimodal = True
    mm_use_im_start_end = False
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

class JsonlDataset(Dataset):
    def __init__(self, glob_pattern, max_cache_size_gb=50, max_files=5):
        self.files = glob.glob(glob_pattern)
        self.files.sort()
        self.line_counts = []
        self.cumulative_sizes = []
        total = 0
        
        userhome = Path.home()
        print(f"Using {userhome} as cache directory")
        cache_dir = userhome / ".cache" / "unidisc"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "file_metadata.json"
        
        # Try to load cached metadata
        cached_metadata = {}
        try:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached_metadata = json.load(f)
        except Exception as e:
            print(f"Error loading cached metadata: {e}")

        print(f"Loaded cached metadata with {len(cached_metadata)} files")
        
        self.file_sizes = {}
        needs_update = False
        for filename in self.files:
            if filename in cached_metadata:
                count = cached_metadata[filename]['line_count']
                size = cached_metadata[filename]['file_size']
            else:
                print(f"Processing {filename}")
                result = subprocess.run(['wc', '-l', filename], capture_output=True, text=True)
                count = int(result.stdout.split()[0])
                size = os.path.getsize(filename)
                cached_metadata[filename] = {
                    'line_count': count,
                    'file_size': size
                }
                needs_update = True
                
            self.line_counts.append(count)
            total += count
            self.cumulative_sizes.append(total)
            self.file_sizes[filename] = size
            
        if needs_update and os.environ.get("SLURM_ARRAY_TASK_ID", None) is None:
            print(f"Writing cached metadata to {cache_file}")
            with open(cache_file, 'w') as f:
                json.dump(cached_metadata, f)
        
        self._cache = {}
        self.current_cache_size = 0
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
        self.max_files = max_files

        if len(self) == 0:
            print("No files to process")
            print(len(cached_metadata))
            print(cached_metadata)
            exit()
        else:
            print(f"Total len: {len(self)}")
    
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        # print(f"File idx: {file_idx}")
        if file_idx == 0:
            line_idx = idx
        else:
            line_idx = idx - self.cumulative_sizes[file_idx - 1]
        
        filename = self.files[file_idx]
        if filename in self._cache:
            lines = self._cache[filename]
        else:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # print(f"Adding {filename} to cache")
            # Remove files from cache until we have enough space and are under max files
            while ((self.current_cache_size + self.file_sizes[filename] > self.max_cache_size or
                   len(self._cache) >= self.max_files) and self._cache):
                removed_file = next(iter(self._cache))
                self.current_cache_size -= self.file_sizes[removed_file]
                self._cache.pop(removed_file)
            
            if (self.file_sizes[filename] <= self.max_cache_size and 
                len(self._cache) < self.max_files):
                self._cache[filename] = lines
                self.current_cache_size += self.file_sizes[filename]
        
        data = json.loads(lines[line_idx])
        data["idx"] = idx
        return data


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    tokenizer.model_max_length = 100000
    tokenizer.padding_side = 'right'
    tokenizer.add_eos_token = True

    i = 0
    dataset = JsonlDataset(glob_pattern="/scratch/aswerdlo/cambrian/jsons/gpt4v_77k.jsonl")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=lambda x: x)

    for batch in dataloader:
        image_paths = []
        for i in range(len(batch)):
            if "image" in batch[i]:
                image_paths.append(batch[i]["image"])

        image_ids = torch.zeros((len(image_paths), 256), dtype=torch.int64)
        for i, sources in enumerate(batch):
            has_image = _has_image(sources)
            sources = copy.deepcopy([e["conversations"] for e in [sources]])
            if has_image:
                sources = preprocess_multimodal(sources)

            data_dict = preprocess(sources, tokenizer, has_image=has_image, image_ids=image_ids[[i]])
            breakpoint()