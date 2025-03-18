from __future__ import annotations
from tensordict import tensorclass
import torch
from torch import nn
from typing import Optional
from unidisc.utils.tensor_utils import get_contiguous_blocks, get_interleaved_indices
from tensordict import TensorDict

@tensorclass
class InterleavedBatch:
    input_ids: torch.Tensor
    modality: torch.Tensor
    sample_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None

    def to_ragged_batch(self):
        data = []
        batch_indices, start_positions, end_positions = get_contiguous_blocks(self.sample_ids)
        first_sample_ids = self.sample_ids[batch_indices, start_positions]
        self.auto_batch_size_()
        for i in range(batch_indices.shape[0]):
            if first_sample_ids[i] == -1:
                continue
            data.append(self[batch_indices[i], start_positions[i]:end_positions[i]])

        return TensorDict.lazy_stack(data, dim=0)
    
    def to_elements(self):
        data = self.to_ragged_batch()
        new_data = []
        for i in range(data.shape[0]):
            new_data.append(InterleavedElement.from_raw(data[i]))
        return TensorDict.lazy_stack(new_data, dim=0)
    
    @classmethod
    def custom_from_dict(cls, data: TensorDict):
        new_dict = {}
        for field in cls.fields():
            if field.name in data:
                new_dict[field.name] = data[field.name]
        
        return cls(**new_dict)
    

@tensorclass
class InterleavedElement:
    txt_input_ids: Optional[list[torch.Tensor]] = None
    img_input_ids: Optional[list[torch.Tensor]] = None
    txt: Optional[torch.Tensor] = None
    img: Optional[torch.Tensor] = None
    img_pos_ids: Optional[torch.Tensor] = None
    batch_indices: Optional[torch.Tensor] = None
    start_positions: Optional[torch.Tensor] = None
    end_positions: Optional[torch.Tensor] = None
    raw_data: Optional[InterleavedBatch] = None

    @classmethod
    def from_raw(cls, interleaved_batch: InterleavedBatch):
        batch_indices, start_positions, end_positions = get_contiguous_blocks(interleaved_batch.modality[None])
        block_modality = interleaved_batch.modality[start_positions]
        
        img_input_ids = []
        txt_input_ids = []
        img_pos_ids = []
        for i in range(batch_indices.shape[0]):
            if block_modality[i] == 1:
                assert len(txt_input_ids) > 0
                img_input_ids.append(interleaved_batch.input_ids[start_positions[i]:end_positions[i]])
                img_pos_ids.append(len(txt_input_ids) - 1)
            else:
                txt_input_ids.append(interleaved_batch.input_ids[start_positions[i]:end_positions[i]])

        return cls(img_input_ids=img_input_ids, txt_input_ids=txt_input_ids, img_pos_ids=torch.tensor(img_pos_ids), batch_indices=batch_indices, start_positions=start_positions, end_positions=end_positions, raw_data=interleaved_batch)

    def to_list(self):
        txt_idx = 0
        img_idx = 0
        has_added_txt = False
        data = []
        modalities = []
        while txt_idx < len(self.txt_input_ids) or img_idx < len(self.img_input_ids):
            if not has_added_txt and txt_idx < len(self.txt_input_ids):
                data.append(self.txt_input_ids[txt_idx])
                modalities.append(0)
                has_added_txt = True
            elif img_idx < len(self.img_input_ids) and self.img_pos_ids[img_idx] == txt_idx:
                data.append(self.img_input_ids[img_idx])
                modalities.append(1)
                img_idx += 1
            else:
                has_added_txt = False
                txt_idx += 1

        return data, modalities