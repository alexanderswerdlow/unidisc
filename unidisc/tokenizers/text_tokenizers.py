
import einops
import hydra
import hydra.utils
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from constants import LIB_DIR, UNIDISC_DIR
from decoupled_utils import (Profiler, barrier, dprint, get_rank,
                             get_slurm_job_id, get_world_size, gprint,
                             is_main_process, print_memory, rank_zero_fn,
                             rprint, save_memory_profile, try_except)


class VggFaceTokenizer:
    def __init__(self, mask_token_id, v2=False):
        self.mask_token_id = mask_token_id
        self.v2 = v2
        self.idx_to_attr = [
            "Male",
            "Young",
            "Middle_Aged",
            "Senior",
            "Asian",
            "White",
            "Black",
            "Rosy_Cheeks",
            "Shiny_Skin",
            "Bald",
            "Wavy_Hair",
            "Receding_Hairline",
            "Bangs",
            "Sideburns",
            "Black_Hair",
            "Blond_Hair",
            "Brown_Hair",
            "Gray_Hair",
            "No_Beard",
            "Mustache",
            "5_o_Clock_Shadow",
            "Goatee",
            "Oval_Face",
            "Square_Face",
            "Round_Face",
            "Double_Chin",
            "High_Cheekbones",
            "Chubby",
            "Obstructed_Forehead",
            "Fully_Visible_Forehead",
            "Brown_Eyes",
            "Bags_Under_Eyes",
            "Bushy_Eyebrows",
            "Arched_Eyebrows",
            "Mouth_Closed",
            "Smiling",
            "Big_Lips",
            "Big_Nose",
            "Pointy_Nose",
            "Heavy_Makeup",
            "Wearing_Hat",
            "Wearing_Earrings",
            "Wearing_Necktie",
            "Wearing_Lipstick",
            "No_Eyewear",
            "Eyeglasses",
            "Attractive",
        ]
        if self.v2:
            self.idx_to_attr.insert(0, "Female")

    def batch_decode(self, tokens_list, show_mask_token=False):
        decoded_strs = []
        for tokens in tokens_list:
            if self.v2:
                assert len(tokens) == 48, f"Expected 49 tokens, got {len(tokens)}"
            else:
                assert len(tokens) == 47, f"Expected 48 tokens, got {len(tokens)}"

            example_str = []
            for attr in tokens:
                if 2 <= attr <= (49 if self.v2 else 48):
                    example_str.append(self.idx_to_attr[attr - 2])
                elif attr == self.mask_token_id:
                    if show_mask_token:
                        example_str.append("[MASK]")
                elif 0 <= attr < 2:
                    pass
                else:
                    gprint(f"Unknown attribute id: {attr}")

            decoded_strs.append("The person is " + ", and ".join(example_str))
        return decoded_strs

    @property
    def eos_token(self):
        return "END OF SENTENCE"

    @property
    def eos_token_id(self):
        return 999999