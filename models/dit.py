import math
import typing
from contextlib import nullcontext
import os

# Torch must be imported before flash-attn
from unidisc.utils.tensor_utils import get_contiguous_blocks, get_interleaved_indices
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina
from decoupled_utils import gprint, is_torch_xla_available, rprint
from models.standalone_rotary import flash_torch_apply_rotary_emb_torch

import huggingface_hub
import omegaconf
from einops import rearrange

is_xla_available = is_torch_xla_available()

force_cudnn_spda_context = os.environ.get("UNIDISC_FORCE_CUDNN_SPDA_CONTEXT", "0") == "1"
allow_any_spda = os.environ.get("UNIDISC_ALLOW_ANY_SPDA", "0") == "1"
force_xla_flash_attention = os.environ.get("UNIDISC_FORCE_XLA_FLASH_ATTENTION", "0") == "1"
use_non_packed_fa2 = os.getenv("UNIDISC_USE_NON_PACKED_FA2", "0") == "1"
disable_flash_attention_3 = os.getenv("UNIDISC_FORCE_DISABLE_FA3", "0") == "1"
is_xla_linear_patched = os.getenv("UNIDISC_IS_XLA_LINEAR_PATCHED", "0") == "1"
use_causal_attn = os.getenv("UNIDISC_USE_CAUSAL_ATTN", "0") == "1"

if force_cudnn_spda_context: rprint("Forcing cudnn spda context")
if allow_any_spda: rprint("Allowing any spda")
if force_xla_flash_attention: rprint("Forcing xla flash attention")
if use_non_packed_fa2: rprint("Using non-packed Flash Attention 2!")
if disable_flash_attention_3: rprint("Disabling Flash Attention 3!")

try:
    failed_to_import_fa3 = True
    if disable_flash_attention_3 is False:
        from flash_attn_interface import flash_attn_func as flash_attn_func_v3, flash_attn_varlen_func as flash_attn_varlen_func_v3

        failed_to_import_fa3 = False
        rprint("Imported Flash Attention 3!")
except:
    rprint("Not using Flash Attention 3!")

try:
    import flash_attn.layers.rotary
    from flash_attn.layers.rotary import apply_rotary_emb

    if failed_to_import_fa3:
        from flash_attn.flash_attn_interface import (flash_attn_func,
                                                     flash_attn_qkvpacked_func,
                                                     flash_attn_varlen_func,
                                                     flash_attn_varlen_qkvpacked_func)
        rprint("Imported Flash Attention 2!")
except:
    rprint("Failed to import Flash Attention 2!")

try:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    from torch.nn.attention import SDPBackend, sdpa_kernel
except:
    pass

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    compiled_flex_attention = torch.compile(flex_attention)
except:
    pass

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

@torch.no_grad()
def get_transfusion_mask(B, N_tot, img_start_idx, img_length, modality):
    # todo - this is temporary and only works for [text+image] mode. does NOT handle interleaved (need to use modality_mask for this)
    # (B, N_tot) -> (B, N_tot, N_tot)
    rows, cols = torch.meshgrid(torch.arange(N_tot), torch.arange(N_tot), indexing="ij")
    idxs = torch.stack([rows, cols], dim=-1).to(modality.device)
    idxs = idxs.expand(B, -1, -1, -1)
    q_idx, kv_idx = idxs.unbind(dim=-1)

    offset = torch.full((B,), img_start_idx, device=modality.device).unsqueeze(-1).unsqueeze(-1)
    limit = torch.full((B,), img_length, device=modality.device).unsqueeze(-1).unsqueeze(-1)

    ar = q_idx >= kv_idx
    nar = (q_idx >= offset) & (kv_idx >= limit)
    mask = ar | nar

    # Assume that batches with all text are autoregressive only
    mask = torch.where(((modality == 0).all(dim=-1))[:, None, None], ar, mask)
    return mask

@torch.compiler.disable()
def add_img_data_to_blocks(input_emb, rotary_emb, modality_mask, sample_ids, add_data, img_count_embedding):
    """
    Dynamically adds 2D RoPE embeddings to image blocks. Handles variable resolutions by matching to hardcoded block sizes.
    """
    assert sample_ids is not None
    B, N = modality_mask.shape
    batch_indices, start_positions, end_positions = get_interleaved_indices(modality_mask)

    block_sizes = end_positions - start_positions
    unique_block_sizes = [size for size in torch.unique(block_sizes).tolist() if size in add_data.keys()]

    # For each block, count number of blocks before it within same sample_id group
    block_counts = torch.zeros_like(batch_indices)
    for i in range(len(batch_indices)):
        curr_sample_id = sample_ids[batch_indices[i], start_positions[i]]
        
        # Find blocks before this one with same batch index and sample_id
        prev_blocks_mask = (batch_indices[:i] == batch_indices[i]) & \
                          (sample_ids[batch_indices[:i], start_positions[:i]] == curr_sample_id)
        
        block_counts[i] = prev_blocks_mask.sum()

    for block_size in unique_block_sizes:
        block_mask = (block_sizes == block_size)
        block_indices = block_mask.nonzero(as_tuple=False).squeeze()
        if block_indices.ndim == 0:
            block_indices = block_indices.unsqueeze(0)

        if block_indices.numel() == 0:
            continue

        # Get the batch indices and start positions for these blocks
        batch_idx = batch_indices[block_indices]
        start_pos = start_positions[block_indices]
        img_idx = block_counts[block_indices]  # Get the block count for each selected block

        # Calculate the maximum valid length for each block (in case they exceed N)
        max_lengths = torch.clamp(N - start_pos, max=block_size)
        max_block_length = max_lengths.max().item()

        positions = start_pos.unsqueeze(1) + torch.arange(max_block_length, device=rotary_emb.device).unsqueeze(0) # [num_blocks, max_block_length]

        # Create a mask to handle blocks that may be shorter than block_size
        valid_mask = torch.arange(max_block_length, device=rotary_emb.device).unsqueeze(0) < max_lengths.unsqueeze(1)
        positions = positions * valid_mask  # Positions beyond valid lengths are set to zero
        batch_idx_expanded = batch_idx.unsqueeze(1).expand(-1, max_block_length)

        if input_emb is not None:
            input_emb_to_add_full = img_count_embedding[img_idx][:, None, :]  # Shape: [block_size]
            input_emb_to_add = input_emb_to_add_full.expand(-1, valid_mask.shape[-1], -1)
            input_emb_to_add = input_emb_to_add * valid_mask.unsqueeze(-1)  # Mask data beyond valid lengths
            input_emb[batch_idx_expanded[valid_mask], positions[valid_mask], :] = input_emb[batch_idx_expanded[valid_mask], positions[valid_mask], :] + input_emb_to_add[valid_mask]

        rotary_emb_to_add_full = add_data[block_size]  # Shape: [block_size]
        rotary_emb_to_add = rotary_emb_to_add_full[:max_block_length].unsqueeze(0).expand(batch_idx.size(0), -1, -1)
        rotary_emb_to_add = rotary_emb_to_add * valid_mask.unsqueeze(-1)  # Mask data beyond valid lengths
        rotary_emb[batch_idx_expanded[valid_mask], positions[valid_mask], :] = rotary_emb_to_add[valid_mask]

@torch.compiler.disable()
def add_txt_data_to_blocks(rotary_emb, modality_mask, sample_ids, add_data):
    assert sample_ids is not None
    batch_indices, start_positions, end_positions = get_contiguous_blocks(sample_ids)
    block_sizes = end_positions - start_positions
    for i in range(len(batch_indices)):
        batch_idx = batch_indices[i]
        start_pos = start_positions[i]
        block_size = block_sizes[i]
        sample_slice = slice(start_pos, start_pos+block_size)
        rotary_emb[batch_idx, sample_slice, :] = torch.where(modality_mask[batch_idx, sample_slice, None], rotary_emb[batch_idx, sample_slice, :], add_data[:block_size])

def apply_xla_flash_attention_with_spmd(query_states, key_states, value_states, causal=False):
    from torch_xla.experimental.custom_kernel import flash_attention

    # q, k, v should all have the shape [B, n_head, S, head_dim]
    head_dim = query_states.size()[-1]
    query_states = query_states / math.sqrt(head_dim)

    # Our simplified version of decoder only model does not use any mask.
    # flash_attention will use the global_mesh set in the TrainDecoderOnlyFSDPv2.
    attn_output = flash_attention(query_states, key_states, value_states, causal=causal, partition_spec=("fsdp", None, None, None))
    return attn_output


def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs

    return ckpt_forward


# To avoid XLA issues
if is_xla_available:
    def _dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
        if p > 0.0:
            return F.dropout(input=x, p=p, training=training).to(torch.bfloat16)
        else:
            return x
else:
    def _dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
        if p > 0.0:
            return F.dropout(input=x, p=p, training=training)
        else:
            return x


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: typing.Optional[torch.Tensor],
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool,
    modality: typing.Optional[torch.Tensor] = None,
) -> torch.Tensor:

    out = _dropout(x=(x + bias) if bias is not None else x, p=prob, training=training)

    if scale is not None:
        out = scale * out

    if modality is not None:
        out = torch.where((modality == 1).unsqueeze(-1), out, _dropout(x, p=prob, training=training))

    if modality is not None:
        out = torch.where((modality == 1).unsqueeze(-1), out, x)

    if residual is not None:
        out = residual + out

    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)

    return _bias_dropout_add

# function overload
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift

def modulate_with_mask(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
    # Only images need time conditioning
    return torch.where(modality.unsqueeze(-1) == 1, x * (1 + scale) + shift, x)

if is_xla_available:
    def bias_dropout_add_scale_fused_train(
        x: torch.Tensor, bias: typing.Optional[torch.Tensor], scale: typing.Optional[torch.Tensor], residual: typing.Optional[torch.Tensor], prob: float, modality: typing.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return bias_dropout_add_scale(x, bias, scale, residual, prob, True, modality)

    def bias_dropout_add_scale_fused_inference(
        x: torch.Tensor, bias: typing.Optional[torch.Tensor], scale: typing.Optional[torch.Tensor], residual: typing.Optional[torch.Tensor], prob: float, modality: typing.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return bias_dropout_add_scale(x, bias, scale, residual, prob, False, modality)

    def modulate_fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, modality: torch.Tensor=None) -> torch.Tensor:
        if modality is not None and modality.any():
            return modulate_with_mask(x, shift, scale, modality)
        return modulate(x, shift, scale)
else:
    @torch.jit.script
    def bias_dropout_add_scale_fused_train(
        x: torch.Tensor, bias: typing.Optional[torch.Tensor], scale: typing.Optional[torch.Tensor], residual: typing.Optional[torch.Tensor], prob: float, modality: typing.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return bias_dropout_add_scale(x, bias, scale, residual, prob, True, modality)


    @torch.jit.script
    def bias_dropout_add_scale_fused_inference(
        x: torch.Tensor, bias: typing.Optional[torch.Tensor], scale: typing.Optional[torch.Tensor], residual: typing.Optional[torch.Tensor], prob: float, modality: typing.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return bias_dropout_add_scale(x, bias, scale, residual, prob, False, modality)


    @torch.jit.script
    def modulate_fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, modality: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        if modality is not None and modality.any():
            return modulate_with_mask(x, shift, scale, modality)
        return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, seq_len, device=None):
        # seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # This makes the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)

        return self.cos_cached, self.sin_cached

    @staticmethod
    def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0, rope_scaling_factor: float = 1.0, ntk_factor: float = 1.0):
        """
        Precompute the frequency tensor for complex exponentials (cis) with
        given dimensions.

        This function calculates a frequency tensor with complex exponentials
        using the given dimension 'dim' and the end index 'end'. The 'theta'
        parameter scales the frequencies. The returned tensor contains complex
        values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
                Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex
                exponentials.
        """

        theta = theta * ntk_factor

        rprint(f"theta {theta} rope scaling {rope_scaling_factor} ntk {ntk_factor}")

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float().cuda() / dim))
        t = torch.arange(seq_len, device=freqs.device, dtype=torch.float)  # type: ignore
        t = t / rope_scaling_factor
        freqs = torch.outer(t, freqs).float()  # type: ignore
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        cos = cos[:, : cos.shape[-1] // 2]
        sin = sin[:, : sin.shape[-1] // 2]
        return cos, sin


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
    # cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    # sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)

#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast(x.device.type, enabled=False):
            x = F.layer_norm(x.float(), [self.dim])

        if is_xla_available:
            x = x.to(torch.bfloat16)
            if x.ndim == 3:
                return (x * self.weight[None, None, :]).to(torch.bfloat16)
            elif x.ndim == 2:
                return (x * self.weight[None]).to(torch.bfloat16)
        else:
            if x.ndim == 3:
                return x * self.weight[None, None, :]
            elif x.ndim == 2:
                return x * self.weight[None]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(x_skip.view(-1, dim_out), x.view(-1, dim_in), W.T, alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True)
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = (t[:, None].float() * freqs[None] if t.ndim == 1 else t[..., None].float() * freqs[None, None]) # TODO @sid I think this is right but remind me if things aren't working
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedderCFG(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def get_norm(*args, norm_type="layernorm", elementwise_affine=False, **kwargs):
    if norm_type == "layernorm":
        return LayerNorm(*args, **kwargs)
    elif norm_type == "rms":
        return RMSNorm(*args, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def get_linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        dropout=0.1,
        cross_attn=False,
        attn_type="flash",
        is_compiled=False,
        force_varlen_attn=False,
        force_cast_bf16=False,
        qk_norm=False,
        use_flash_attn_3=False,
        use_spda_attn=False,
        compile_flag_pos_emb=False,
        causal=False,
        use_kv_cache=False,
        time_conditioning=False,
        use_flex_attention=False,
        idx=None,
        attn_dropout=None
    ):
        super().__init__()
        self.cross_attn = cross_attn
        self.attn_type = attn_type
        self.force_varlen_attn = force_varlen_attn
        self.is_compiled = is_compiled
        self.compile_flag_pos_emb = compile_flag_pos_emb
        self.n_heads = n_heads
        self.force_cast_bf16 = force_cast_bf16
        self.qk_norm = qk_norm
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.use_flash_attn_3 = use_flash_attn_3
        self.use_spda_attn = use_spda_attn
        self.causal = causal
        self.use_kv_cache = use_kv_cache
        self.time_conditioning = time_conditioning
        self.use_flex_attention = use_flex_attention
        self.idx = idx
        self.attn_dropout = attn_dropout
        if self.attn_dropout is None:
            self.attn_dropout = 0
        
        self.old_start_pos = None

        self.attn_qkv = get_linear(dim, 3 * dim, bias=False)

        if self.cross_attn:
            self.attn_qkv_cond = get_linear(dim, 3 * dim, bias=False)

        self.attn_out = get_linear(dim, dim, bias=False)

        if self.qk_norm:
            self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)
            assert self.cross_attn is False

        self.softmax_scale = None

        if self.use_flash_attn_3 or self.use_spda_attn:
            assert self.attn_type == "flash" and self.force_varlen_attn is False
            assert self.cross_attn is False

        if self.use_flex_attention:
            assert self.attn_type == "flash" and self.use_spda_attn
            assert allow_any_spda is False
            assert self.softmax_scale is None

        self.use_flex_attention_cache = False
        self.warn_cache_dtype = True

    def update_kv_cache(self, q, new_k, new_v, batch_size, start_pos, seq_len):
        self.cache_k[:, start_pos : start_pos + seq_len] = new_k
        self.cache_v[:, start_pos : start_pos + seq_len] = new_v
        k = self.cache_k[:, :start_pos + seq_len] # (batch_size, cache_len + seq_len, nheads, headdim)
        v = self.cache_v[:, :start_pos + seq_len] # (batch_size, cache_len + seq_len, nheads, headdim)
        return q, k, v # q is (batch_size, seq_len, nheads*headdim)

    def reset_kv_cache(self, batch_size, seq_len, dtype, device, set_to_none=False):
        assert self.use_kv_cache
        if set_to_none:
            del self.cache_k
            del self.cache_v
            self.cache_k = None
            self.cache_v = None
        else:
            self.cache_k = torch.zeros(
                batch_size, seq_len, self.n_heads, self.head_dim, dtype=dtype, device=device
            )
            self.cache_v = torch.zeros(
                batch_size, seq_len, self.n_heads, self.head_dim, dtype=dtype, device=device
            )
    
    def set_flex_attention_cache(self, batch_size, seq_len, device, dtype):
        assert self.use_flex_attention
        self.use_flex_attention_cache = True
        self.cache_k = torch.zeros(batch_size, self.n_heads, seq_len, self.head_dim, device=device, dtype=dtype)
        self.cache_v = torch.zeros(batch_size, self.n_heads, seq_len, self.head_dim, device=device, dtype=dtype)

    def forward(
        self,
        x,
        x_cond=None,
        x_skip=None,
        rotary_cos_sin=None,
        cu_seqlens=None,
        max_seqlen_in_batch=None,
        bias_dropout_scale_fn=None,
        gate_msa=None,
        attention_mask=None,
        start_pos=None,
        modality=None,
        block_mask=None,
        update_cache_slice=None,
    ):
        if x.ndim == 2:
            batch_size, seq_len = 1, x.shape[0]
            has_batch_dim = False
        else:
            batch_size, seq_len = x.shape[0], x.shape[1]
            has_batch_dim = True

        if is_xla_linear_patched:
            x = x.to(torch.float32)

        qkv = self.attn_qkv(x)
        if self.use_kv_cache and start_pos is not None:
            if not self.cache_k.dtype == self.cache_v.dtype == qkv.dtype:
                self.cache_k = self.cache_k.to(qkv.dtype)
                self.cache_v = self.cache_v.to(qkv.dtype)
                
        if is_xla_linear_patched:
            qkv = qkv.to(torch.bfloat16)

        if self.cross_attn:
            qkv_cond = self.attn_qkv_cond(x_cond)

        if not has_batch_dim:
            if self.cross_attn:
                q = q.unsqueeze(0)
                kv = kv.unsqueeze(0)
            else:
                qkv = qkv.unsqueeze(0)

        # qkv now has b s (three h d)
        if self.qk_norm:
            if is_xla_available:
                if is_xla_linear_patched:
                    qkv_size = self.n_heads * self.head_dim
                    qkv = torch.cat(
                        [
                            self.q_norm(qkv[:, :, :qkv_size].to(torch.bfloat16)).to(torch.bfloat16),
                            self.k_norm(qkv[:, :, qkv_size : 2 * qkv_size].to(torch.bfloat16)).to(torch.bfloat16),
                            qkv[:, :, 2 * qkv_size :].to(torch.bfloat16),
                        ],
                        dim=-1,
                    ).to(torch.bfloat16)
                else:
                    qkv_size = self.n_heads * self.head_dim
                    qkv = torch.cat(
                        [self.q_norm(qkv[:, :, :qkv_size]), self.k_norm(qkv[:, :, qkv_size : 2 * qkv_size]), qkv[:, :, 2 * qkv_size :]], dim=-1
                    )
            else:
                qkv_size = self.n_heads * self.head_dim
                qkv[:, :, :qkv_size] = self.q_norm(qkv[:, :, :qkv_size])
                qkv[:, :, qkv_size : 2 * qkv_size] = self.k_norm(qkv[:, :, qkv_size : 2 * qkv_size])
            
        if rotary_cos_sin is not None:
            orig_dtype = qkv.dtype
            assert not (self.is_compiled and self.qk_norm is None)
            if cu_seqlens is not None and self.force_varlen_attn is False:
                assert not self.cross_attn, "Not yet supported"
                assert qkv.is_contiguous()
                qkv = rearrange(qkv, "b s (three h d) -> (b s) three h d", three=3, h=self.n_heads)
                qk = qkv[:, :2].reshape(seq_len, -1, self.head_dim)  # (b s) (two h) d
                with torch.autocast(x.device.type, enabled=False):
                    cos, sin = rotary_cos_sin
                    qk = apply_rotary_emb(
                        qk, cos.to(qkv.dtype), sin.to(qkv.dtype), inplace=True, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch
                    )
                qkv[:, :2] = qk.reshape(seq_len, 2, -1, self.head_dim)
            else:
                qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)
                if self.cross_attn:
                    qkv_cond = rearrange(qkv_cond, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)

                with torch.autocast(x.device.type, enabled=is_xla_available):
                    cos, sin = rotary_cos_sin

                    # TODO: This causes a ~4-8% slowdown on XLA
                    if self.compile_flag_pos_emb:
                        if is_xla_available:
                            if is_xla_linear_patched:
                                cos, sin, qkv = cos.to(torch.bfloat16), sin.to(torch.bfloat16), qkv.to(torch.bfloat16)
                                qk = qkv[:, :, :2].to(torch.bfloat16).reshape(batch_size, seq_len, -1, self.head_dim).to(torch.bfloat16)
                                qk = flash_torch_apply_rotary_emb_torch(qk, cos, sin)
                                qkv = qkv.clone()  # TODO: Appears to be needed for XLA
                                qkv = qkv.to(torch.bfloat16)
                                qkv[:, :, :2] = qk.to(torch.bfloat16).reshape(batch_size, seq_len, 2, -1, self.head_dim).to(torch.bfloat16)
                                qkv = qkv.to(torch.bfloat16)
                            else:
                                qk = qkv[:, :, :2].reshape(batch_size, seq_len, -1, self.head_dim)
                                qk = flash_torch_apply_rotary_emb_torch(qk, cos, sin).to(x)
                                qkv = qkv.clone()  # TODO: Appears to be needed for XLA
                                qkv[:, :, :2] = qk.reshape(batch_size, seq_len, 2, -1, self.head_dim)
                                qkv = qkv.to(x)
                        else:
                            qk = qkv[:, :, :2].reshape(batch_size, seq_len, -1, self.head_dim)
                            qk = flash_torch_apply_rotary_emb_torch(qk, cos, sin)
                            qkv[:, :, :2] = qk.reshape(batch_size, seq_len, 2, -1, self.head_dim)
                    else:
                        qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

                    if self.cross_attn:
                        qkv_cond = apply_rotary_pos_emb(qkv_cond, cos.to(qkv_cond.dtype), sin.to(qkv_cond.dtype))
                        qkv_cond = qkv_cond.to(orig_dtype)
                        q, _, _ = qkv.unbind(dim=2)
                        _, k_cond, v_cond = qkv_cond.unbind(dim=2)

                qkv = qkv.to(orig_dtype)
                if self.force_varlen_attn:
                    assert start_pos is not None
                    qkv = rearrange(qkv, "b s ... -> (b s) ...")
        else:
            assert not self.use_flash_attn_3
            if cu_seqlens is not None:
                assert False
            else:
                qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)

        if self.use_kv_cache:
            assert self.attn_type == "flash" and self.use_spda_attn and allow_any_spda is False and not self.use_flex_attention

        if self.attn_type == "flash":
            if cu_seqlens is None and self.force_varlen_attn is False:  # qkv: (batch_size, seqlen, 3, nheads, headdim)
                if self.use_flash_attn_3:
                    # We do not yet support flash attn 3 for cross attention
                    q, k, v = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :]
                    x = flash_attn_func_v3(
                        q, k, v, softmax_scale=self.softmax_scale, causal=self.causal
                    )[0]
                elif self.use_spda_attn:
                    if allow_any_spda:
                        b, s, _, h, d = qkv.shape
                        q, k, v = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :]
                        q = q.view(b, -1, h, d).transpose(1, 2)
                        k = k.view(b, -1, h, d).transpose(1, 2)
                        v = v.view(b, -1, h, d).transpose(1, 2)

                        if attention_mask is None:
                            with nullcontext() if allow_any_spda else sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]):
                                x = sdpa(q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=None, is_causal=self.causal)
                        else:
                            x = sdpa(q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=attention_mask, is_causal=self.causal)
                    else:
                        if is_xla_linear_patched:
                            qkv = qkv.to(torch.bfloat16)

                        q, k, v = qkv.unbind(dim=2)
                        disable_causal_attn = False
                        if self.use_kv_cache and start_pos is not None:
                            disable_causal_attn = True
                            q, k, v = self.update_kv_cache(q, k, v, batch_size, start_pos, seq_len)

                        is_causal = self.causal and not disable_causal_attn
                        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

                        if self.use_flex_attention:
                            # During inference we have a variable batch size which is not supported by torch.compile w/flex attention right now
                            # See: https://github.com/pytorch/pytorch/issues/136196
                            if self.training:
                                x = compiled_flex_attention(q, k, v, block_mask=block_mask)
                            else:
                                # Step 0: We want full attention for joint img/txt update
                                # Step 1: We want txt -> (txt + img) attention and img -> img attention. Cache the img kv for the next step
                                # Step 2...N: We want txt -> (txt + img) attention, using the cached kv for img
                                if self.use_flex_attention_cache:
                                    if seq_len != self.cache_k.shape[2]: # Step 2
                                        assert update_cache_slice is not None
                                        # (B, H, S, D)
                                        self.cache_k[:, :, update_cache_slice] = k
                                        self.cache_v[:, :, update_cache_slice] = v
                                    elif block_mask is not None and block_mask is not True: # Step 1
                                        assert update_cache_slice is not None
                                        assert (update_cache_slice.stop - update_cache_slice.start) == k.shape[2]
                                        self.cache_k = k
                                        self.cache_v = v
                                    else: # Step 0
                                        pass
                                
                                assert block_mask is not None
                                
                                # Hack to set full attention when we explicitly want it
                                if block_mask is True:
                                    block_mask = None
                                x = flex_attention(q, k, v, block_mask=block_mask)
                        elif force_xla_flash_attention:
                            assert not is_causal, "XLA Flash Attention does not support causal attention"
                            x = apply_xla_flash_attention_with_spmd(q=q, k=k, v=v, causal=is_causal)
                        elif force_cudnn_spda_context:
                            with (
                                nullcontext()
                                if (is_xla_available or attention_mask is not None)
                                else sdpa_kernel(backends=[
                                    SDPBackend.CUDNN_ATTENTION,
                                    *([] if (self.use_spda_attn and force_cudnn_spda_context) else [SDPBackend.FLASH_ATTENTION])
                                ])
                            ):
                                dropout_p = self.attn_dropout if self.training else 0
                                x = sdpa(q, k, v, attn_mask=None, is_causal=is_causal, scale=self.softmax_scale, dropout_p=dropout_p)
                        else:
                            dropout_p = self.attn_dropout if self.training else 0
                            x = sdpa(q, k, v, attn_mask=attention_mask, is_causal=is_causal, scale=self.softmax_scale, dropout_p=dropout_p)

                        if is_xla_linear_patched:
                            x = x.to(torch.bfloat16)

                elif self.cross_attn:
                    x = flash_attn_func(q, k_cond, v_cond, dropout_p=0.0, softmax_scale=self.softmax_scale, causal=self.causal)
                else:
                    if use_non_packed_fa2:
                        q, k, v = qkv.unbind(dim=2)
                        x = flash_attn_func(
                            q, k, v, dropout_p=0.0, softmax_scale=self.softmax_scale, causal=self.causal
                        )
                    else:
                        x = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=self.softmax_scale, causal=self.causal)

                if self.use_spda_attn:
                    x = rearrange(x, "b h s d -> b s (h d)", b=batch_size)
                else:
                    x = rearrange(x, "b s h d -> b s (h d)", b=batch_size)
            else:
                if cu_seqlens is None:
                    cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=qkv.device)

                # If we want all *other* ops to be FP32, we still need to cast the input for attn to BF16 as Flash Attn only supports FP16/BF16. This is a quick hack to do this.
                with torch.amp.autocast(x.device.type, dtype=torch.bfloat16) if self.force_cast_bf16 else nullcontext():
                    if self.cross_attn:
                        if self.force_cast_bf16:
                            q = q.to(torch.bfloat16)
                            k_cond = k_cond.to(torch.bfloat16)
                            v_cond = v_cond.to(torch.bfloat16)
                        x = flash_attn_varlen_func(
                            q, k_cond, v_cond, cu_seqlens, seq_len, dropout_p=0.0, softmax_scale=self.softmax_scale, causal=self.causal
                        )
                    else:
                        if self.force_cast_bf16:
                            qkv = qkv.to(torch.bfloat16)
                        x = flash_attn_varlen_qkvpacked_func(
                            qkv, cu_seqlens, seq_len, dropout_p=0.0, softmax_scale=self.softmax_scale, causal=self.causal
                        )
                x = rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)

        if not has_batch_dim:
            x = x.squeeze(0)

        if is_xla_linear_patched:
            x = x.to(torch.float32)

        if bias_dropout_scale_fn is not None:
            return bias_dropout_scale_fn(
                x=self.attn_out(x),
                bias=None,
                scale=gate_msa,
                residual=x_skip,
                prob=self.dropout,
                modality=(modality if self.time_conditioning else None),
            )
        else:
            return self.attn_out(x)


class DDiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        cond_dim,
        mlp_ratio=4,
        dropout=0.1,
        time_conditioning=True,
        img_cond=False,
        norm_type="layernorm",
        sandwich_normalization=False,
        **kwargs,
    ):
        super().__init__()
        self.time_conditioning = time_conditioning

        self.dropout = dropout
        self.attention = Attention(dim, n_heads, dropout, **kwargs)
        self.img_cond = img_cond
        if img_cond:
            self.cross_attention = Attention(dim, n_heads, dropout, cross_attn=True, **kwargs)

        self.norm1 = get_norm(dim, norm_type=norm_type)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = get_norm(dim, norm_type=norm_type)

        self.mlp = nn.Sequential(
            get_linear(dim, mlp_ratio * dim, bias=True), nn.GELU(approximate="tanh"), get_linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        if self.time_conditioning:
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

        self.sandwich_normalization = sandwich_normalization
        if self.sandwich_normalization:
            self.post_ff_norm = get_norm(dim, norm_type=norm_type)
            self.pre_residual_norm = get_norm(dim, norm_type=norm_type)
            assert self.img_cond is False, "Sandwich normalization is not supported with cross attention."
        else:
            self.pre_residual_norm = nn.Identity()
            self.post_ff_norm = nn.Identity()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def reset_kv_cache(self, *args, **kwargs):
        self.attention.reset_kv_cache(*args, **kwargs)
    
    def set_flex_attention_cache(self, *args, **kwargs):
        self.attention.set_flex_attention_cache(*args, **kwargs)

    def forward(
            self, 
            x, 
            rotary_cos_sin=None, 
            c=None, 
            cu_seqlens=None, 
            max_seqlen_in_batch=None, 
            x_cond=None, 
            attention_mask=None, 
            modality=None, 
            start_pos=None,
            block_mask=None,
            update_cache_slice=None,
        ):

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        if self.time_conditioning:
            _cond = self.adaLN_modulation(c)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (_cond if _cond.ndim == 3 else _cond[:, None, :]).chunk(6, dim=2)
        else:
            gate_msa, gate_mlp = None, None
        x_skip = x
        x = self.norm1(x)

        if self.time_conditioning:
            x = modulate_fused(x, shift_msa, scale_msa, modality)

        # Self Attention Start
        x = self.attention(
            x,
            rotary_cos_sin=rotary_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
            x_skip=x_skip,
            bias_dropout_scale_fn=None if self.sandwich_normalization else bias_dropout_scale_fn,
            gate_msa=gate_msa,
            attention_mask=attention_mask,
            modality=modality,
            start_pos=start_pos,
            block_mask=block_mask,
            update_cache_slice=update_cache_slice,
        )
        
        # Self Attention End
        if self.sandwich_normalization:
            x = x_skip + self.pre_residual_norm(x)


        # Cross Attention Start
        if self.img_cond:
            x = self.cross_attention(
                x,
                x_cond=x_cond,
                rotary_cos_sin=rotary_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=max_seqlen_in_batch,
                x_skip=x_skip,
                bias_dropout_scale_fn=bias_dropout_scale_fn,
                gate_msa=gate_msa,
            )
        # Cross Attention End

        # mlp operation
        _modality = (modality if self.time_conditioning else None)
        if self.time_conditioning:
            # assert not self.sandwich_normalization
            x = bias_dropout_scale_fn(
                x=self.post_ff_norm(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp, modality))),
                bias=None,
                scale=gate_mlp,
                residual=x,
                prob=self.dropout,
                modality=_modality,
            )
        else:
            x = bias_dropout_scale_fn(
                x=self.post_ff_norm(self.mlp(self.norm2(x))),
                bias=None,
                scale=None,
                residual=x,
                prob=self.dropout,
                modality=_modality,
            )

        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


def get_2d_rope(seq_len_2d, dim, linear_factor):
    seq_len_2d_side = int(math.sqrt(seq_len_2d))
    assert seq_len_2d_side**2 == seq_len_2d, f"seq_len_2d must be a square number, got {seq_len_2d}"
    if linear_factor is not None:
        rprint(f"Using Scale factor: {linear_factor}")
    ntk_factor = 1.0
    rotary_emb_2d = get_2d_rotary_pos_embed_lumina(
        dim,
        seq_len_2d_side,
        seq_len_2d_side,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
    )
    cos_2d_emb = rotary_emb_2d.flatten(0, 1).real
    sin_2d_emb = rotary_emb_2d.flatten(0, 1).imag
    return cos_2d_emb, sin_2d_emb

class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim, time_conditioning=True, norm_type="layernorm", zero_linear_init=True):
        super().__init__()
        self.time_conditioning = time_conditioning
        self.norm_final = get_norm(hidden_size, norm_type=norm_type)

        linear_kwargs = dict()
        self.linear = get_linear(hidden_size, out_channels, **linear_kwargs)
    
        if zero_linear_init:
            self.linear.weight.data.zero_()
            self.linear.bias.data.zero_()
        else:
            self.linear.bias.data.zero_()

        if self.time_conditioning:
            self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c, modality):
        if self.time_conditioning:
            _cond = self.adaLN_modulation(c)
            shift, scale = (_cond if _cond.ndim == 3 else _cond[:, None, :]).chunk(2, dim=2)
            x = modulate_fused(self.norm_final(x), shift, scale, modality)
        else:
            x = self.norm_final(x)

        x = self.linear(x)
        return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int, text_vocab_size: int, mask_index: int, dtype=None, device=None, static_img_sl=None, static_txt_sl=None, **kwargs):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        self.config = config
        self.autocast_dtype = dtype
        self.vocab_size = vocab_size
        self.text_vocab_size = text_vocab_size
        self.time_conditioning = config.time_conditioning or getattr(self.config.model, "force_time_conditioning", False)
        self.use_gradient_checkpointing = getattr(config.trainer, "use_gradient_checkpointing", False)
        self.img_cond = getattr(config.model, "img_cond", False)
        self.mask_index = mask_index
        self.force_cast_bf16 = (self.autocast_dtype == torch.float32)
        self.use_flash_attn_3 = getattr(config.model, "use_flash_attn_3", False)
        self.use_spda_attn = getattr(config.model, "use_spda_attn", False)
        self.compile_flag_pos_emb = getattr(config.trainer, "compile_flag_pos_emb", False)
        self.sandwich_normalization = getattr(config.model, "sandwich_normalization", False)
        self.use_kv_cache = getattr(config.model, "use_kv_cache", False)
        self.use_flex_attention = getattr(config.model, "use_flex_attention", False)
        self.static_img_sl = static_img_sl
        self.static_txt_sl = static_txt_sl
        self.causal = not config.model.full_attention

        if getattr(config.model, "use_flash_attn_3", False):
            assert not failed_to_import_fa3

        if getattr(self.config.model, "cond_label", False):
            self.y_embedder = LabelEmbedderCFG(1000, config.model.cond_dim, 0.1)

        if getattr(config.model, "use_pretrained_img_emb", False):
            from model import get_vae

            self.vocab_embed = EmbeddingLayer(config.model.hidden_size, text_vocab_size + 1)
            if getattr(config.model, "freeze_txt_emb", False):
                self.vocab_embed.requires_grad_(False)
            device = next(iter(self.vocab_embed.parameters())).device
            vae = get_vae(config, device)
            self.img_vocab_embed = vae.quantize.embedding
            if self.time_conditioning:  # TODO: Debug
                rprint("Requires grad: False")
                self.img_vocab_embed.requires_grad_(False)
            self.img_vocab_proj = get_linear(self.img_vocab_embed.embedding_dim, config.model.hidden_size)
            self.split_embed = True
            self.new_mask_index = text_vocab_size
            rprint(f"Using pretrained image embedding. Projecting from: {self.img_vocab_embed.embedding_dim} to {config.model.hidden_size}")
        else:
            self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
            self.split_embed = False

        self.is_compiled = getattr(config.trainer, "compile", False)
        if self.img_cond:
            if getattr(config.model, "use_pretrained_img_emb", False):
                cond_vae = get_vae(config, device, use_cond=True)
                self.cond_img_vocab_embed = cond_vae.quantize.embedding
                self.cond_img_vocab_proj = get_linear(self.cond_img_vocab_embed.embedding_dim, config.model.hidden_size)
            else:
                self.cond_img_vocab_embed = EmbeddingLayer(config.model.hidden_size, config.model.cond_image_vocab_size)

            img_cond_blocks = []
            for idx in range(8):
                img_cond_blocks.append(
                    DDiTBlock(
                        config.model.hidden_size,
                        config.model.n_heads,
                        config.model.cond_dim,
                        dropout=config.model.dropout,
                        img_cond=False,
                        time_conditioning=self.time_conditioning,
                        attn_type=config.model.attn_type,
                        is_compiled=self.is_compiled,
                        force_varlen_attn=config.model.force_varlen_attn,
                        force_cast_bf16=self.force_cast_bf16,
                        norm_type=config.model.norm_type,
                        qk_norm=config.model.qk_norm,
                        use_flash_attn_3=self.use_flash_attn_3,
                        use_spda_attn=self.use_spda_attn,
                        compile_flag_pos_emb=self.compile_flag_pos_emb,
                        sandwich_normalization=self.sandwich_normalization,
                        causal=not config.model.full_attention,
                        use_kv_cache=self.use_kv_cache,
                        use_flex_attention=self.use_flex_attention,
                        idx=idx,
                        attn_dropout=getattr(config.model, "attn_dropout", None),
                    )
                )
            self.img_cond_blocks = nn.ModuleList(img_cond_blocks)
            self.img_cond_rotary_emb = Rotary(config.model.hidden_size // config.model.n_heads)
            assert not self.is_compiled, "Need to fix rotary embeddings"

        self.sigma_map = None
        if self.time_conditioning and getattr(self.config.model, "cond_label", False) is False:
            self.sigma_map = TimestepEmbedder(config.model.cond_dim)
            rprint(f"Using timestep embedder with dim: {config.model.cond_dim}")

        self.use_legacy_rotary = False
        self.modality_embed = None

        if self.config.model.modality_embed:
            self.modality_embed = EmbeddingLayer(self.config.model.hidden_size, 2)

        continuous_mode = self.config.trainer.image_mode == "continuous"
        if continuous_mode:
            assert getattr(config.model, "vae_type", None) == "stable_diffusion"
            # an extra projection layer for the continuous diffusion
            self.continuous_img_proj = get_linear(4 * (config.model.patching_downscale ** 2), config.model.hidden_size) # todo remove 4 (vae hardcode)

        if self.config.model.rope_2d:
            seq_len_1d = self.config.model.txt_length
            seq_len_2d = self.config.model.img_length
            linear_factor = getattr(config.model, "linear_factor", 1.0)
            dim = config.model.hidden_size // config.model.n_heads

            if self.config.data.require_sample_ids:
                for seq_len_2d, linear_factor in ((256, 1), (1024, 2), (2304, 3), (4096, 4)):
                    cos_2d_emb, sin_2d_emb = get_2d_rope(seq_len_2d, dim, linear_factor)
                    self.register_buffer(f'rotary_cos_emb_img_{seq_len_2d}', cos_2d_emb, persistent=False)
                    self.register_buffer(f'rotary_sin_emb_img_{seq_len_2d}', sin_2d_emb, persistent=False)

                max_images_in_sequence = 16
                self.img_count_embedding = nn.Parameter(torch.zeros((max_images_in_sequence, config.model.hidden_size)))
            else:
                cos_2d_emb, sin_2d_emb = get_2d_rope(seq_len_2d, dim, linear_factor)
                self.register_buffer('rotary_cos_emb_img', cos_2d_emb, persistent=False)
                self.register_buffer('rotary_sin_emb_img', sin_2d_emb, persistent=False)

            rotary_emb_1d = Rotary(dim)(seq_len_1d)
            cos_1d_emb = rotary_emb_1d[0][0, :, 0, 0, : cos_2d_emb.shape[1]]
            sin_1d_emb = rotary_emb_1d[1][0, :, 0, 0, : sin_2d_emb.shape[1]]

            if self.config.trainer.multimodal_batches:
                seq_len_1d = self.config.model.length
                rotary_emb_1d = Rotary(config.model.hidden_size // config.model.n_heads)(seq_len_1d)
                cos_1d_emb = rotary_emb_1d[0][0,:,0, 0,: cos_2d_emb.shape[1]]
                sin_1d_emb = rotary_emb_1d[1][0,:,0, 0,: sin_2d_emb.shape[1]]
                self.register_buffer('rotary_cos_emb_txt', cos_1d_emb, persistent=False)
                self.register_buffer('rotary_sin_emb_txt', sin_1d_emb, persistent=False)
        else:
            seq_len_1d = self.config.model.length
            self.rotary_emb_1d = Rotary(config.model.hidden_size // config.model.n_heads)(seq_len_1d)
            cos_1d_emb = self.rotary_emb_1d[0][0,:,0, 0,: self.rotary_emb_1d[0].shape[-1] // 2]
            sin_1d_emb = self.rotary_emb_1d[1][0,:,0, 0,: self.rotary_emb_1d[1].shape[-1] // 2]
            self.register_buffer('rotary_cos_emb', cos_1d_emb, persistent=False)
            self.register_buffer('rotary_sin_emb', sin_1d_emb, persistent=False)

        blocks = []
        for idx in range(config.model.n_blocks):
            blocks.append(
                DDiTBlock(
                    config.model.hidden_size,
                    config.model.n_heads,
                    config.model.cond_dim,
                    dropout=config.model.dropout,
                    time_conditioning=self.time_conditioning,
                    img_cond=self.img_cond,
                    attn_type=config.model.attn_type,
                    is_compiled=self.is_compiled,
                    force_varlen_attn=config.model.force_varlen_attn,
                    force_cast_bf16=self.force_cast_bf16,
                    norm_type=config.model.norm_type,
                    qk_norm=config.model.qk_norm,
                    use_flash_attn_3=self.use_flash_attn_3,
                    use_spda_attn=self.use_spda_attn,
                    compile_flag_pos_emb=self.compile_flag_pos_emb,
                    sandwich_normalization=self.sandwich_normalization,
                    causal=not config.model.full_attention,
                    use_kv_cache=self.use_kv_cache,
                    use_flex_attention=self.use_flex_attention,
                    idx=idx,
                    attn_dropout=getattr(config.model, "attn_dropout", None),
                )
            )
        
        self.blocks = nn.ModuleList(blocks)
        self.output_layer = DDitFinalLayer(
            config.model.hidden_size,
            1 if config.parameterization == "planner" else vocab_size,
            config.model.cond_dim,
            time_conditioning=self.time_conditioning,
            norm_type=config.model.norm_type,
            zero_linear_init=config.model.zero_linear_init,
        )
        
        if continuous_mode:
            assert getattr(self.config.model, "vae_type", None) == "stable_diffusion"
            self.output_later_img = DDitFinalLayer(
                config.model.hidden_size,
                4 * (config.model.patching_downscale ** 2),  # todo, remove hardcoding
                config.model.cond_dim,
                time_conditioning=self.time_conditioning,
                norm_type=config.model.norm_type,
                zero_linear_init=config.model.zero_linear_init,
            )
            
        self.scale_by_sigma = config.model.scale_by_sigma
        self.txt_dropout = getattr(config.model, "txt_dropout", None)
        if config.parameterization != "ar":
            rprint(f"Not using AR, disabling txt dropout")
            self.txt_dropout = None

        self.txt_length = self.config.model.txt_length
        self.img_length = self.config.model.img_length
        self.total_length = self.config.model.length
        assert (self.txt_length + self.img_length == self.total_length) or self.config.trainer.multimodal_batches
        self.allow_compiled_embed = self.config.model.rope_2d is False and self.config.model.modality_embed is False and not getattr(self.config.model, "disable_allow_compiled_embed", False)
        self.multimodal_batches = self.config.trainer.multimodal_batches
        self.rope_2d = self.config.model.rope_2d
        rprint(f"DIT Found XLA: {is_xla_available}")
        self.require_sample_ids = self.config.data.require_sample_ids

        if self.config.model.force_optimized_native_attn:
            assert force_cudnn_spda_context
            assert self.config.model.use_spda_attn

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference
        
    def reset_kv_cache(self, *args, **kwargs):
        for block in self.blocks:
            block.reset_kv_cache(*args, **kwargs)

    def set_flex_attention_cache(self, *args, **kwargs):
        for block in self.blocks:
            block.set_flex_attention_cache(*args, **kwargs)

    def forward(
        self,
        indices,
        sigma=None,
        label=None,
        x_cond=None,
        attention_mask=None,
        continuous_mode=False,
        x_img_emb=None,
        modality=None,
        start_pos=None,
        block_mask=None,
        update_cache_slice=None,
        sample_ids=None,
    ):
        if self.txt_dropout is not None and self.training:
            mask = torch.rand_like(indices, dtype=torch.float) < self.txt_dropout
            indices = torch.where(mask & (modality == 0), self.mask_index, indices)

        if self.split_embed:
            # TODO: This is a bit inefficient
            text_mask = indices < self.text_vocab_size
            img_mask = (indices >= self.text_vocab_size) & (indices != self.mask_index)
            mask_token_mask = indices == self.mask_index

            text_indices = indices.clone()
            text_indices[~text_mask] = 0  # Set non-text tokens to 0
            text_indices[mask_token_mask] = self.new_mask_index
            txt_x = self.vocab_embed(text_indices)

            img_indices = indices.clone() - self.text_vocab_size
            img_indices[~img_mask] = 0  # Set non-image tokens to 0
            img_x = self.img_vocab_proj(self.img_vocab_embed(img_indices))

            mask_x = self.vocab_embed(torch.full_like(indices, self.new_mask_index))
            x = torch.where(text_mask.unsqueeze(-1), txt_x, torch.where(img_mask.unsqueeze(-1), img_x, mask_x))
        elif continuous_mode:
            assert sigma is not None
            text_embed = self.vocab_embed(indices)
            img_embed = self.continuous_img_proj(x_img_emb)
            x = torch.where(modality[:, :, None] == 1, img_embed, text_embed)
            attention_mask_shape = self.total_length if self.use_kv_cache else modality.shape[1]
            attention_mask = get_transfusion_mask(indices.shape[0], attention_mask_shape, self.txt_length, self.img_length, modality)
            if self.use_kv_cache:
                # we only care about (seq_len, cache_len+seq_len)
                assert self.total_length <= self.inference_max_seq_len
                seq_len = indices.shape[1]
                attention_mask = attention_mask[:, start_pos:start_pos+seq_len, :start_pos+seq_len]
                x = x[:, start_pos:start_pos+seq_len, :]
            attention_mask = attention_mask.unsqueeze(1).to(x.device)  # (B, 1, N_tot, N_tot) for SDPA
        else:
            x = self.vocab_embed(indices)
        x = x.to(self.autocast_dtype)
        c = None
        if self.sigma_map is not None:
            c = F.silu(self.sigma_map(sigma))

        if label is not None:
            assert c is None
            c = self.y_embedder(label, train=self.training)

        if x_cond is not None:
            assert not self.use_kv_cache
            if self.split_embed:
                x_cond = self.cond_img_vocab_proj(self.cond_img_vocab_embed(x_cond))
            else:
                x_cond = self.cond_img_vocab_embed(x_cond)

            img_cond_rotary_cos_sin = True if self.is_compiled else self.img_cond_rotary_emb(x_cond)
            img_cond_attention_args = (img_cond_rotary_cos_sin, None, None, None, None, attention_mask, start_pos)
            with torch.autocast(x_cond.device.type, dtype=self.autocast_dtype):
                for i in range(len(self.img_cond_blocks)):
                    x_cond = (
                        checkpoint(ckpt_wrapper(self.img_cond_blocks[i]), x_cond, *img_cond_attention_args, use_reentrant=True)
                        if (self.use_gradient_checkpointing and self.training)
                        else self.img_cond_blocks[i](x_cond, *img_cond_attention_args)
                    )

        if self.modality_embed is not None:
            if self.multimodal_batches:
                assert modality is not None
                try:
                    x = x + torch.where((modality == 0).unsqueeze(-1), self.modality_embed(0).unsqueeze(0).unsqueeze(0), self.modality_embed(1).unsqueeze(0).unsqueeze(0))
                except:
                    breakpoint()
            else:
                x[:, self.static_txt_sl] = x[:, self.static_txt_sl] + self.modality_embed(0).unsqueeze(0).unsqueeze(0)
                x[:, self.static_img_sl] = x[:, self.static_img_sl] + self.modality_embed(1).unsqueeze(0).unsqueeze(0)
        
        if self.is_compiled and self.allow_compiled_embed:
            rotary_cos_sin = True
        else:
            if self.use_legacy_rotary:
                rotary_cos_sin = self.rotary_emb(x)
            else:        
                if self.modality_embed is not None and self.rope_2d and self.multimodal_batches:
                    valid_sl = slice(start_pos, start_pos+x.shape[1]) if start_pos is not None else slice(None, x.shape[1])
                    if self.require_sample_ids:
                        assert modality.shape == indices.shape == sample_ids.shape
                        cos = torch.zeros((x.shape[0], *self.rotary_cos_emb_txt.shape), device=x.device, dtype=x.dtype)
                        sin = torch.zeros((x.shape[0], *self.rotary_sin_emb_txt.shape), device=x.device, dtype=x.dtype)
                        modality_mask = modality.bool()
                        @torch.compiler.disable()
                        def fn():
                            add_img_data_to_blocks(x, cos, modality_mask, sample_ids, {
                                256: self.rotary_cos_emb_img_256,
                                1024: self.rotary_cos_emb_img_1024,
                                2304: self.rotary_cos_emb_img_2304,
                                4096: self.rotary_cos_emb_img_4096
                            },  self.img_count_embedding)
                            add_img_data_to_blocks(None, sin, modality_mask, sample_ids, {
                                256: self.rotary_sin_emb_img_256,
                                1024: self.rotary_sin_emb_img_1024,
                                2304: self.rotary_sin_emb_img_2304,
                                4096: self.rotary_sin_emb_img_4096
                            }, None)
                            add_txt_data_to_blocks(cos, modality_mask, sample_ids, self.rotary_cos_emb_txt)
                            add_txt_data_to_blocks(sin, modality_mask, sample_ids, self.rotary_sin_emb_txt)

                        fn()
                        rotary_cos_sin = (cos, sin)
                    elif modality.shape[-1] != self.img_length:
                        # Pretty hacky but we want to support the following batch: [[text img], [text], [img]]
                        pad_size = modality.shape[-1] - self.img_length
                        pad_size = max(pad_size, 0)
                        padding = torch.full((1, pad_size, self.rotary_cos_emb_img.shape[-1]), torch.nan, device=x.device, dtype=x.dtype)
                        rotary_cos_sin = (
                            torch.where(modality[:, :, None] == 0, self.rotary_cos_emb_txt[None, valid_sl], torch.cat([padding, self.rotary_cos_emb_img[None, valid_sl]], dim=1)[:, valid_sl]).squeeze(0), 
                            torch.where(modality[:, :, None] == 0, self.rotary_sin_emb_txt[None, valid_sl], torch.cat([padding, self.rotary_sin_emb_img[None, valid_sl]], dim=1)[:, valid_sl]).squeeze(0)
                        )
                    else:
                        rotary_cos_sin = (
                            torch.where(modality[:, :, None] == 0, self.rotary_cos_emb_txt[None, valid_sl], self.rotary_cos_emb_img[None, valid_sl]).squeeze(0), 
                            torch.where(modality[:, :, None] == 0, self.rotary_sin_emb_txt[None, valid_sl], self.rotary_sin_emb_img[None, valid_sl]).squeeze(0)
                        )
                else:
                    rotary_cos_sin = (self.rotary_cos_emb, self.rotary_sin_emb)

            if start_pos is not None: assert self.use_kv_cache
            if self.use_kv_cache and start_pos is not None:
                cos, sin = rotary_cos_sin
                seq_len = x.shape[1]
                if cos.ndim == 3:
                    rotary_cos_sin = (
                        cos[:, start_pos:start_pos+seq_len],
                        sin[:, start_pos:start_pos+seq_len]
                    )
                elif cos.ndim == 2:
                    rotary_cos_sin = (
                        cos[start_pos:start_pos+seq_len],
                        sin[start_pos:start_pos+seq_len]
                    )
                else:
                    raise ValueError(f"Invalid rotary cos and sin shape for KV cache slicing: {cos.shape}")
                
        if self.causal and self.use_flex_attention and block_mask is None and not (self.use_kv_cache and start_pos is not None):
            # For causal, we do not need a mask if we are using KV cache
            block_mask = create_block_mask(causal_mask, B=None, H=None, Q_LEN=x.shape[1], KV_LEN=x.shape[1])
                
        attention_args = (rotary_cos_sin, c, None, None, x_cond, attention_mask, modality, start_pos, block_mask, update_cache_slice)
        with torch.autocast(x.device.type, dtype=self.autocast_dtype):
            for i in range(len(self.blocks)):
                x = (
                    checkpoint(ckpt_wrapper(self.blocks[i]), x, *attention_args, use_reentrant=True)
                    if (self.use_gradient_checkpointing and self.training)
                    else self.blocks[i](x, *attention_args)
                )
            
        if continuous_mode:
            x_img_emb = self.output_later_img(x, c, modality)
        
        x = self.output_layer(x, c, modality)
        
        if continuous_mode:
            return (x, x_img_emb)
        
        return x
