# Copyright (2025) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adopted from https://github.com/QwenLM/Qwen2.5-VL. The original license is located at 'third-party-license/qwenvl.txt'.

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache, EncoderDecoderCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput, BaseModelOutput, CausalLMOutputWithCrossAttentions
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from qwenvl.model.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig, WhisperConfig
from qwenvl.model.modeling_whisper import WhisperEncoder

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss

if is_flash_attn_available():
    from transformers.modeling_flash_attention_utils import apply_rotary_emb, flash_attn_varlen_func


if is_flash_attn_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Qwen2_5_VLConfig"
IGNORE_INDEX = -100


import torch
import torch.nn.functional as F

def pad_and_stack(t1, t2, pad_value=0.0):
    # Check whether inputs are 2D or 3D
    is_3d = t1.dim() == 3 and t2.dim() == 3
    is_2d = t1.dim() == 2 and t2.dim() == 2

    assert is_2d or is_3d, "Inputs must be 2D or 3D tensors"

    if is_3d:
        B1, N1, D = t1.shape
        B2, N2, D = t2.shape
        assert B1 == 1 and B2 == 1, "Batch size must be 1"
    elif is_2d:
        B1, N1 = t1.shape
        B2, N2 = t2.shape
        assert B1 == 1 and B2 == 1, "Batch size must be 1"

    max_len = max(N1, N2)

    def pad_tensor(t, max_len):
        length = t.size(1)
        if length < max_len:
            pad_size = (0, 0, 0, max_len - length) if t.dim() == 3 else (0, max_len - length)
            t = F.pad(t, pad_size, "constant", pad_value)
        return t

    t1_padded = pad_tensor(t1, max_len)
    t2_padded = pad_tensor(t2, max_len)

    result = torch.cat([t1_padded, t2_padded], dim=0)
    return result


class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed


class Qwen2_5_VLVisionFlashAttention2(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


def _vision_varlen_sdpa(q, k, v, cu_seqlens):
    """Run SDPA per segment given cu_seqlens; equivalent to block-diagonal mask + one SDPA call,
    but without materializing the [1, L, L] mask.

    Materializing torch.zeros([1, L, L], dtype=bool) fails on long sequences:
      1) O(L^2) memory (e.g. bs=3, 256 frames gives L~1e5, ~14GB just for the mask);
      2) when L^2 exceeds INT32_MAX (2.147e9), 32-bit indexing in the CUDA kernel
         overflows ("CUDA error: unrecognized error code").
    The mask is block-diagonal along cu_seqlens (fully visible within a segment, invisible
    across segments), so per-segment full attention without a mask is mathematically
    identical and lets SDPA use its efficient backends.

    q/k/v: [num_heads, seq_len, head_dim]; returns the same shape.
    """
    num_heads, seq_len, head_dim = q.shape
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if lengths.numel() == 0:
        return q

    # Fast path: all segments equal length and covering the whole sequence (common when all frames share h*w); direct view, no gather
    if bool((lengths == lengths[0]).all()) and int(cu_seqlens[0]) == 0 and int(cu_seqlens[-1]) == seq_len:
        n, L = lengths.numel(), int(lengths[0])
        qs, ks, vs = (t.view(num_heads, n, L, head_dim).transpose(0, 1) for t in (q, k, v))
        o = F.scaled_dot_product_attention(qs, ks, vs, dropout_p=0.0)  # [n, H, L, D]
        return o.transpose(0, 1).reshape(num_heads, seq_len, head_dim)

    # General path: group segments by length and batch equal-length segments into one SDPA call (usually few distinct lengths)
    out = torch.empty_like(q)
    starts = cu_seqlens[:-1]
    for length in torch.unique(lengths):
        L = int(length)
        if L <= 0:
            continue
        sel = (lengths == length).nonzero(as_tuple=True)[0]
        idx = starts[sel].unsqueeze(1) + torch.arange(L, device=q.device).unsqueeze(0)  # [n, L]
        flat = idx.reshape(-1)
        n = sel.numel()
        qs, ks, vs = (t[:, flat].view(num_heads, n, L, head_dim).transpose(0, 1) for t in (q, k, v))
        o = F.scaled_dot_product_attention(qs, ks, vs, dropout_p=0.0)  # [n, H, L, D]
        out[:, flat] = o.transpose(0, 1).reshape(num_heads, n * L, head_dim)
    return out


class Qwen2_5_VLVisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        # Segmented SDPA instead of a dense [1, L, L] mask: avoids O(L^2) memory and INT32 index overflow on long sequences (large batch / many frames)
        attn_output = _vision_varlen_sdpa(q, k, v, cu_seqlens)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


QWEN2_5_VL_VISION_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLVisionAttention,
    "flash_attention_2": Qwen2_5_VLVisionFlashAttention2,
    "sdpa": Qwen2_5_VLVisionSdpaAttention,
}


class Qwen2_5_VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = QWEN2_5_VL_VISION_ATTENTION_CLASSES[attn_implementation](
            config.hidden_size, num_heads=config.num_heads
        )
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


Qwen2_5_VL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2_5_VLConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2_5_VL Model outputting raw hidden-states without any specific head on top.",
    Qwen2_5_VL_START_DOCSTRING,
)
class Qwen2_5_VLPreTrainedModel(PreTrainedModel):
    config_class = Qwen2_5_VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False  # TODO (joao): fix. torch.compile failing probably due to `cache_positions`

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()


class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VLPreTrainedModel):
    config_class = Qwen2_5_VLVisionConfig
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2_5_VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states

    def print_trainable_parameters(self) -> None:
        pass
        # """
@dataclass
class Qwen2_5_VLModelOutputWithPast(ModelOutput):
    """
    Base class for Llava outputs, with hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class Qwen2_5_VLRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2_5_VLAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2_5_VLFlashAttention2(Qwen2_5_VLAttention):
    """
    Qwen2_5_VL flash attention module, following Qwen2_5_VL attention module. This module inherits from `Qwen2_5_VLAttention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = flash_attn_supports_top_left_mask()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()


        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2_5_VLSdpaAttention(Qwen2_5_VLAttention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2_5_VLModel is using Qwen2_5_VLSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


QWEN2_5_VL_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLAttention,
    "flash_attention_2": Qwen2_5_VLFlashAttention2,
    "sdpa": Qwen2_5_VLSdpaAttention,
}


class Qwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_5_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@add_start_docstrings(
    "The bare Qwen2_5_VL Model outputting raw hidden-states without any specific head on top.",
    Qwen2_5_VL_START_DOCSTRING,
)
class Qwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vision_start_positions: Optional[torch.LongTensor] = None,
        vision_end_positions: Optional[torch.LongTensor] = None,
        event_id: Optional[torch.LongTensor] = None,
        output_attentions_layer_index: Optional[int] = None,
        output_hidden_states_layer_index: Optional[Union[int, List[int]]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:   # not taken in this pipeline
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

          
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions, 
            vision_start_positions=vision_start_positions,
            vision_end_positions=vision_end_positions,
            event_id=event_id,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        # If only specific layers' hidden_states are needed, store just those to save memory (accepts int or list; negative indices like -2 supported)
        num_hidden_slots = len(self.layers) + 1  # slots 0..num_layers-1 are pre-layer states; slot num_layers is post-norm
        if output_hidden_states and output_hidden_states_layer_index is not None:
            _idx = output_hidden_states_layer_index
            _indices = [_idx] if isinstance(_idx, int) else list(_idx)
            _requested = set((i + num_hidden_slots if i < 0 else i) for i in _indices)
            all_hidden_states = [None] * num_hidden_slots
        else:
            _requested = None
            all_hidden_states = () if output_hidden_states else None
        # If only one layer's attention is needed, request/store it only at that layer, saving memory (otherwise every layer stores [B, num_heads, L, L])
        if output_attentions and output_attentions_layer_index is not None:
            all_self_attns = [None] * len(self.layers)
        else:
            all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                if _requested is not None:
                    if layer_idx in _requested:
                        all_hidden_states[layer_idx] = hidden_states
                else:
                    all_hidden_states += (hidden_states,)

            output_attentions_this_layer = output_attentions and (
                output_attentions_layer_index is None or layer_idx == output_attentions_layer_index
            )
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions_this_layer,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions_this_layer,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:   # if this layer outputs attention weights the cache is the third return value, otherwise the second
                next_decoder_cache = layer_outputs[2 if output_attentions_this_layer else 1]

            if output_attentions:
                if output_attentions_layer_index is not None:
                    if layer_idx == output_attentions_layer_index:
                        all_self_attns[layer_idx] = layer_outputs[1]
                else:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer (index = num_layers)
        if output_hidden_states:
            if _requested is not None:
                if len(self.layers) in _requested:
                    all_hidden_states[len(self.layers)] = hidden_states
                all_hidden_states = tuple(all_hidden_states)
            else:
                all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        # Originally a tuple; a list means the layer-specific path was used, so convert back to tuple for consistency
        if output_attentions and isinstance(all_self_attns, list):
            all_self_attns = tuple(all_self_attns)
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
        vision_start_positions: Optional[torch.LongTensor] = None,
        vision_end_positions: Optional[torch.LongTensor] = None,
        event_id: Optional[torch.LongTensor] = None,
    ):
        if self.config._attn_implementation == "flash_attention_2" and getattr(self.config, 'use_flashattention', False):
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2_5_VL. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # However, if use_flashattention=False and vision-token mutual visibility is needed, we must build a mask instead of returning None
        need_vision_token_visible = (
            not getattr(self.config, 'use_flashattention', True)
            and vision_start_positions is not None
            and vision_end_positions is not None
        )
        
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                # If vision-token visibility is needed, we cannot return None; a mask must be built
                if not need_vision_token_visible:
                    return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
            vision_start_positions=vision_start_positions,
            vision_end_positions=vision_end_positions,
            event_id=event_id,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _allow_vision_tokens_mutual_visibility(
        causal_mask: torch.Tensor,
        config: Qwen2_5_VLConfig,
        cache_position: torch.Tensor,
        batch_size: int,
        past_key_values: Optional[Cache],
        vision_start_positions: Optional[torch.LongTensor],
        vision_end_positions: Optional[torch.LongTensor],
        sequence_length: int,
        target_length: int,
    ) -> torch.Tensor:
        """
        When use_flashattention=False, allow vision tokens to attend to each other in the causal_mask.
        Visibility is set per batch sample based on its vision token start/end positions.
        Caller must ensure this is only invoked when not use_flashattention and both
        vision_start/end_positions are not None.

        Accepts 2D or 4D input: a 4D [batch_size, 1, sequence_length, target_length] mask
        (shape after the event_id step) is squeeze(1)-ed to 3D before zeroing the vision region;
        a 2D mask is unsqueeze(0).expand-ed to 3D as before.

        Returns:
            Updated causal_mask of shape [batch_size, sequence_length, target_length]
        """
        device = causal_mask.device
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        
        # 4D input (after the event_id step): [batch_size, 1, sequence_length, target_length] -> [batch_size, sequence_length, target_length]
        if causal_mask.dim() == 4:
            causal_mask_batch = causal_mask.squeeze(1).clone()
        else:
            # Original 2D input: [sequence_length, target_length] -> [batch_size, sequence_length, target_length]
            causal_mask_batch = causal_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()

        # Convert vision_start_positions and vision_end_positions to tensors
        if vision_start_positions.dim() > 0:
            vision_start_abs = (vision_start_positions[:batch_size] + 1).to(device)  # (batch_size,)
            vision_end_abs = vision_end_positions[:batch_size].to(device)  # (batch_size,)
        else:
            # Single value: expand to all batch samples
            vision_start_abs = (vision_start_positions + 1).unsqueeze(0).expand(batch_size).to(device)  # (batch_size,)
            vision_end_abs = vision_end_positions.unsqueeze(0).expand(batch_size).to(device)  # (batch_size,)
        
        # Compute absolute key-side positions (shared across the batch)
        key_range = min(target_length, past_seen_tokens + sequence_length)
        if key_range == 0:
            return causal_mask_batch
        key_abs_positions = torch.full((key_range,), -1, dtype=torch.long, device=device)
        key_abs_positions[: min(past_seen_tokens, key_range)] = torch.arange(
            min(past_seen_tokens, key_range), device=device
        )
        num_rest = key_range - past_seen_tokens
        if num_rest > 0 and cache_position.shape[0] > 0:
            take = min(num_rest, cache_position.shape[0])
            key_abs_positions[past_seen_tokens : past_seen_tokens + take] = cache_position[:take].to(device)
        
        # Process each batch sample separately
        valid_query_len = min(sequence_length, cache_position.shape[0])
        if valid_query_len == 0:
            return causal_mask_batch
        abs_positions_q = cache_position[:valid_query_len].to(device)  # (valid_query_len,)
        
        for b in range(batch_size):
            vision_start_b = vision_start_abs[b].item()
            vision_end_b = vision_end_abs[b].item()
            
            # Check whether query positions fall inside this sample's vision range
            in_vision_q = (abs_positions_q >= vision_start_b) & (abs_positions_q < vision_end_b)  # (valid_query_len,)
            vision_query_indices = torch.where(in_vision_q)[0]  # (num_vision_queries,)
            
            # Check whether key positions fall inside this sample's vision range
            in_vision_k = (key_abs_positions >= vision_start_b) & (key_abs_positions < vision_end_b) & (key_abs_positions >= 0)  # (key_range,)
            vision_key_indices = torch.where(in_vision_k)[0]  # (num_vision_keys,)
            
            if vision_query_indices.numel() == 0 or vision_key_indices.numel() == 0:
                continue
            
            # Make this sample's vision tokens mutually visible: causal_mask[b, q, k] = 0
            q_idx = vision_query_indices.unsqueeze(1).expand(-1, vision_key_indices.numel()).reshape(-1)
            k_idx = vision_key_indices.unsqueeze(0).expand(vision_query_indices.numel(), -1).reshape(-1)
            causal_mask_batch[b, q_idx, k_idx] = 0
        
        return causal_mask_batch

    @staticmethod
    def _update_causal_mask_with_event_id(
        causal_mask: torch.Tensor,
        event_id: Optional[List[torch.Tensor]],
        sequence_length: int,
        target_length: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        min_dtype: float,
    ) -> torch.Tensor:
        """
        Update the causal_mask visibility rules according to event_id.

        Rule: query position q may attend to key position k iff both hold:
        1. k <= q (causality)
        2. event_id[k] == -1 (e.g. global tokens) or event_id[k] == event_id[q]

        Args:
            causal_mask: current causal mask, shape [batch_size, 1, sequence_length, target_length] (4D)
            event_id: list of length batch_size; each element is a tensor (possibly None) for one batch sample
            sequence_length: query sequence length
            target_length: key sequence length
            batch_size: batch size
            device: device
            dtype: data type
            min_dtype: minimum value used for masking

        Returns:
            Updated causal_mask, shape [batch_size, 1, sequence_length, target_length] (4D)
        """
        # Drop the head dim: [batch_size, 1, sequence_length, target_length] -> [batch_size, sequence_length, target_length]
        if causal_mask.dim() == 4:
            causal_mask_batch = causal_mask.squeeze(1).clone()

        if event_id is None or len(event_id) == 0:
            # No event_id: restore the head dim and return
            return causal_mask_batch.unsqueeze(1)  # [batch_size, 1, sequence_length, target_length]

        # event_id list length must equal batch_size
        if len(event_id) != batch_size:
            raise ValueError(f"event_id list length ({len(event_id)}) must equal batch_size ({batch_size})")

        # Position indices (shared across the batch)
        q_indices = torch.arange(sequence_length, device=device).unsqueeze(1)  # [sequence_length, 1]
        k_indices = torch.arange(target_length, device=device).unsqueeze(0)   # [1, target_length]

        # Process each batch sample separately
        for b in range(batch_size):
            if sequence_length > 1:  # training phase
                eid = event_id[b]

                # Skip samples with event_id None (keep the original causal_mask)
                if eid is None:
                    continue

                # Move event_id to the right device
                eid = eid.to(device)

                # Match event_id length to target_length
                valid_length = min(eid.shape[0], target_length) # lengths can genuinely differ due to batching, not just for safety
                if eid.shape[0] > target_length:
                    event_id_for_mask = eid[:target_length]
                else:
                    event_id_for_mask = eid

                # Per-position event_id
                # event_id_k: event_id of key position k, shape [1, target_length]
                # Valid positions (k < valid_length) use event_id_for_mask[k];
                # padding positions (k >= valid_length) use the sentinel -2 and are invisible
                event_id_k_full = torch.full((target_length,), -2, dtype=eid.dtype, device=device)  # -2 marks padding (invisible)
                event_id_k_full[:valid_length] = event_id_for_mask
                event_id_k = event_id_k_full.unsqueeze(0)  # [1, target_length]

                # event_id_q: event_id of query position q
                # Query positions should all be in range; clamp to the last valid value otherwise
                q_clamped = torch.clamp(q_indices.squeeze(1), 0, valid_length - 1)  # [sequence_length]
                event_id_q = event_id_for_mask[q_clamped].unsqueeze(1)  # [sequence_length, 1]

                # Event-match mask: event_id_k == -1 or event_id_k == event_id_q
                # Note: padding positions (event_id_k == -2) match nothing (neither -1 nor event_id_q), so they stay invisible
                # event_id_k: [1, target_length], event_id_q: [sequence_length, 1]
                # After broadcast: [sequence_length, target_length]
                event_match = (event_id_k == -1) | (event_id_k == event_id_q)  # [sequence_length, target_length]
                # Attention is allowed only when the key is global (event_id == -1) or in the same event group (equal event_id)
                # Combine with causality: k <= q
                k_le_q = k_indices <= q_indices  # [sequence_length, target_length]

                # Final event_mask: requires both causality (k <= q) and event match
                # Padding positions (event_id_k == -2) are automatically excluded since event_match is False
                event_mask = k_le_q & event_match  # [sequence_length, target_length]

                # Apply event_mask to this sample's causal_mask
                # Disallowed positions (event_mask False) are set to min_dtype
                # Padding positions keep their min_dtype (invisible) since event_match is False
                causal_mask_batch[b] = causal_mask_batch[b].masked_fill(~event_mask, min_dtype)
            else:   # inference generating phase
                # Update the causal mask only for this step's input token (the last row), layering the
                # event-match rule on top of the existing causal mask.
                # The input event_id covers all tokens in the sequence; its last element is this step's token.
                # That token may see all tokens with event_id -1, plus preceding tokens with the same event_id.
                eid = event_id[b]
                if eid is None:
                    continue
                eid = eid.to(device)

                # Valid key length (usually equals target_length; truncate for safety)
                valid_length = min(eid.shape[0], target_length)
                if valid_length <= 0:
                    # No valid tokens: whole row invisible
                    causal_mask_batch[b, -1, :] = min_dtype
                    continue

                # Build key-side event_id (padding positions use -2, invisible)
                event_id_k_full = torch.full((target_length,), -2, dtype=eid.dtype, device=device )
                event_id_k_full[:valid_length] = eid[:valid_length]  # keep the first valid_length entries
                # Current query token: event_id of the last valid token
                event_id_q = eid[valid_length - 1]

                # Event match: (-1 globally visible) or (same event_id)
                event_match_row = (event_id_k_full == -1) | (event_id_k_full == event_id_q)  # [target_length]

                # Causality: key position <= current query's absolute position (valid_length-1)
                k_le_q_row = torch.arange(target_length, device=device) <= (valid_length - 1)  # [target_length]

                # Final: requires both causality and event match (padding(-2) never matches)
                allow_row = event_match_row & k_le_q_row  # [target_length]

                # Update only the last query token's row
                causal_mask_batch[b, -1, :] = causal_mask_batch[b, -1, :].masked_fill(~allow_row, min_dtype)

        # Restore the head dim: [batch_size, sequence_length, target_length] -> [batch_size, 1, sequence_length, target_length]
        return causal_mask_batch.unsqueeze(1)

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2_5_VLConfig,
        past_key_values: Cache,
        vision_start_positions: Optional[torch.LongTensor] = None,
        vision_end_positions: Optional[torch.LongTensor] = None,
        event_id: Optional[List[torch.Tensor]] = None,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2_5_VLConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)    # long sequences: mask out tokens beyond the sliding window length
            causal_mask *= diagonal_attend_mask

            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if event_id is not None and all(x is not None for x in event_id):
                # Update causal_mask visibility according to event_id
                # Input and output shapes are both [batch_size, 1, sequence_length, target_length] (4D)
                # The function handles visibility per batch sample
                causal_mask = Qwen2_5_VLModel._update_causal_mask_with_event_id(
                    causal_mask=causal_mask,
                    event_id=event_id,
                    sequence_length=sequence_length,
                    target_length=target_length,
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                    min_dtype=min_dtype,
                )

            if not (getattr(config, 'use_flashattention', True) or vision_start_positions is None or vision_end_positions is None):
                # Returns shape [batch_size, sequence_length, target_length]
                causal_mask = Qwen2_5_VLModel._allow_vision_tokens_mutual_visibility(
                    causal_mask=causal_mask,
                    config=config,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    past_key_values=past_key_values,
                    vision_start_positions=vision_start_positions,
                    vision_end_positions=vision_end_positions,
                    sequence_length=sequence_length,
                    target_length=target_length,
                )
                # Add the head dim: [batch_size, sequence_length, target_length] -> [batch_size, 1, sequence_length, target_length]
                causal_mask = causal_mask.unsqueeze(1)

                
            if attention_mask is not None:  # handle visibility of padded tokens
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        
        return causal_mask

    def print_trainable_parameters(self) -> None:
        pass


@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2_5_VL causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        loss_match (`torch.FloatTensor`, *optional*, only when use_evidence=True):
            Match/similarity loss for evidence timesteps.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    loss_match: Optional[torch.FloatTensor] = None


QWEN2_5_VL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        pixel_values (`torch.FloatTensor` of shape `(seq_length, num_channels * image_size * image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2_5_VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
            [`Qwen2_5_VLImageProcessor`] for processing images.
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2_5_VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
            [`Qwen2_5_VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
"""


class video_SALMONN2_plus(Qwen2_5_VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.audio = WhisperEncoder._from_config(
            config.audio_config, attn_implementation=config._attn_implementation
        )
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # frm_head and evi_head (used in evidence mode)
        # Note: use_evidence may be set dynamically after model creation, so both heads are always created
        hidden_size = config.hidden_size
        self.frm_head = nn.Sequential(
            nn.LayerNorm(hidden_size), 
            nn.Linear(hidden_size, hidden_size // 2), 
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        self.evi_head = nn.Sequential(
            nn.LayerNorm(hidden_size), 
            nn.Linear(hidden_size, hidden_size // 2), 
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_size), 
            nn.Linear(hidden_size, hidden_size // 2), 
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        audio_lengths: Optional[list] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            audio_lengths is not None and video_grid_thw is not None
        ):
            # Handle interleaved audio-video input
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            video_index = 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_videos = video_nums
                for _ in range(video_nums):
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = expanded_range * second_per_grid_t * 2

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    video_pos = torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    # llm_pos_ids_list.append(video_pos)

                    audio_len = audio_lengths[video_index]

                    time_index_audio = torch.arange(audio_len, device=input_ids.device) # audio sampling rate is 0.5, so a step of 1 matches the temporal spacing
                    w_index_audio = time_index_audio
                    h_index_audio = torch.zeros_like(time_index_audio)
                    audio_pos = torch.stack([time_index_audio, h_index_audio, w_index_audio]) + st_idx + text_len
                    # llm_pos_ids_list.append(audio_pos)
                    audio_visual_pos = torch.zeros_like(torch.cat((video_pos, audio_pos), dim=1))
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w + audio_len
                    audio_visual_pos[:, input_ids[ed:st] == audio_token_id] = audio_pos # tokens here are already interleaved, so positions are written into the selected tokens in order
                    audio_visual_pos[:, input_ids[ed:st] == video_token_id] = video_pos
                    llm_pos_ids_list.append(audio_visual_pos)
                    video_index += 1
                    remain_videos -= 1

                if st < len(input_tokens):
                    if len(llm_pos_ids_list) > 0:
                        # Take the max of the time dim (dim 0) of the last audio block
                        last_time_max = llm_pos_ids_list[-1][0].max()
                        st_idx = last_time_max + 1
                    else:
                        st_idx = 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        elif input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = expanded_range * second_per_grid_t * 2

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        elif input_ids is not None and audio_lengths is not None:
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            audio_index = 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                audio_nums = 0
                # Locate audio start positions (audio shares the vision start token)
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                audio_nums = (vision_tokens == audio_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_audios = audio_nums
                for _ in range(audio_nums):
                    # Find the end position of the current audio block
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio = input_tokens.index(audio_token_id, st)
                    else:
                        ed_audio = len(input_tokens) + 1
                    ed = ed_audio
                    
                    # audio_lengths is already a token count, used directly as the temporal token count
                    llm_grid_t = audio_lengths[audio_index]  # temporal token count
                    llm_grid_h = 1  # audio has no height dim, fixed to 1
                    llm_grid_w = 1  # audio has no width dim, fixed to 1
                    
                    text_len = ed - st
                    st_idx = (llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0)
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    time_index = torch.arange(llm_grid_t, device=input_ids.device)
                    h_index = time_index  
                    w_index = torch.zeros_like(time_index)  
                    audio_pos = torch.stack([time_index, h_index, w_index]) + st_idx + text_len
                    llm_pos_ids_list.append(audio_pos)
                    
                    st = ed + llm_grid_t 
                    audio_index += 1
                    remain_audios -= 1
                
                # Handle the trailing text segment
                if st < len(input_tokens):
                    if len(llm_pos_ids_list) > 0:
                        # Take the max of the time dim (dim 0) of the last audio block
                        last_time_max = llm_pos_ids_list[-1][0].max()
                        st_idx = last_time_max + 1
                    else:
                        st_idx = 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )
                
                # Combine all position IDs
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas


    def get_image_embeds(self, pixel_values, image_grid_thw, input_ids, inputs_embeds, original_image_embeds=None):
        if original_image_embeds is None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            return inputs_embeds, image_embeds
        else:
            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)
            image_embeds = original_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            return inputs_embeds


    def get_video_embeds(self, pixel_values_videos, video_grid_thw, input_ids, inputs_embeds, original_video_embeds=None):
        if original_video_embeds is None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            return inputs_embeds, video_embeds
        else:
            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)
            video_embeds = original_video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            return inputs_embeds

    def get_audio_embeds(self, audio_feature, input_ids, inputs_embeds, original_audio_embeds=None):
        if original_audio_embeds is None:
            audio_feature = audio_feature.type(self.audio.dtype)
            audio_embeds = self.audio(audio_feature)
            n_audio_tokens = (input_ids == self.config.audio_token_id).sum().item()
            audio_embeds = audio_embeds.flatten(0, 1)
            n_audio_features = audio_embeds.shape[0]
            if n_audio_tokens != n_audio_features:
                raise ValueError(
                    f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
                )

            mask = input_ids == self.config.audio_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            audio_mask = mask_expanded.to(inputs_embeds.device)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
            return inputs_embeds, audio_embeds
        else:
            mask = input_ids == self.config.audio_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            audio_mask = mask_expanded.to(inputs_embeds.device)
            audio_embeds = original_audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
            return inputs_embeds

    def _register_attn_row_capture(self, layer_idx):
        """Register a pre-hook on layer layer_idx's self_attn to capture its input hidden states
        and position embeddings. Used to recompute the "last text token vs. full sequence"
        attention row in O(L*d), instead of materializing the whole layer's [B, H, L, L]
        via output_attentions (O(L^2) memory). Caller must handle.remove() afterwards."""
        storage = {}
        attn_module = self.model.layers[layer_idx].self_attn

        def pre_hook(module, args, kwargs):
            storage['hidden'] = args[0] if len(args) > 0 else kwargs.get('hidden_states')
            storage['position_embeddings'] = kwargs.get('position_embeddings')
            storage['attention_mask'] = args[1] if len(args) > 1 else kwargs.get('attention_mask')
            return None

        handle = attn_module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        return handle, storage

    def _attn_row_from_capture(self, storage, last_text_token_indices, layer_idx):
        """Recompute the target attention row from the captured layer input, step-for-step
        matching the eager implementation: q/k projection -> M-RoPE -> GQA repeat ->
        scores/sqrt(d) + causal mask -> fp32 softmax -> average over heads.
        Returns [B, seq_len, 1], numerically matching the sliced result of the old
        output_attentions path (within bf16 precision). Gradients flow through the q/k
        projections and hidden as usual, so training semantics are equivalent."""
        attn_module = self.model.layers[layer_idx].self_attn
        hidden = storage['hidden']                                   # [B, L, hidden_size]
        cos, sin = storage['position_embeddings']
        bsz, seq_len, _ = hidden.shape
        num_heads = attn_module.num_heads
        num_kv_heads = attn_module.num_key_value_heads
        head_dim = attn_module.head_dim

        q = attn_module.q_proj(hidden).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k = attn_module.k_proj(hidden).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, attn_module.rope_scaling["mrope_section"])
        k = repeat_kv(k, num_heads // num_kv_heads)                  # [B, num_heads, L, head_dim]

        idx = last_text_token_indices.to(device=hidden.device).view(-1)  # [B]
        batch_arange = torch.arange(bsz, device=hidden.device)
        q_row = q[batch_arange, :, idx]                               # [B, num_heads, head_dim]
        scores = torch.matmul(q_row.unsqueeze(2), k.transpose(2, 3)).squeeze(2) / (head_dim ** 0.5)  # [B, num_heads, L]

        causal_mask = storage.get('attention_mask')
        if causal_mask is not None:                                   # 4D additive mask [B, 1, L, kv_len]
            scores = scores + causal_mask[batch_arange, 0, idx][:, None, :]
        else:                                                         # mask is None on the sdpa is_causal fast path; apply this row's causal constraint manually
            invisible = torch.arange(seq_len, device=hidden.device).unsqueeze(0) > idx.unsqueeze(1)  # [B, L]
            scores = scores.masked_fill(invisible.unsqueeze(1), torch.finfo(scores.dtype).min)

        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden.dtype)  # fp32 softmax, matching eager
        return probs.mean(dim=1).unsqueeze(-1)                        # [B, seq_len, 1]

    def find_and_merge(self, matrix):
        """Grounding readout for one event token: similarity scores -> ONE interval.

        Expects 1D [num_timesteps] (a leading batch dim of 1 is squeezed). Steps:
          1. threshold: timesteps scoring >= grounding_threshold * max are salient
             (paper setting: 0.5 for LongVALE, 0.7 for ChronusAV);
          2. group salient timesteps into contiguous runs;
          3. bridge runs whose gap <= grounding_merge_gap timesteps -- close runs are one
             event interrupted by noise, far-apart runs are NOT collapsed into one span
             (the old first-to-last-salient span inflated intervals ~2.7x vs GT);
          4. pick ONE winner among the merged candidates by grounding_select:
               "mass"    (default) sum of scores over the segment -- rewards segments that
                         are both strong and long; a single-step spike cannot outrank a
                         sustained event, and a long weak tail cannot outrank a solid core
               "longest" longest duration
               "mean"    highest mean score (biases toward short spikes; kept for ablation)
        Returned as [[start, end]] so downstream per-event interval lists keep their shape.
        """
        if matrix.dim() == 2:
            matrix = matrix.squeeze(0)
        similarity_Threshold = getattr(self.config, 'grounding_threshold', 0.8)
        threshold = torch.max(matrix) * similarity_Threshold
        salient = (matrix >= threshold).nonzero(as_tuple=True)[0].tolist()
        if not salient:
            return []

        # contiguous runs of salient timesteps
        runs = [[salient[0], salient[0]]]
        for t in salient[1:]:
            if t == runs[-1][1] + 1:
                runs[-1][1] = t
            else:
                runs.append([t, t])

        # bridge small gaps (in timesteps; at 256-frame sampling one step is ~2s for a
        # typical few-minute video, so the default 2 bridges gaps of roughly <= 4-6s)
        merge_gap = getattr(self.config, 'grounding_merge_gap', 2)
        merged = [runs[0]]
        for st, en in runs[1:]:
            if st - merged[-1][1] - 1 <= merge_gap:
                merged[-1][1] = en
            else:
                merged.append([st, en])

        if len(merged) == 1:
            return [[merged[0][0], merged[0][1]]]
        select = getattr(self.config, 'grounding_select', 'mass')
        if select == 'longest':
            score = lambda seg: seg[1] - seg[0]
        elif select == 'mean':
            score = lambda seg: matrix[seg[0]:seg[1] + 1].mean().item()
        else:  # 'mass'
            score = lambda seg: matrix[seg[0]:seg[1] + 1].sum().item()
        best = max(merged, key=score)
        return [[best[0], best[1]]]

    def _prepare_first_forward_pass(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        labels: Optional[torch.LongTensor],
        position_ids: Optional[torch.LongTensor],
        last_text_token_indices: Optional[torch.LongTensor] = None,
    ):
        """
        Prepare the first forward pass: truncate the sequence at the vision_end position.

        Returns:
            input_ids_part1: truncated input_ids
            inputs_embeds_part1: truncated inputs_embeds
            labels_part1: truncated labels (if labels is not None)
            attention_mask_part1: truncated attention_mask
            position_ids_part1: truncated position_ids (if position_ids is not None)
            vision_start_positions: position indices of vision_start
            vision_end_positions: position indices of vision_end
        """
        vision_end_positions = None
        vision_start_positions = None
        input_ids_part1 = None
        inputs_embeds_part1 = None
        labels_part1 = None
        attention_mask_part1 = None
        position_ids_part1 = None
        
        if input_ids is not None:
            # Find each sample's <|vision_end|> position (one per sample)
            vision_end_mask = input_ids == self.config.vision_end_token_id  # [batch_size, seq_len]
            vision_end_positions = torch.argmax(vision_end_mask.long(), dim=1)  # [batch_size]
            
            # Find each sample's <|vision_start|> position (one per sample)
            vision_start_mask = input_ids == self.config.vision_start_token_id  # [batch_size, seq_len]
            vision_start_positions = torch.argmax(vision_start_mask.long(), dim=1)  # [batch_size]
            
            # Truncate each sample at vision_end_positions, keeping only tokens up to and including <|vision_end|>
            batch_size = input_ids.shape[0]
            if last_text_token_indices is not None:
                # Find each sample's text end position
                text_end_positions = last_text_token_indices.to(device=input_ids.device).squeeze(-1)
                max_length = (text_end_positions).max().item() + 1  # max length within the batch (including vision_end)
            else:
                max_length = (vision_end_positions).max().item() + 1  # max length (including vision_end)

            # Truncate sequences
            truncated_input_ids = []
            truncated_inputs_embeds = []
            truncated_labels = []  
            truncated_position_ids = []
            for b in range(batch_size):
                end_pos = vision_end_positions[b].item() + 1 if last_text_token_indices is None else text_end_positions[b].item() + 1  
                truncated_input_ids.append(input_ids[b, :end_pos])
                truncated_inputs_embeds.append(inputs_embeds[b, :end_pos])
                if labels is not None:
                    truncated_labels.append(labels[b, :end_pos])
                truncated_position_ids.append(position_ids[:, b, :end_pos])  # [3, end_pos]
            
            # Update input_ids
            pad_token_id = 151643
            input_ids_part1 = torch.nn.utils.rnn.pad_sequence(truncated_input_ids, batch_first=True, padding_value=pad_token_id)
            
            # Update embeddings
            embed_tokens = self.model.get_input_embeddings()
            pad_token_embedding = embed_tokens(torch.tensor([pad_token_id], device=inputs_embeds.device))  # [1, hidden_size]
            pad_token_embedding = pad_token_embedding.squeeze(0)  # [hidden_size]
            inputs_embeds_part1 = pad_token_embedding.unsqueeze(0).unsqueeze(0).expand(
                batch_size, max_length, inputs_embeds.shape[-1]).clone()
            for b in range(batch_size):
                seq_len = truncated_inputs_embeds[b].shape[0]
                inputs_embeds_part1[b, :seq_len] = truncated_inputs_embeds[b]
            
            # Update labels
            if labels is not None:
                ignore_index = -100  
                labels_part1 = torch.full((batch_size, max_length),ignore_index,device=labels.device, dtype=labels.dtype)
                for b in range(batch_size):
                    seq_len = truncated_labels[b].shape[0]
                    labels_part1[b, :seq_len] = truncated_labels[b]

            # Update attention_mask
            attention_mask_part1 = (input_ids_part1 != pad_token_id).long()  # [batch_size, max_length]

            # Update position_ids
            if position_ids is not None and position_ids.shape[0] == 3:
                padded_position_ids_list = []
                for b in range(batch_size):
                    seq_len = truncated_position_ids[b].shape[1]
                    pad_length = max_length - seq_len
                    if pad_length > 0:
                        padded_pos_ids = torch.nn.functional.pad(truncated_position_ids[b], (0, pad_length), "constant", 1)  # [3, max_length]
                    else:
                        padded_pos_ids = truncated_position_ids[b]  # [3, max_length]
                    padded_position_ids_list.append(padded_pos_ids)
                position_ids_part1 = torch.stack(padded_position_ids_list, dim=1)  # Stack: [batch_size, 3, max_length] -> [3, batch_size, max_length]
        
        return (
            input_ids_part1,
            inputs_embeds_part1,
            labels_part1,
            attention_mask_part1,
            position_ids_part1,
            vision_start_positions,
            vision_end_positions,
        )

    def _prepare_forward_pass_for_infer(
        self,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.FloatTensor],
        mode: str,
        video_token_indices_list_src: Optional[list],
        audio_token_indices_list_src: Optional[list],
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
    ):
        batch_size = input_ids.size(0)
        event_id = [None] * batch_size 
        if mode == "caching":
            # step1: find vision_end and vision_start positions
            vision_end_mask = input_ids == self.config.vision_end_token_id  # [batch_size, seq_len]
            vision_end_positions = torch.argmax(vision_end_mask.long(), dim=1)  # [batch_size]
            vision_start_mask = input_ids == self.config.vision_start_token_id  # [batch_size, seq_len]
            vision_start_positions = torch.argmax(vision_start_mask.long(), dim=1)  # [batch_size]
            evi_token_id = self.config.evi_token_id        # evi_token_id = getattr(self.config, "evi_token_id", None)
            video_token_id = self.config.video_token_id
            audio_token_id = self.config.audio_token_id


            # step2: extract initial states of vision+audio tokens and build video_mask/audio_mask
            vision_audio_hidden_states_ori = None
            video_masks_list = None
            audio_masks_list = None
            if vision_start_positions is not None and vision_end_positions is not None:
                vision_audio_hidden_states_list_ori = []
                video_masks_list_temp = []
                audio_masks_list_temp = []
                for b in range(batch_size):
                    start_pos = vision_start_positions[b].item()
                    end_pos = vision_end_positions[b].item() + 1
                    vision_audio_hidden_ori = inputs_embeds[b, start_pos:end_pos]  # [vision_token_len, hidden_size]
                    vision_audio_hidden_states_list_ori.append(vision_audio_hidden_ori)
                    
                    # Compute video_mask and audio_mask
                    video_mask = input_ids[b, start_pos:end_pos] == video_token_id  # [vision_token_len]
                    audio_mask = input_ids[b, start_pos:end_pos] == audio_token_id  # [vision_token_len]
                    video_masks_list_temp.append(video_mask)
                    audio_masks_list_temp.append(audio_mask)
                vision_audio_hidden_states_ori = vision_audio_hidden_states_list_ori
                video_masks_list = video_masks_list_temp
                audio_masks_list = audio_masks_list_temp  

            # Helper: mean token features per time range (simple, unweighted average); same logic as compute_token_means in _prepare_second_forward_pass but without scores
            def compute_token_means_no_score(pure_hidden_states, token_indices_list, b):
                if token_indices_list is not None and b < len(token_indices_list) and token_indices_list[b] is not None:
                    means = []
                    for time_range_indices in token_indices_list[b]:
                        if time_range_indices is None or len(time_range_indices) == 0:
                            means.append(None)
                            continue
                        # Handle a bare single range: with one src/evi this may be (0, 1495); iterating "for start_idx, end_idx in (0,1495)" would fail unpacking 0
                        if isinstance(time_range_indices, (tuple, list)) and len(time_range_indices) == 2 and not isinstance(time_range_indices[0], (tuple, list)):
                            ranges = [tuple(time_range_indices)]
                        else:
                            ranges = time_range_indices
                        hidden_states_in_range = []
                        for start_idx, end_idx in ranges:
                            if start_idx < pure_hidden_states.shape[0]:
                                end_idx = min(end_idx, pure_hidden_states.shape[0])
                                hidden_states_in_range.append(pure_hidden_states[start_idx:end_idx])
                        if len(hidden_states_in_range) > 0:
                            hidden_states_tensor = torch.cat(hidden_states_in_range, dim=0)
                            mean = torch.mean(hidden_states_tensor, dim=0)
                            means.append(mean)
                        else:
                            means.append(None)
                else:
                    return None
                return means

            # Helper: update evi token embeddings (same as update_evi_embeddings in _prepare_second_forward_pass)
            def update_evi_embeddings(inputs_embeds, evi_indices, video_means, audio_means, b):
                """Update evi token embeddings by adding the video/audio means at the corresponding positions"""
                if inputs_embeds is not None and evi_indices is not None and len(evi_indices) > 0:
                    for i, evi_idx in enumerate(evi_indices):
                        idx = int(evi_idx.item() if torch.is_tensor(evi_idx) else evi_idx)
                        if idx < inputs_embeds.shape[1]:
                            if video_means is not None and i < len(video_means) and video_means[i] is not None:
                                inputs_embeds[b, idx] = inputs_embeds[b, idx] + video_means[i]
                            if audio_means is not None and i < len(audio_means) and audio_means[i] is not None:
                                inputs_embeds[b, idx] = inputs_embeds[b, idx] + audio_means[i] * getattr(self.config, 'audio_mean_weight', 1.0)

            # step3: add the original vision token features onto the src evi tokens
            for b in range(batch_size):
                # 3.1) find src evi token positions and check the count
                evi_mask = input_ids[b] == evi_token_id  # [seq_len]
                evi_positions = torch.where(evi_mask)[0]  # position indices of all evi tokens
                if len(evi_positions) > 0 and video_token_indices_list_src is not None and b < len(video_token_indices_list_src):
                    assert len(evi_positions) == len(video_token_indices_list_src[b]), f"Sample {b}: valid_evi_indices_src length)"

                    # 3.2) get this sample's vision_audio_hidden_states_ori
                    batch_vision_audio_hidden_ori = vision_audio_hidden_states_ori[b]

                    # 3.2) select initial states of video and audio tokens using the precomputed masks
                    video_mask = video_masks_list[b]
                    audio_mask = audio_masks_list[b]
                    pure_video_hidden_states_ori = batch_vision_audio_hidden_ori[video_mask]  # [num_video_tokens, hidden_size]
                    pure_audio_hidden_states_ori = batch_vision_audio_hidden_ori[audio_mask]  # [num_audio_tokens, hidden_size]

                    # 3.3) compute src video/audio token means (simple average; same as _prepare_second_forward_pass but unweighted)
                    sample_video_means_src = compute_token_means_no_score(pure_video_hidden_states_ori, video_token_indices_list_src, b) if video_token_indices_list_src and len(video_token_indices_list_src) > 0 and b < len(video_token_indices_list_src) else None
                    sample_audio_means_src = compute_token_means_no_score(pure_audio_hidden_states_ori, audio_token_indices_list_src, b) if audio_token_indices_list_src and len(audio_token_indices_list_src) > 0 and b < len(audio_token_indices_list_src) else None

                    # 3.4) update src evi token embeddings
                    update_evi_embeddings(inputs_embeds, evi_positions, sample_video_means_src, sample_audio_means_src, b)
        
        elif mode == "generating":
            assert input_ids.size(1) == 1
            vision_start_positions, vision_end_positions = None, None  # generating phase uses only the cache; no valid range is returned

            _evi_like_ids = {self.config.evi_token_id}
            for b in range(batch_size):
                if input_ids[b].item() in _evi_like_ids:
                    # cache_timestep_token_ranges is built at caching time from video/audio_token_indices_per_timestep; each step is a (v_idx, a_idx) index tensor pair
                    cached = self.cache_timestep_token_ranges[b]
                    cache_hidden = self.cache_img_state[b]  # [vision_token_len, hidden_size]
                    cache_hidden_score = self.cache_img_state_score[b] if self.cache_img_state_score[b] is not None else None  # [vision_token_len, 1] or None
                    hidden_size = cache_hidden.shape[1]
                    device, dtype = cache_hidden.device, cache_hidden.dtype
                    video_list, audio_list = [], []
                    video_scores_list, audio_scores_list = [], []

                    # Read the predicted time ranges for this <G> token (list of ranges, possibly multiple)
                    if getattr(self, 'serial_end_positions', None) is not None and self.serial_end_positions[b] is not None:
                        # Parallel anchor evi: keep semantics consistent with planning; reuse the grounding range of the corresponding serial G token
                        anchor_j = cache_position.item() - self.serial_end_positions[b] - 1
                        if 0 <= anchor_j < len(self.serial_evi_tgts[b]):
                            new_tgt = self.serial_evi_tgts[b][anchor_j]
                        else:
                            new_tgt = self.tgt[b][-1] if self.tgt[b] else [[0, 0]]
                    else:
                        # Serial planning evi: use the range predicted at the previous step, and record it for reuse by parallel anchors
                        new_tgt = self.tgt[b][-1] if self.tgt[b] else [[0, 0]]
                        self.serial_evi_tgts[b].append(new_tgt)

                    # Collect video/audio hidden states and scores within the ranges (supports multiple segments)
                    ts_list = [ts for seg_start, seg_end in new_tgt for ts in range(seg_start, seg_end + 1)]
                    for ts in ts_list:
                        if ts < 0 or ts >= len(cached):
                            continue
                        v_idx, a_idx = cached[ts]
                        if v_idx is not None and v_idx.numel() > 0:
                            video_list.append(cache_hidden[v_idx])
                            if cache_hidden_score is not None:
                                video_scores_list.append(cache_hidden_score[v_idx])
                        if a_idx is not None and a_idx.numel() > 0:
                            audio_list.append(cache_hidden[a_idx])
                            if cache_hidden_score is not None:
                                audio_scores_list.append(cache_hidden_score[a_idx])
                    
                    # Compute the video mean (weighted or simple average)
                    if video_list:
                        video_tensor = torch.cat(video_list, dim=0)  # [num_video_tokens, hidden_size]
                        if cache_hidden_score is not None and len(video_scores_list) > 0:
                            video_scores_tensor = torch.cat(video_scores_list, dim=0)  # [num_video_tokens, 1]
                            # Softmax within the current timestep range, then weight (each range normalized independently)
                            video_scores_tensor = F.softmax(video_scores_tensor.squeeze(-1), dim=0).unsqueeze(-1)  # [num_video_tokens, 1]
                            video_mean = torch.sum(video_tensor * video_scores_tensor, dim=0).to(device=device, dtype=dtype)  # [hidden_size]
                        else:
                            video_mean = torch.mean(video_tensor, dim=0).to(device=device, dtype=dtype)
                    else:
                        video_mean = torch.zeros(hidden_size, device=device, dtype=dtype)
                    
                    # Compute the audio mean (weighted or simple average)
                    if audio_list:
                        audio_tensor = torch.cat(audio_list, dim=0)  # [num_audio_tokens, hidden_size]
                        if cache_hidden_score is not None and len(audio_scores_list) > 0:
                            audio_scores_tensor = torch.cat(audio_scores_list, dim=0)  # [num_audio_tokens, 1]
                            # Softmax within the current timestep range, then weight (each range normalized independently)
                            audio_scores_tensor = F.softmax(audio_scores_tensor.squeeze(-1), dim=0).unsqueeze(-1)  # [num_audio_tokens, 1]
                            audio_mean = torch.sum(audio_tensor * audio_scores_tensor, dim=0).to(device=device, dtype=dtype)  # [hidden_size]
                        else:
                            audio_mean = torch.mean(audio_tensor, dim=0).to(device=device, dtype=dtype)
                    else:
                        audio_mean = torch.zeros(hidden_size, device=device, dtype=dtype)
                    # Same as _compute_vision_info_per_timestep: video + audio_mean_weight * audio, added back at the current evi position
                    aw = getattr(self.config, 'audio_mean_weight', 1.0)
                    inputs_embeds[b, 0] = inputs_embeds[b, 0] + video_mean + aw * audio_mean

                    # Count only serial-planning-phase evi tokens (before <S> appears); parallel anchors are not counted
                    self.evi_count[b] += 1 if self.serial_end_positions[b] is None else 0

                elif input_ids[b].item() == self.config.evi_end_token_id:
                    # <S> means serial prediction has ended and parallel prediction is about to start
                    # step1: record the serial endpoint position in the full sequence
                    self.serial_end_positions[b] = cache_position.item()

                    # step2: decide whether to start parallel prediction
                    self.parallel[b] = True if self.evi_count[b] > 1 else False

                # Once parallel decoding has started, record the event id of every token
                if self.parallel[b]:
                    if self.event_id[b] is None:
                        # step1: initialize event ids as a tensor of length cache_position.item()+1 filled with -1
                        self.event_id[b] = torch.full((cache_position.item() + 1,), -1, dtype=torch.long, device=input_ids.device)

                        # Fill in the current token's event id
                        self.event_id[b][-1] = (cache_position.item() - self.serial_end_positions[b] - 1) % self.evi_count[b]
                    else:
                        # Extend event ids by one element initialized to 0
                        self.event_id[b] = torch.cat([self.event_id[b], torch.tensor([0], dtype=torch.long, device=input_ids.device)])

                        self.event_id[b][-1] = (cache_position.item() - self.serial_end_positions[b] - 1) % self.evi_count[b]
                    event_id[b] = self.event_id[b]

        return vision_start_positions, vision_end_positions, event_id
    

    def _prepare_second_forward_pass(
        self,
        vision_start_positions: Optional[torch.LongTensor],
        vision_end_positions: Optional[torch.LongTensor],
        hidden_states_part1: Optional[tuple],
        input_ids_part1: Optional[torch.LongTensor],
        input_ids: Optional[torch.LongTensor],
        labels: Optional[torch.LongTensor],
        tgt: Optional[list],
        video_token_indices_list: Optional[list],
        audio_token_indices_list: Optional[list],
        video_token_indices_list_src: Optional[list],
        audio_token_indices_list_src: Optional[list],
        inputs_embeds: Optional[torch.FloatTensor],
        last_text_to_vision_attn_maps: Optional[torch.Tensor] = None,
    ):
        """
        Prepare the second forward pass: extract vision+audio hidden states, compute token means,
        and update G token embeddings.

        Args:
            last_text_to_vision_attn_maps: optional; per-sample attention map of the last text token over the full sequence, shape [B, seq_len, 1]

        Returns:
            vision_audio_hidden_states: list of vision+audio token hidden states
            video_masks_list: list of video token masks
            audio_masks_list: list of audio token masks
            G_token_indices_list: list of G token indices (labels != -100, used for tgt)
            G_token_indices_list_src: list of G token indices (labels == -100, used for src)
            video_token_means_list: list of video token means
            audio_token_means_list: list of audio token means
            inputs_embeds: updated inputs_embeds
        """


        vision_audio_hidden_states = None
        vision_audio_hidden_states_ori = None
        vision_audio_hidden_states_score = None
        video_masks_list = None
        audio_masks_list = None
        evi_token_indices_list = None
        evi_token_indices_list_src = None
        video_token_means_list = None
        audio_token_means_list = None
        
        # step1: extract hidden and initial states of vision+audio tokens; use_score_head decides whether score weighting is applied
        use_score_head = getattr(self.config, 'use_score_head', False)
        use_attention_head = getattr(self.config, 'use_attention_head', False)
        if vision_start_positions is not None and vision_end_positions is not None:
            vision_audio_layer_index = getattr(self.config, 'vision_audio_layer_index', None)
            if vision_audio_layer_index is not None and hidden_states_part1 is not None:
                if vision_audio_layer_index < len(hidden_states_part1) and input_ids_part1 is not None:
                    layer_hidden_states = hidden_states_part1[vision_audio_layer_index]  # [batch_size, seq_len, hidden_size]
                    if use_score_head:
                        layer_hidden_states_score = F.sigmoid(self.score_head(layer_hidden_states))  # [batch_size, seq_len, 1]
                    elif use_attention_head and last_text_to_vision_attn_maps is not None:
                        layer_hidden_states_score = last_text_to_vision_attn_maps # [batch_size, seq_len, 1]
                    else:
                        layer_hidden_states_score = None
                    batch_size = layer_hidden_states.shape[0]
                    video_token_id = self.config.video_token_id
                    audio_token_id = self.config.audio_token_id
                    
                    vision_audio_hidden_states_list = []
                    vision_audio_hidden_states_list_ori = []
                    vision_audio_hidden_states_score_list = [] if use_score_head or use_attention_head else None
                    video_masks_list_temp = []
                    audio_masks_list_temp = []
                    for b in range(batch_size):
                        start_pos = vision_start_positions[b].item()
                        end_pos = vision_end_positions[b].item() + 1
                        vision_audio_hidden = layer_hidden_states[b, start_pos:end_pos]  # [vision_token_len, hidden_size]
                        vision_audio_hidden_states_list.append(vision_audio_hidden)
                        vision_audio_hidden_ori = inputs_embeds[b, start_pos:end_pos]  # [vision_token_len, hidden_size]
                        vision_audio_hidden_states_list_ori.append(vision_audio_hidden_ori)
                        if layer_hidden_states_score is not None:
                            vision_audio_hidden_states_score_b = layer_hidden_states_score[b, start_pos:end_pos]  # [vision_token_len, 1]
                            vision_audio_hidden_states_score_list.append(vision_audio_hidden_states_score_b)
                        
                        # Compute video_mask and audio_mask
                        video_mask = input_ids_part1[b, start_pos:end_pos] == video_token_id  # [vision_token_len]
                        audio_mask = input_ids_part1[b, start_pos:end_pos] == audio_token_id  # [vision_token_len]
                        video_masks_list_temp.append(video_mask)
                        audio_masks_list_temp.append(audio_mask)
                    vision_audio_hidden_states = vision_audio_hidden_states_list
                    vision_audio_hidden_states_ori = vision_audio_hidden_states_list_ori
                    vision_audio_hidden_states_score = vision_audio_hidden_states_score_list
                    video_masks_list = video_masks_list_temp
                    audio_masks_list = audio_masks_list_temp  

        # step2: for each sample, build the evi token index sequence from input_ids (only positions whose evi token loss is not -100 are added)
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            evi_token_id = self.config.evi_token_id
            # torch.where returns positions in order: serial segment first, anchor segment after,
            # matching tgt's [K serial ranges, K identical anchor ranges] layout one-to-one.
            evi_token_indices_list = []
            evi_token_indices_list_src = []

            for b in range(batch_size):
                evi_mask = input_ids[b] == evi_token_id  # [seq_len]
                evi_positions = torch.where(evi_mask)[0]  # position indices of all evi tokens

                # Filter: only positions whose labels are not -100 are added (used for tgt)
                valid_evi_indices = []
                # Filter: only positions whose labels are -100 are added (used for src)
                valid_evi_indices_src = []
                if labels is not None:
                    for pos in evi_positions:
                        pos_item = pos.item()
                        if pos_item < labels.shape[1]:  # ensure index is valid
                            if labels[b, pos_item] != -100:
                                valid_evi_indices.append(pos_item)
                            else:
                                valid_evi_indices_src.append(pos_item)
                else:
                    # Without labels, all evi token positions go into valid_evi_indices
                    valid_evi_indices = [pos.item() for pos in evi_positions]

                # Check: len(valid_evi_indices) * 2 should equal this sample's tgt length
                if tgt is not None and b < len(tgt):
                    expected_length = len(valid_evi_indices) * 2
                    actual_length = len(tgt[b]) if isinstance(tgt[b], (list, tuple)) else 0
                    assert expected_length == actual_length, f"Sample {b}: valid_evi_indices length * 2 ({expected_length}) != tgt length ({actual_length})"
                    evi_token_indices_list.append(valid_evi_indices)

                # Check: len(valid_evi_indices_src) should equal len(video_token_indices_list_src[b])
                # Distributed training: assert only when video_token_indices_list_src is non-empty and indexable, to avoid a crash / comm mismatch when some rank has None
                if len(valid_evi_indices_src) > 0 and video_token_indices_list_src is not None and b < len(video_token_indices_list_src):
                    assert len(valid_evi_indices_src) == len(video_token_indices_list_src[b]), f"Sample {b}: valid_evi_indices_src length)"
                    evi_token_indices_list_src.append(valid_evi_indices_src)
                    
        # step3: compute per-sample, per-time-range video and audio token means
        if vision_audio_hidden_states is not None and video_masks_list is not None and audio_masks_list is not None and (video_token_indices_list is not None or audio_token_indices_list is not None):
            batch_size = len(vision_audio_hidden_states)


            def _collect_hidden_and_scores_in_range(pure_hidden_states, time_range_indices, pure_scores=None):
                """Collect hidden_states and scores within one time range. Returns (hidden_tensor, scores_tensor or None); empty ranges return (None, None)."""
                if time_range_indices is None or (isinstance(time_range_indices, (tuple, list)) and len(time_range_indices) == 0):
                    return (None, None)
                # Handle a bare single range: with one src/evi this may be (0, 1495); iterating "for start_idx, end_idx in (0,1495)" would fail unpacking 0
                if isinstance(time_range_indices, (tuple, list)) and len(time_range_indices) == 2 and not isinstance(time_range_indices[0], (tuple, list)):
                    ranges = [tuple(time_range_indices)]
                else:
                    ranges = time_range_indices
                hidden_states_in_range = []
                scores_in_range = []
                for start_idx, end_idx in ranges:
                    if start_idx < pure_hidden_states.shape[0]:
                        end_idx = min(end_idx, pure_hidden_states.shape[0])
                        hidden_states_in_range.append(pure_hidden_states[start_idx:end_idx])
                        if pure_scores is not None:
                            scores_in_range.append(pure_scores[start_idx:end_idx])
                if len(hidden_states_in_range) == 0:
                    return (None, None)
                hidden_tensor = torch.cat(hidden_states_in_range, dim=0)
                scores_tensor = torch.cat(scores_in_range, dim=0) if pure_scores is not None and len(scores_in_range) > 0 else None
                return (hidden_tensor, scores_tensor)

            def compute_multimodal_tokens_mean(pure_video_hidden_states, pure_audio_hidden_states, video_token_indices_list, audio_token_indices_list, b, pure_video_hidden_states_score=None, pure_audio_hidden_states_score=None):
                """Concatenate the video and audio features of one evi, then softmax-weight/average them jointly to get a single multimodal mean per evi. Returns a list (one mean per evi)."""
                video_list = (video_token_indices_list[b] if video_token_indices_list is not None and b < len(video_token_indices_list) and video_token_indices_list[b] is not None else [])
                audio_list = (audio_token_indices_list[b] if audio_token_indices_list is not None and b < len(audio_token_indices_list) and audio_token_indices_list[b] is not None else [])
                assert len(video_list)==len(audio_list)
                num_segments = max(len(video_list), len(audio_list))
                if num_segments == 0:
                    return None
                multimodal_means = []
                for i in range(num_segments):
                    time_range_v = video_list[i]
                    time_range_a = audio_list[i]
                    hidden_v, scores_v = _collect_hidden_and_scores_in_range(pure_video_hidden_states, time_range_v, pure_video_hidden_states_score)
                    hidden_a, scores_a = _collect_hidden_and_scores_in_range(pure_audio_hidden_states, time_range_a, pure_audio_hidden_states_score)
                    parts_h = [p for p in [hidden_v, hidden_a] if p is not None]
                    if len(parts_h) == 0:
                        multimodal_means.append(None)
                        continue
                    combined_hidden = torch.cat(parts_h, dim=0)  # [num_tokens, hidden_size]
                    has_scores = (scores_v is not None or scores_a is not None)
                    if has_scores:
                        parts_s = [p for p in [scores_v, scores_a] if p is not None]
                        combined_scores = torch.cat(parts_s, dim=0)  # [num_tokens, 1]
                        combined_scores = F.softmax(combined_scores.squeeze(-1), dim=0).unsqueeze(-1)
                        mean = torch.sum(combined_hidden * combined_scores, dim=0)
                    else:
                        mean = torch.mean(combined_hidden, dim=0)
                    multimodal_means.append(mean)
                return multimodal_means

            # Helper: compute token means (supports weighted averaging)
            def compute_token_means(pure_hidden_states, token_indices_list, b, pure_scores=None):
                """Compute the list of token means for one sample; uses weighted averaging if scores are provided"""
                if token_indices_list is not None and b < len(token_indices_list) and token_indices_list[b] is not None:
                    means = []
                    for time_range_indices in token_indices_list[b]:
                        if time_range_indices is None or len(time_range_indices) == 0:
                            means.append(None)
                            continue
                        # Handle a bare single range: with one src/evi this may be (0, 1495); iterating "for start_idx, end_idx in (0,1495)" would fail unpacking 0
                        if isinstance(time_range_indices, (tuple, list)) and len(time_range_indices) == 2 and not isinstance(time_range_indices[0], (tuple, list)):
                            ranges = [tuple(time_range_indices)]
                        else:
                            ranges = time_range_indices
                        # Collect hidden states and scores of all tokens within this time range
                        hidden_states_in_range = []
                        scores_in_range = []
                        for start_idx, end_idx in ranges:
                            if start_idx < pure_hidden_states.shape[0]:
                                end_idx = min(end_idx, pure_hidden_states.shape[0])
                                hidden_states_in_range.append(pure_hidden_states[start_idx:end_idx])
                                if pure_scores is not None:
                                    scores_in_range.append(pure_scores[start_idx:end_idx])
                        
                        # Compute the mean for this time range (weighted or simple average)
                        if len(hidden_states_in_range) > 0:
                            hidden_states_tensor = torch.cat(hidden_states_in_range, dim=0)  # [num_tokens, hidden_size]
                            if pure_scores is not None and len(scores_in_range) > 0:
                                scores_tensor = torch.cat(scores_in_range, dim=0)  # [num_tokens, 1]
                                # Softmax within the current tgt range, then weight (each range normalized independently)
                                scores_tensor = F.softmax(scores_tensor.squeeze(-1), dim=0).unsqueeze(-1)  # [num_tokens, 1]
                                weighted_sum = torch.sum(hidden_states_tensor * scores_tensor, dim=0)  # [hidden_size]
                                mean = weighted_sum
                            else:
                                # No scores: use a simple average
                                mean = torch.mean(hidden_states_tensor, dim=0)
                            means.append(mean)
                        else:
                            # No tokens found: append None
                            means.append(None)
                else:
                    return None
                return means
            
            # Helper: update evi token embeddings
            def update_evi_embeddings(inputs_embeds, evi_indices, video_means, audio_means, b):
                """Update evi token embeddings"""
                if inputs_embeds is not None and evi_indices is not None and len(evi_indices) > 0:
                    for i, evi_idx in enumerate(evi_indices):
                        if evi_idx < inputs_embeds.shape[1]:
                            if video_means is not None and i < len(video_means) and video_means[i] is not None:
                                inputs_embeds[b, evi_idx] = inputs_embeds[b, evi_idx] + video_means[i]
                            if audio_means is not None and i < len(audio_means) and audio_means[i] is not None:
                                inputs_embeds[b, evi_idx] = inputs_embeds[b, evi_idx] + audio_means[i] * getattr(self.config, 'audio_mean_weight', 1.0)

            def update_union_evi_embeddings(inputs_embeds, evi_indices, video_audio_means, b):
                """Update evi token embeddings"""
                if inputs_embeds is not None and evi_indices is not None and len(evi_indices) > 0:
                    for i, evi_idx in enumerate(evi_indices):
                        if evi_idx < inputs_embeds.shape[1]:
                            if video_audio_means is not None and i < len(video_audio_means) and video_audio_means[i] is not None:
                                inputs_embeds[b, evi_idx] = inputs_embeds[b, evi_idx] + video_audio_means[i]
            
            video_token_means_list = []
            audio_token_means_list = []
            video_token_means_list_src = []
            audio_token_means_list_src = []
            
            for b in range(batch_size):
                # 3.1) get this sample's vision_audio_hidden_states, vision_audio_hidden_states_ori, and optional score
                batch_vision_audio_hidden = vision_audio_hidden_states[b]  # [vision_token_len, hidden_size]
                batch_vision_audio_hidden_ori = vision_audio_hidden_states_ori[b]  # [vision_token_len, hidden_size]
                batch_vision_audio_hidden_score = vision_audio_hidden_states_score[b] if vision_audio_hidden_states_score is not None else None  # [vision_token_len, 1] or None
                
                # 3.2) select hidden and initial states (and optional scores) of video/audio tokens using the precomputed masks
                video_mask = video_masks_list[b]
                audio_mask = audio_masks_list[b]
                pure_video_hidden_states = batch_vision_audio_hidden[video_mask]  # [num_video_tokens, hidden_size]
                pure_audio_hidden_states = batch_vision_audio_hidden[audio_mask]  # [num_audio_tokens, hidden_size]
                pure_video_hidden_states_ori = batch_vision_audio_hidden_ori[video_mask]  # [num_video_tokens, hidden_size]
                pure_audio_hidden_states_ori = batch_vision_audio_hidden_ori[audio_mask]  # [num_audio_tokens, hidden_size]
                if batch_vision_audio_hidden_score is not None:
                    pure_video_hidden_states_score = batch_vision_audio_hidden_score[video_mask]  # [num_video_tokens, 1]
                    pure_audio_hidden_states_score = batch_vision_audio_hidden_score[audio_mask]  # [num_audio_tokens, 1]
                    # No global softmax over the whole sequence; compute_token_means applies softmax per tgt range instead
                else:
                    pure_video_hidden_states_score = None
                    pure_audio_hidden_states_score = None

                # 3.3) compute tgt and src video/audio token means (weighted with use_score_head, simple mean otherwise)

                if getattr(self.config, 'union_aggregation'):
                    sample_multimodal_means = compute_multimodal_tokens_mean(pure_video_hidden_states, pure_audio_hidden_states, video_token_indices_list, audio_token_indices_list, b, pure_video_hidden_states_score, pure_audio_hidden_states_score)
                    sample_multimodal_means_src = compute_multimodal_tokens_mean(pure_video_hidden_states_ori, pure_audio_hidden_states_ori, video_token_indices_list_src, audio_token_indices_list_src, b, None, None) if (video_token_indices_list_src and len(video_token_indices_list_src) > 0 and b < len(video_token_indices_list_src)) else []
                    # 3.4) update tgt and src evi token embeddings (add the mean feature onto the evi)
                    if evi_token_indices_list is not None and b < len(evi_token_indices_list):
                        update_union_evi_embeddings(inputs_embeds, evi_token_indices_list[b], sample_multimodal_means, b)
                    if evi_token_indices_list_src is not None and b < len(evi_token_indices_list_src):
                        update_union_evi_embeddings(inputs_embeds, evi_token_indices_list_src[b], sample_multimodal_means_src, b)
                else:

                    sample_video_means = compute_token_means(pure_video_hidden_states, video_token_indices_list, b, pure_video_hidden_states_score)
                    sample_audio_means = compute_token_means(pure_audio_hidden_states, audio_token_indices_list, b, pure_audio_hidden_states_score)
                    sample_video_means_src = compute_token_means(pure_video_hidden_states_ori, video_token_indices_list_src, b, None) if video_token_indices_list_src and len(video_token_indices_list_src) > 0 and b < len(video_token_indices_list_src) else None
                    sample_audio_means_src = compute_token_means(pure_audio_hidden_states_ori, audio_token_indices_list_src, b, None) if audio_token_indices_list_src and len(audio_token_indices_list_src) > 0 and b < len(audio_token_indices_list_src) else None
                    
                    video_token_means_list.append(sample_video_means)   # returned but unused downstream
                    audio_token_means_list.append(sample_audio_means)   # returned but unused downstream
                    video_token_means_list_src.append(sample_video_means_src)   # returned but unused downstream
                    audio_token_means_list_src.append(sample_audio_means_src)   # returned but unused downstream

                    # 3.4) update tgt and src evi token embeddings (handled uniformly)
                    if evi_token_indices_list is not None and b < len(evi_token_indices_list):
                        update_evi_embeddings(inputs_embeds, evi_token_indices_list[b], sample_video_means, sample_audio_means, b)
                    if evi_token_indices_list_src is not None and b < len(evi_token_indices_list_src):
                        update_evi_embeddings(inputs_embeds, evi_token_indices_list_src[b], sample_video_means_src, sample_audio_means_src, b)
        
        score_head_dummy = None
        if vision_audio_hidden_states_score is not None and len(vision_audio_hidden_states_score) > 0:
            score_head_dummy = sum(t.sum() for t in vision_audio_hidden_states_score)
        
        return (
            vision_audio_hidden_states,
            video_masks_list,
            audio_masks_list,
            evi_token_indices_list,
            evi_token_indices_list_src,
            video_token_means_list,
            audio_token_means_list,
            inputs_embeds,
            score_head_dummy,
        )

    def _compute_vision_info_per_timestep(
        self,
        b: int,
        vision_start_positions: torch.Tensor,
        vision_end_positions: torch.Tensor,
        input_ids: torch.LongTensor,
        frm_vision_tokens: torch.Tensor,
        num_timesteps: int,
        video_token_indices_per_timestep: list,
        audio_token_indices_per_timestep: list,
    ) -> torch.Tensor:
        """
        Using per-timestep video/audio token indices, average the corresponding hidden states
        over the vision sequence to obtain per-timestep vision info (video + audio_mean_weight * audio).
        """
        start_pos = vision_start_positions[b].item() + 1
        end_pos = vision_end_positions[b].item()
        vision_input_ids = input_ids[b, start_pos:end_pos]
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        video_mask = vision_input_ids == video_token_id
        audio_mask = vision_input_ids == audio_token_id
        device = frm_vision_tokens.device
        dtype = frm_vision_tokens.dtype
        hidden_size = frm_vision_tokens.shape[1]
        zero_vec = torch.zeros(hidden_size, device=device, dtype=dtype)

        vid_per_ts = video_token_indices_per_timestep[b] if b < len(video_token_indices_per_timestep) else []
        aud_per_ts = audio_token_indices_per_timestep[b] if b < len(audio_token_indices_per_timestep) else []

        video_positions = torch.where(video_mask)[0]
        audio_positions = torch.where(audio_mask)[0]
        n_video = video_positions.numel()
        n_audio = audio_positions.numel()

        def ranges_to_vision_indices(ranges, positions, max_len):
            if not ranges:
                return None
            indices_list = []
            for r in ranges:
                if isinstance(r, (tuple, list)) and len(r) == 2 and not isinstance(r[0], (tuple, list)):
                    ranges_one = [tuple(r)]
                else:
                    ranges_one = list(r) if r else []
                for (s, e) in ranges_one:
                    s = max(0, min(int(s), max_len))
                    e = max(s, min(int(e), max_len))
                    if s < e:
                        indices_list.append(positions[s:e])
            if not indices_list:
                return None
            return torch.cat(indices_list, dim=0)

        num_ts = max(num_timesteps, len(vid_per_ts), len(aud_per_ts))   # the three should be equal in theory
        video_timestep_means = []
        audio_timestep_means = []
        for t in range(num_ts): # iterate over timesteps; each timestep yields one mean feature of video+audio
            v_ranges = vid_per_ts[t] if t < len(vid_per_ts) else []
            a_ranges = aud_per_ts[t] if t < len(aud_per_ts) else []
            vid_idx = ranges_to_vision_indices(v_ranges, video_positions, n_video)
            aud_idx = ranges_to_vision_indices(a_ranges, audio_positions, n_audio)
            video_mean = frm_vision_tokens[vid_idx].mean(dim=0) if vid_idx is not None and vid_idx.numel() > 0 else zero_vec
            audio_mean = frm_vision_tokens[aud_idx].mean(dim=0) if aud_idx is not None and aud_idx.numel() > 0 else zero_vec
            video_timestep_means.append(video_mean)
            audio_timestep_means.append(audio_mean)
        video_timestep_means_tensor = torch.stack(video_timestep_means, dim=0)
        audio_timestep_means_tensor = torch.stack(audio_timestep_means, dim=0)
        aw = getattr(self.config, 'audio_mean_weight', 1.0)
        return video_timestep_means_tensor + aw * audio_timestep_means_tensor


    def _get_timestep_token_ranges_from_per_timestep(
        self,
        b: int,
        vision_start_positions: torch.Tensor,
        vision_end_positions: torch.Tensor,
        input_ids: torch.LongTensor,
        num_timesteps: int,
        video_token_indices_per_timestep: list,
        audio_token_indices_per_timestep: list,
    ) -> list:
        """
        Convert per-timestep video/audio indices into per-timestep token indices in the vision
        sequence, so generating can average tokens from cache_img_state over tgt timestep ranges.

        Returns:
            list of length num_timesteps; each element is (v_idx, a_idx), where
            v_idx/a_idx are index LongTensors into the vision segment (or None if the step has no video/audio).
        """
        start_pos = vision_start_positions[b].item() + 1
        end_pos = vision_end_positions[b].item()
        vision_input_ids = input_ids[b, start_pos:end_pos]
        video_mask = (vision_input_ids == self.config.video_token_id)
        audio_mask = (vision_input_ids == self.config.audio_token_id)
        video_positions = torch.where(video_mask)[0]
        audio_positions = torch.where(audio_mask)[0]
        n_video, n_audio = video_positions.numel(), audio_positions.numel()

        def _ranges_to_vision_indices(ranges, positions, max_len):
            """Convert per-timestep (s,e) ranges (in video/audio token index space) to an index tensor into the vision segment."""
            if not ranges:
                return None
            idx_list = []
            # A timestep's ranges may be [(s,e), ...] or a single (s,e)
            if isinstance(ranges, (tuple, list)) and len(ranges) == 2 and not isinstance(ranges[0], (tuple, list)):
                ranges = [tuple(ranges)]
            for r in ranges:
                if isinstance(r, (tuple, list)) and len(r) == 2 and not isinstance(r[0], (tuple, list)):
                    rs = [tuple(r)]
                else:
                    rs = list(r) if r else []
                for (s, e) in rs:
                    s = max(0, min(int(s), max_len))
                    e = max(s, min(int(e), max_len))
                    if s < e:
                        idx_list.extend(positions[s:e].tolist())
            if not idx_list:
                return None
            return torch.tensor(idx_list, dtype=torch.long, device=positions.device)

        vid_per_ts = video_token_indices_per_timestep[b] if b < len(video_token_indices_per_timestep) else []
        aud_per_ts = audio_token_indices_per_timestep[b] if b < len(audio_token_indices_per_timestep) else []
        out = []
        for t in range(num_timesteps):
            vr = vid_per_ts[t] if t < len(vid_per_ts) else []
            ar = aud_per_ts[t] if t < len(aud_per_ts) else []
            v_idx = _ranges_to_vision_indices(vr, video_positions, n_video)
            a_idx = _ranges_to_vision_indices(ar, audio_positions, n_audio)
            out.append((v_idx, a_idx))
        return out

    def _get_timestep_token_ranges(
        self,
        b: int,
        vision_start_positions: torch.Tensor,
        vision_end_positions: torch.Tensor,
        input_ids: torch.LongTensor,
        position_ids: torch.Tensor,
        num_timesteps: int,
    ) -> list:
        """
        Return each timestep's video/audio token index ranges in the vision sequence
        (used to average tokens from cache_img_state per timestep range).

        Returns:
            list of length num_timesteps; each element is ((v_start, v_end), (a_start, a_end))
            in vision-sequence indices (consistent with the row indices of cache_img_state[b]).
        """
        start_pos = vision_start_positions[b].item() + 1
        end_pos = vision_end_positions[b].item()
        vision_input_ids = input_ids[b, start_pos:end_pos]
        vision_position_ids = position_ids[0, b, start_pos:end_pos]
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        video_mask = vision_input_ids == video_token_id
        audio_mask = vision_input_ids == audio_token_id

        unique_timestep_ids = torch.unique(vision_position_ids)
        unique_timestep_ids_sorted = torch.sort(unique_timestep_ids)[0]
        timestep_ranges = []
        for timestep_idx in range(num_timesteps):
            if timestep_idx < len(unique_timestep_ids_sorted):
                timestep_pos_id = unique_timestep_ids_sorted[timestep_idx]
                timestep_mask = vision_position_ids == timestep_pos_id
            else:
                timestep_mask = torch.zeros_like(vision_position_ids, dtype=torch.bool)
            video_ts = (timestep_mask & video_mask).nonzero(as_tuple=True)[0]
            audio_ts = (timestep_mask & audio_mask).nonzero(as_tuple=True)[0]
            v_start, v_end = (video_ts.min().item(), video_ts.max().item() + 1) if video_ts.numel() > 0 else (0, 0)
            a_start, a_end = (audio_ts.min().item(), audio_ts.max().item() + 1) if audio_ts.numel() > 0 else (0, 0)
            timestep_ranges.append(((v_start, v_end), (a_start, a_end)))
        return timestep_ranges

    def evi_forward(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        labels: Optional[torch.LongTensor],
        position_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        use_cache: Optional[bool],
        output_attentions: Optional[bool],
        output_hidden_states: Optional[bool],
        return_dict: Optional[bool],
        cache_position: Optional[torch.LongTensor],
        tgt: Optional[list],
        video_token_indices_list: Optional[list],
        audio_token_indices_list: Optional[list],
        video_token_indices_list_src: Optional[list],
        audio_token_indices_list_src: Optional[list],
        tgt_timestep_distances: Optional[list],
        timestep_duration: Optional[float],
        video_token_indices_per_timestep: Optional[list] = None,
        audio_token_indices_per_timestep: Optional[list] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        event_id: Optional[torch.LongTensor] = None,
        parallel_start_idx: Optional[torch.LongTensor] = None,
        audio_lengths: Optional[list] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        last_text_token_indices: Optional[torch.LongTensor] = None,
        loss_kwargs: dict = None,
    ):
        if self.training:
            ########################################################################## Prepare the first forward pass
            use_attention_head = getattr(self.config, 'use_attention_head', False)
            # Default: recompute the attention row via hook in O(L*d); use_legacy_attn_row=True falls back to full-layer output_attentions materialization (for parity checks / fallback)
            use_legacy_attn_row = getattr(self.config, 'use_legacy_attn_row', False)
            output_attentions = True if (use_attention_head and use_legacy_attn_row) else output_attentions
            (
                input_ids_part1,
                inputs_embeds_part1,
                labels_part1,
                attention_mask_part1,
                position_ids_part1,
                vision_start_positions,
                vision_end_positions,
            ) = self._prepare_first_forward_pass(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                position_ids=position_ids,
                last_text_token_indices=last_text_token_indices,
            )
            ########################################################################## First forward pass (only the target layer's attention / hidden_states, saving memory)
            vision_audio_layer_index = getattr(self.config, 'vision_audio_layer_index', None)
            attn_row_handle, attn_row_storage = None, None
            if use_attention_head and not use_legacy_attn_row and vision_audio_layer_index is not None:
                attn_row_handle, attn_row_storage = self._register_attn_row_capture(vision_audio_layer_index)
            outputs_part1 = self.model(
                input_ids=None,
                position_ids=position_ids_part1,
                attention_mask=attention_mask_part1,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds_part1,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
                cache_position=cache_position,
                vision_start_positions=vision_start_positions,
                vision_end_positions=vision_end_positions,
                event_id=None,
                output_attentions_layer_index=vision_audio_layer_index if (use_attention_head and use_legacy_attn_row and getattr(self.config, 'save_gpu_trick')) else None,
                output_hidden_states_layer_index=vision_audio_layer_index if getattr(self.config, 'save_gpu_trick') else None,
            )
            if attn_row_handle is not None:
                attn_row_handle.remove()
            ########################################################################## Prepare the second forward pass
            if use_attention_head and last_text_token_indices is not None and vision_audio_layer_index is not None:
                if use_legacy_attn_row:
                    # Legacy path: full-layer attention materialization [B, num_heads, L, L] (O(L^2) memory), for parity checks / fallback only
                    attn = outputs_part1.attentions[vision_audio_layer_index]
                    batch_size = attn.shape[0]
                    last_text_token_indices_local = last_text_token_indices.to(device=attn.device).squeeze(-1)  # slice by last text token first, then average over heads
                    attn_sliced = attn[torch.arange(batch_size, device=attn.device), :, last_text_token_indices_local, :]  # [B, num_heads, L]
                    last_text_to_vision_attn_maps = attn_sliced.mean(dim=1).unsqueeze(-1)  # [B, seq_len, 1]
                else:
                    # New path: recompute the target row in O(L*d) from the hook-captured layer input
                    last_text_to_vision_attn_maps = self._attn_row_from_capture(attn_row_storage, last_text_token_indices, vision_audio_layer_index)
                    batch_size = last_text_to_vision_attn_maps.shape[0]
            else:
                last_text_to_vision_attn_maps = None
                batch_size = attention_mask.shape[0]
            
            (
                vision_audio_hidden_states,
                video_masks_list,
                audio_masks_list,
                evi_token_indices_list,
                evi_token_indices_list_src,
                video_token_means_list,
                audio_token_means_list,
                inputs_embeds,
                score_head_dummy,
            ) = self._prepare_second_forward_pass(
                vision_start_positions=vision_start_positions,
                vision_end_positions=vision_end_positions,
                hidden_states_part1=outputs_part1['hidden_states'],
                input_ids_part1=input_ids_part1,
                input_ids=input_ids,
                labels=labels,
                tgt=tgt,
                video_token_indices_list=video_token_indices_list,
                audio_token_indices_list=audio_token_indices_list,
                video_token_indices_list_src=video_token_indices_list_src,
                audio_token_indices_list_src=audio_token_indices_list_src,
                inputs_embeds=inputs_embeds,
                last_text_to_vision_attn_maps=last_text_to_vision_attn_maps,
            )

            ########################################################################## Second forward pass (only layer -2 hidden_states, saving memory)
            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=return_dict,
                cache_position=cache_position,
                vision_start_positions=vision_start_positions,
                vision_end_positions=vision_end_positions,
                event_id=event_id,
                output_hidden_states_layer_index=-2 if getattr(self.config, 'save_gpu_trick') else None,
            )
            ########################################################################## Compute loss
            hidden_states = outputs.last_hidden_state
            shift_labels = loss_kwargs.pop("shift_labels", None)
            loss = None
            logits = None

            # The if-block below builds the output labels (a plain shift of the input is not enough)
            if shift_labels is None:
                if event_id is not None:
                    new_labels_list = []
                    event_num = []
                    for b in range(batch_size):
                        if event_id[b] is None:
                            event_num.append(None)
                            new_labels_list.append(nn.functional.pad(labels[b], (0, 1), value=-100))
                            continue

                        # 1) number of tokens to delete (= number of events): the input contains two rounds of evi
                        #    but the output has only one, so the second round is removed when building labels
                        num_deleted_tokens = event_id[b].max().item() + 1
                        event_num.append(num_deleted_tokens)

                        # 2) start index of the deleted tokens
                        deleted_token_start_index = parallel_start_idx[b].item()

                        # 3) Insert the <S> token at the start index (the switch signal S in the paper marking the end
                        #    of planning; it exists only in labels, never in the input sequence, which teaches the model to
                        #    emit <S> at the end of the serial block), delete num_deleted_tokens anchor evi tokens starting
                        #    at that index (anchors are input-only, never predicted), and pad on the right
                        insert_token = torch.tensor([self.config.evi_end_token_id], device=labels.device, dtype=labels.dtype)
                        pad_length = max(0, num_deleted_tokens - 1) + 1 # the +1 matches the standard case, since labels are shifted by one
                        pad_tensor = torch.full((pad_length,), -100, device=labels.device, dtype=labels.dtype)
                        labels_b = torch.cat([
                            labels[b, :deleted_token_start_index],
                            insert_token,
                            labels[b, deleted_token_start_index + num_deleted_tokens:],
                            pad_tensor,
                        ])
                        new_labels_list.append(labels_b)
                    labels = torch.stack(new_labels_list, dim=0)
                else:
                    labels = nn.functional.pad(labels, (0, 1), value=-100)
                shift_labels = labels[..., 1:].contiguous()

            frm_tokens_all = self.frm_head(outputs.hidden_states[-2])   # process all tokens at once, easier with batching
            evi_tokens_all = self.evi_head(outputs.hidden_states[-2])
                
            
            batch_size = frm_tokens_all.shape[0]
            # Initialize as torch.Tensor to keep the type consistent (even when the value is 0)
            device = frm_tokens_all.device
            dtype = frm_tokens_all.dtype
            loss_match = torch.tensor(0.0, device=device, dtype=dtype)
            avg_factor = 0
            for b in range(batch_size):
                # step1: get audio + video tokens
                frm_vision_tokens = frm_tokens_all[b, vision_start_positions[b].item()+1:vision_end_positions[b].item()]  # [vision_token_len, hidden_size]

                # step2: read GT time distances and compute GT similarity
                # Check whether tgt_timestep_distances is empty
                if (tgt_timestep_distances is not None 
                    and b < len(tgt_timestep_distances) 
                    and tgt_timestep_distances[b] is not None 
                    and len(tgt_timestep_distances[b]) > 0):
                    tgt_timestep_distances_b = torch.tensor(tgt_timestep_distances[b],dtype=torch.float32,device=frm_tokens_all.device)
                    if event_id is not None and event_id[b] is not None:
                        # dvc only: tgt is mirrored (planning + anchors), so keep the first event_num
                        # entries. Single-event tasks store the interval once and need no truncation.
                        tgt_timestep_distances_b = tgt_timestep_distances_b[:event_num[b]]
                    sim_tgt = torch.pow(0.5, tgt_timestep_distances_b)  # [num_time_intervals, num_timesteps]

                    # step3: compute vision info from per-timestep video/audio indices
                    vision_info_tensor = self._compute_vision_info_per_timestep(
                        b=b,
                        vision_start_positions=vision_start_positions,
                        vision_end_positions=vision_end_positions,
                        input_ids=input_ids,
                        frm_vision_tokens=frm_vision_tokens,
                        num_timesteps=tgt_timestep_distances_b.shape[1],
                        video_token_indices_per_timestep=video_token_indices_per_timestep or [],
                        audio_token_indices_per_timestep=audio_token_indices_per_timestep or [],
                    )  # [num_timesteps, hidden_size]
                    
                    # step4: compute the similarity loss
                    inds = torch.where(shift_labels[b] == self.config.evi_token_id)[0]
                    if len(inds) > 0:  # only compute the loss when evi tokens exist
                        evi_tokens = evi_tokens_all[b][inds]  # [num_evi_tokens, hidden_size]
                        sim_pred = torch.matmul(evi_tokens, vision_info_tensor.t()) / (evi_tokens.size(1) ** 0.5)  # [num_evi_tokens, num_timesteps]
                        # sim_pred = torch.sigmoid(sim_pred)  

                        loss_match = loss_match + F.binary_cross_entropy_with_logits(sim_pred, sim_tgt)
                        avg_factor += 1

            # The LM loss is computed once outside the for loop, not per sample
            # (computing it inside the loop would only reflect the last sample)
            if labels is not None or shift_labels is not None:
                loss = LigerForCausalLMLoss(
                    hidden_states=hidden_states,
                    lm_head_weight=self.lm_head.weight,
                    labels=labels,
                    shift_labels=shift_labels,
                    hidden_size=self.config.hidden_size,
                    **loss_kwargs,
                )
            else:
                # No labels: loss stays None (inference mode)
                loss = None

            if loss is not None:
                # Distributed training: keep frm_head/evi_head/score_head in the computation graph on all ranks.
                # When step3 is skipped (e.g. this rank's batch has no video/audio_token_indices), score_head does not
                # flow into the loss through inputs_embeds; attaching (score_head_dummy * 0) to the loss avoids
                # DDP/DeepSpeed communication mismatches from ranks not backpropagating score_head.
                dummy_terms = (frm_tokens_all.sum() + evi_tokens_all.sum()) * 0
                if score_head_dummy is not None:
                    dummy_terms = dummy_terms + score_head_dummy * 0
                if avg_factor > 0:
                    loss_match_weight = getattr(self.config, 'loss_match_weight', 3.0)
                    loss = loss + (loss_match / avg_factor) * loss_match_weight + dummy_terms
                else:
                    loss = loss + dummy_terms + loss_match * 0

            return Qwen2_5_VLCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=self.rope_deltas,
                loss_match=loss_match,
            )
        else:
            use_attention_head = getattr(self.config, 'use_attention_head', False)
            use_legacy_attn_row = getattr(self.config, 'use_legacy_attn_row', False)
            output_attentions = True if (use_attention_head and use_legacy_attn_row) else output_attentions
            cache_len = (
                0 if past_key_values is None
                else (past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else len(past_key_values))
            )
            mode = "caching" if cache_len == 0 else "generating"

            # Forward-pass preparation (evi embedding update)
            vision_start_positions, vision_end_positions,event_id = self._prepare_forward_pass_for_infer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                mode=mode,
                video_token_indices_list_src=video_token_indices_list_src,
                audio_token_indices_list_src=audio_token_indices_list_src,
                cache_position=cache_position,
                past_key_values=past_key_values,
            )
            input_ids_copy = input_ids.clone()
            if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):   
                # calculate RoPE index once per generation in the pre-fill stage only
                if (
                    (cache_position is not None and cache_position[0] == 0)     # compute position_ids and self.rope_deltas in the caching phase
                    or self.rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
                ):         
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        audio_lengths,
                        second_per_grid_ts,
                        attention_mask,
                    )
                    self.rope_deltas = rope_deltas
                # then use the prev pre-calculated rope-deltas to get the correct position ids
                else:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    
                    # Compute new_cache_position first (for parallel samples)
                    new_cache_position = [None] * batch_size
                    for b in range(batch_size):
                        if hasattr(self, 'parallel') and self.parallel[b]:
                            new_cache_position[b] = (cache_position.item() - self.serial_end_positions[b]- 1) // self.evi_count[b] + 1 + self.serial_end_positions[b]
                    
                    # Compute delta per sample
                    if cache_position is not None:
                        delta_list = []
                        for b in range(batch_size):
                            if hasattr(self, 'parallel') and self.parallel[b] and new_cache_position[b] is not None:
                                # Parallel sample: use new_cache_position[b]
                                cache_pos_b = new_cache_position[b]
                            else:
                                # Non-parallel sample: use cache_position[0]
                                cache_pos_b = cache_position[0].item()
                            
                            # self.rope_deltas has shape [batch_size, 1], so self.rope_deltas[b] is [1]; .item() extracts the scalar
                            delta_b = cache_pos_b + self.rope_deltas[b].item()
                            delta_list.append(delta_b)
                        
                        delta = torch.tensor(delta_list, dtype=torch.long, device=inputs_embeds.device).unsqueeze(1)  # [batch_size, 1]
                    else:
                        assert False, "cache_position should not be None in generating phase"
                    
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    position_ids = position_ids.add(delta)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                            

            # Forward pass (in the caching phase, register the attn-row capture hook beforehand if the hook path is used)
            attn_row_handle, attn_row_storage = None, None
            infer_layer_index = getattr(self.config, 'vision_audio_layer_index', None)
            if (mode == 'caching' and use_attention_head and not use_legacy_attn_row
                    and infer_layer_index is not None):
                attn_row_handle, attn_row_storage = self._register_attn_row_capture(infer_layer_index)
            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
                cache_position=cache_position,
                vision_start_positions=vision_start_positions,
                vision_end_positions=vision_end_positions,
            )
            if attn_row_handle is not None:
                attn_row_handle.remove()
                # Compute attention rows [B, seq_len, 1] for the whole batch at once, for the caching loop below
                attn_rows_all = self._attn_row_from_capture(attn_row_storage, last_text_token_indices, infer_layer_index)
            else:
                attn_rows_all = None

            # Process forward-pass outputs in the caching phase
            if mode == 'caching':
                self.cache_img_state = []
                self.cache_img_state_score = []
                self.cache_timestep_token_ranges = []
                self.cache_frm_tokens_all = []
                batch_size = inputs_embeds.shape[0]
                self.tgt = [[] for _ in range(batch_size)]  
                self.serial_evi_tgts = [[] for _ in range(batch_size)]  
                self.serial_end_positions = [None] * batch_size  
                self.parallel = [False] * batch_size  
                self.event_id = [None] * batch_size  
                self.evi_count = [0] * batch_size

                for b in range(batch_size):
                    # Number of timesteps: taken from the per-timestep index list length (audio_token_indices_per_timestep preferred)
                    if audio_token_indices_per_timestep and b < len(audio_token_indices_per_timestep):
                        num_timesteps = len(audio_token_indices_per_timestep[b])
                    elif video_token_indices_per_timestep and b < len(video_token_indices_per_timestep):
                        num_timesteps = len(video_token_indices_per_timestep[b])
                    else:
                        num_timesteps = 0

                    # Use video/audio_token_indices_per_timestep to get each timestep's range within the vision sequence, for tgt-based averaging during generating
                    self.cache_timestep_token_ranges.append(
                        self._get_timestep_token_ranges_from_per_timestep(
                            b=b,
                            vision_start_positions=vision_start_positions,
                            vision_end_positions=vision_end_positions,
                            input_ids=input_ids_copy,
                            num_timesteps=num_timesteps,
                            video_token_indices_per_timestep=video_token_indices_per_timestep or [],
                            audio_token_indices_per_timestep=audio_token_indices_per_timestep or [],
                        )
                    )

                    # Extract the vision segment from hidden states
                    vision_audio_layer_index = getattr(self.config, 'vision_audio_layer_index', None)
                    layer_hidden_states = outputs.hidden_states[vision_audio_layer_index]  # [batch_size, seq_len, hidden_size]
                    vision_audio_hidden = layer_hidden_states[b, vision_start_positions[b].item()+1:vision_end_positions[b].item()]  # [vision_token_len, hidden_size]

                    # Save processed vision tokens and scores for later weighting
                    use_score_head = getattr(self.config, 'use_score_head', False)
                    use_attention_head = getattr(self.config, 'use_attention_head', False)
                    if use_score_head:
                        score_tokens_all = self.score_head(outputs.hidden_states[-2])
                        score_tokens = score_tokens_all[b, vision_start_positions[b].item()+1:vision_end_positions[b].item()]  # [vision_token_len, 1]
                        last_text_to_vision_attn_maps = torch.sigmoid(score_tokens)  # [vision_token_len, 1]
                    elif use_attention_head:
                        if attn_rows_all is not None:
                            # New path: rows recomputed by the hook [B, seq_len, 1]
                            attn_maps_all = attn_rows_all
                        else:
                            # Legacy path (use_legacy_attn_row=True): materialize the full-layer attention, then slice out the row
                            attn = outputs.attentions[vision_audio_layer_index]
                            batch_size = attn.shape[0]
                            last_text_token_indices_local = last_text_token_indices.to(device=attn.device).squeeze(-1)  # slice at the last text token, then average over heads
                            attn_sliced = attn[torch.arange(batch_size, device=attn.device), :, last_text_token_indices_local, :]  # [B, num_heads, L]
                            attn_maps_all = attn_sliced.mean(dim=1).unsqueeze(-1)  # [B, seq_len, 1]
                        # Keep only the vision-token part
                        last_text_to_vision_attn_maps = attn_maps_all[b, vision_start_positions[b].item()+1:vision_end_positions[b].item()]  # [vision_token_len, 1]
                    else:
                        last_text_to_vision_attn_maps = None
                    self.cache_img_state.append(vision_audio_hidden)
                    self.cache_img_state_score.append(last_text_to_vision_attn_maps)

                    # Pass hidden states through the frm head to get frm_tokens_all, then extract the vision tokens: frm_tokens
                    frm_tokens_all = self.frm_head(outputs.hidden_states[-2])
                    frm_tokens = frm_tokens_all[b, vision_start_positions[b].item()+1:vision_end_positions[b].item()]

                    # Pass hidden states through the evi head to get evi_tokens_all, then take the last token: evi_tokens [1, hidden_size]
                    evi_tokens_all = self.evi_head(outputs.hidden_states[-2])
                    evi_tokens = evi_tokens_all[b][-1].unsqueeze(0)

                    # Compute per-timestep vision info (using per-timestep indices) and cache it
                    vision_info_tensor = self._compute_vision_info_per_timestep(
                        b=b,
                        vision_start_positions=vision_start_positions,
                        vision_end_positions=vision_end_positions,
                        input_ids=input_ids_copy,
                        frm_vision_tokens=frm_tokens,
                        num_timesteps=num_timesteps,
                        video_token_indices_per_timestep=video_token_indices_per_timestep or [],
                        audio_token_indices_per_timestep=audio_token_indices_per_timestep or [],
                    )  # [num_timesteps, hidden_size]
                    self.cache_frm_tokens_all.append(vision_info_tensor)

                    # Compute similarity and tgt from frm_tokens and evi_tokens, then store into sim and tgt
                    # self.sim = []
                    sim_pred = torch.matmul(evi_tokens, vision_info_tensor.t()) / (evi_tokens.size(1) ** 0.5)  # [num_evi_tokens, num_timesteps]
                    sim_pred = torch.sigmoid(sim_pred)  
                    tgt_pred = self.find_and_merge(sim_pred)
                    # Store as a single-interval list [[s,e]] (one event maps to one grounded interval)
                    self.tgt[b].append(tgt_pred if tgt_pred else [[0, 0]])


                # Compute logits
                loss = None
                logits = None
                logits = self.lm_head(outputs.last_hidden_state)
                if not return_dict:
                    output = (logits,) + outputs[1:]
                    return (loss,) + output if loss is not None else output
                return Qwen2_5_VLCausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                    rope_deltas=self.rope_deltas,
                )
            elif mode == 'generating':
                batch_size = inputs_embeds.shape[0]
                for b in range(batch_size):
                    # Pass hidden states through the evi head to get evi_tokens_all, then take the last token: evi_tokens [1, hidden_size]
                    evi_tokens_all = self.evi_head(outputs.hidden_states[-2])
                    evi_tokens = evi_tokens_all[b][-1].unsqueeze(0)
    
                    # Fetch the cached per-timestep vision info from self.cache_frm_tokens_all and compute similarity
                    vision_info_tensor = self.cache_frm_tokens_all[b]
                    sim_pred = torch.matmul(evi_tokens, vision_info_tensor.t()) / (evi_tokens.size(1) ** 0.5)  # [num_evi_tokens, num_timesteps]
                    sim_pred = torch.sigmoid(sim_pred)
                    tgt_pred = self.find_and_merge(sim_pred)
                    # Anchor evi in the parallel phase (this step's output is forced to the j-th anchor): its interval must match serial G_j.
                    # Override the prediction computed from the <S>/previous-anchor hidden state so the anchor's timestamp = planning-phase grounding.
                    appended = None
                    if getattr(self, 'serial_end_positions', None) is not None and self.serial_end_positions[b] is not None:
                        anchor_j = cache_position.item() - self.serial_end_positions[b]
                        if 0 <= anchor_j < len(self.serial_evi_tgts[b]):
                            appended = self.serial_evi_tgts[b][anchor_j]
                    if appended is None:
                        appended = tgt_pred if tgt_pred else [[0, 0]]
                    self.tgt[b].append(appended)

                # Compute logits
                loss = None
                logits = None
                logits = self.lm_head(outputs.last_hidden_state)
                if not return_dict:
                    output = (logits,) + outputs[1:]
                    return (loss,) + output if loss is not None else output
                return Qwen2_5_VLCausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                    rope_deltas=self.rope_deltas,
                )

    def sft_forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        audio_feature: Optional[torch.Tensor] = None,
        audio_lengths: Optional[list] = None,
        tgt: Optional[list] = None,
        video_token_indices_list: Optional[list] = None,
        audio_token_indices_list: Optional[list] = None,
        video_token_indices_list_src: Optional[list] = None,
        audio_token_indices_list_src: Optional[list] = None,
        tgt_timestep_distances: Optional[list] = None,
        timestep_duration: Optional[float] = None,
        video_token_indices_per_timestep: Optional[list] = None,
        audio_token_indices_per_timestep: Optional[list] = None,
        event_id: Optional[torch.LongTensor] = None,
        parallel_start_idx: Optional[torch.LongTensor] = None,
        **loss_kwargs,
    ):
        # step1: prepare forward-pass arguments (attention mask, position ids, inputs embeds, etc.)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # step2: build inputs embeds from input_ids + video/audio embeddings, then build the attention mask
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)  # convert token ids to embeddings; video/audio features are merged in below
            if pixel_values is not None:
                inputs_embeds, _ = self.get_image_embeds(pixel_values, image_grid_thw, input_ids, inputs_embeds)
            if pixel_values_videos is not None:
                inputs_embeds, _ = self.get_video_embeds(pixel_values_videos, video_grid_thw, input_ids, inputs_embeds) # splice encoder video features into the sequence
            if audio_feature is not None:   # same as video above
                inputs_embeds, _ = self.get_audio_embeds(audio_feature, input_ids, inputs_embeds)
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # step3.a: evidence-mode forward pass
        use_evidence = getattr(self.config, 'use_evidence', False)
        if use_evidence:
            return self.evi_forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                tgt=tgt,
                video_token_indices_list=video_token_indices_list,
                audio_token_indices_list=audio_token_indices_list,
                video_token_indices_list_src=video_token_indices_list_src,
                audio_token_indices_list_src=audio_token_indices_list_src,
                tgt_timestep_distances=tgt_timestep_distances,
                timestep_duration=timestep_duration,
                video_token_indices_per_timestep=video_token_indices_per_timestep,
                audio_token_indices_per_timestep=audio_token_indices_per_timestep,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                audio_lengths=audio_lengths,
                second_per_grid_ts=second_per_grid_ts,
                event_id=event_id,
                parallel_start_idx=parallel_start_idx,
                last_text_token_indices=loss_kwargs.get("last_text_token_indices", None) if loss_kwargs else None,
                loss_kwargs=loss_kwargs,
            )
        else:
            # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
            if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):   # not entered in the current pipeline
                # calculate RoPE index once per generation in the pre-fill stage only
                if (
                    (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
                ):
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        audio_lengths,
                        second_per_grid_ts,
                        attention_mask,
                    )
                    self.rope_deltas = rope_deltas
                # then use the prev pre-calculated rope-deltas to get the correct position ids
                else:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    delta = (
                        (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                        if cache_position is not None
                        else 0
                    )
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    if cache_position is not None:  # otherwise `deltas` is an int `0`
                        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    position_ids = position_ids.add(delta)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # step3.b: sft-mode forward pass
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
                # output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
            event_id=event_id,
        )

        hidden_states = outputs[0]

        shift_labels = loss_kwargs.pop("shift_labels", None)
        loss = None
        logits = None

        # step4: compute loss in training mode, logits in eval mode
        if self.training and (labels is not None or shift_labels is not None):
            loss = LigerForCausalLMLoss(
                hidden_states=hidden_states,
                lm_head_weight=self.lm_head.weight,
                labels=labels,
                shift_labels=shift_labels,
                hidden_size=self.config.hidden_size,
                **loss_kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            if labels is not None:
                # Upcast to float if we need to compute the loss to avoid potential precision issues
                logits = logits.float()
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def dpo_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        chosen_ids: Optional[torch.LongTensor] = None,
        reject_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        reject_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        chosen_position_ids: Optional[torch.LongTensor] = None,
        reject_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        chosen_labels: Optional[torch.LongTensor] = None,
        reject_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        train_type: Optional[str] = None,
        audio_feature: Optional[torch.Tensor] = None,
        audio_lengths: Optional[list] = None,
        **loss_kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        chosen_embeds = self.model.embed_tokens(chosen_ids)
        reject_embeds = self.model.embed_tokens(reject_ids)
        if pixel_values is not None:
            chosen_embeds, image_embeds = self.get_image_embeds(pixel_values, image_grid_thw, chosen_ids, chosen_embeds)
            reject_embeds = self.get_image_embeds(pixel_values, image_grid_thw, reject_ids, reject_embeds, image_embeds)

        if pixel_values_videos is not None:
            chosen_embeds, video_embeds = self.get_video_embeds(pixel_values_videos, video_grid_thw, chosen_ids, chosen_embeds)
            reject_embeds = self.get_video_embeds(pixel_values_videos, video_grid_thw, reject_ids, reject_embeds, video_embeds)

        if audio_feature is not None:
            chosen_embeds, audio_embeds = self.get_audio_embeds(audio_feature, chosen_ids, chosen_embeds)
            reject_embeds = self.get_audio_embeds(audio_feature, reject_ids, reject_embeds, audio_embeds)

        if chosen_attention_mask is not None:
            chosen_attention_mask = chosen_attention_mask.to(chosen_embeds.device)

        if reject_attention_mask is not None:
            reject_attention_mask = reject_attention_mask.to(reject_embeds.device)

        policy_chosen_outputs = self.model(
            input_ids=None,
            position_ids=chosen_position_ids,
            attention_mask=chosen_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=chosen_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        policy_reject_outputs = self.model(
            input_ids=None,
            position_ids=reject_position_ids,
            attention_mask=reject_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=reject_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        policy_chosen_hidden_states = policy_chosen_outputs[0]
        policy_reject_hidden_states = policy_reject_outputs[0]

        len_input = sum(labels[0] == IGNORE_INDEX)
        policy_chosen_hidden_states = policy_chosen_hidden_states[:, len_input-1:-1, :]
        policy_reject_hidden_states = policy_reject_hidden_states[:, len_input-1:-1, :]

        chosen_labels_in = chosen_labels[:, len_input:]
        reject_labels_in = reject_labels[:, len_input:]

        inputs_return = pad_and_stack(
            policy_chosen_hidden_states,
            policy_reject_hidden_states,
            0.0
        )
        target_return = pad_and_stack(
            chosen_labels_in,
            reject_labels_in,
            -100
        )
        if train_type == "gdpo":
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                inputs_embeds = self.get_image_embeds(pixel_values, image_grid_thw, input_ids, inputs_embeds, image_embeds)

            if pixel_values_videos is not None:
                inputs_embeds = self.get_video_embeds(pixel_values_videos, video_grid_thw, input_ids, inputs_embeds, video_embeds)

            if audio_feature is not None:
                inputs_embeds = self.get_audio_embeds(audio_feature, input_ids, inputs_embeds, audio_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
            hidden_states = outputs[0]
            shift_labels = loss_kwargs.pop("shift_labels", None)
            loss = LigerForCausalLMLoss(
                hidden_states=hidden_states,
                lm_head_weight=self.lm_head.weight,
                labels=labels,
                shift_labels=shift_labels,
                hidden_size=self.config.hidden_size,
                **loss_kwargs,
            )
            return inputs_return, target_return, loss
        else:
            return inputs_return, target_return

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        chosen_ids: Optional[torch.LongTensor] = None,
        reject_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        reject_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        chosen_position_ids: Optional[torch.LongTensor] = None,
        reject_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        chosen_labels: Optional[torch.LongTensor] = None,
        reject_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        train_type: Optional[str] = "sft",
        audio_feature: Optional[torch.Tensor] = None,
        audio_lengths: Optional[list] = None,
        tgt: Optional[list] = None,
        video_token_indices_list: Optional[list] = None,
        audio_token_indices_list: Optional[list] = None,
        video_token_indices_list_src: Optional[list] = None,
        audio_token_indices_list_src: Optional[list] = None,
        tgt_timestep_distances: Optional[list] = None,
        timestep_duration: Optional[float] = None,
        video_token_indices_per_timestep: Optional[list] = None,
        audio_token_indices_per_timestep: Optional[list] = None,
        event_id: Optional[torch.LongTensor] = None,
        parallel_start_idx: Optional[torch.LongTensor] = None,
        **loss_kwargs,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        if train_type == "sft":
            return self.sft_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                cache_position=cache_position,
                second_per_grid_ts=second_per_grid_ts, 
                audio_feature=audio_feature,
                audio_lengths=audio_lengths,
                tgt=tgt,
                video_token_indices_list=video_token_indices_list,
                audio_token_indices_list=audio_token_indices_list,
                video_token_indices_list_src=video_token_indices_list_src,
                audio_token_indices_list_src=audio_token_indices_list_src,
                tgt_timestep_distances=tgt_timestep_distances,
                timestep_duration=timestep_duration,
                video_token_indices_per_timestep=video_token_indices_per_timestep,
                audio_token_indices_per_timestep=audio_token_indices_per_timestep,
                event_id=event_id,
                parallel_start_idx=parallel_start_idx,
                **loss_kwargs
            )
        elif train_type == "dpo" or train_type == "gdpo":
            return self.dpo_forward(
                input_ids=input_ids,
                chosen_ids=chosen_ids,
                reject_ids=reject_ids,
                attention_mask=attention_mask,
                chosen_attention_mask=chosen_attention_mask,
                reject_attention_mask=reject_attention_mask,
                position_ids=position_ids,
                chosen_position_ids=chosen_position_ids,
                reject_position_ids=reject_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                chosen_labels=chosen_labels,
                reject_labels=reject_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                cache_position=cache_position,
                second_per_grid_ts=second_per_grid_ts, 
                train_type=train_type,
                audio_feature=audio_feature,
                audio_lengths=audio_lengths,
                **loss_kwargs
            )
        else:
            raise NotImplementedError

    def _aggregate_interval_feature(self, b, intervals):
        """Aggregate video/audio features from the prefill cache over the predicted time intervals; returns a [hidden_size] vector.
        Same logic as the generating branch of _prepare_forward_pass_for_infer (per-modality softmax weighting
        or plain mean within the interval, video + audio_mean_weight * audio); used as the input embedding of the parallel anchor <G>."""
        cached = self.cache_timestep_token_ranges[b]
        cache_hidden = self.cache_img_state[b]
        cache_hidden_score = self.cache_img_state_score[b] if self.cache_img_state_score[b] is not None else None
        hidden_size = cache_hidden.shape[1]
        device, dtype = cache_hidden.device, cache_hidden.dtype
        video_list, audio_list, video_scores_list, audio_scores_list = [], [], [], []
        ts_list = [ts for seg_start, seg_end in intervals for ts in range(seg_start, seg_end + 1)]
        for ts in ts_list:
            if ts < 0 or ts >= len(cached):
                continue
            v_idx, a_idx = cached[ts]
            if v_idx is not None and v_idx.numel() > 0:
                video_list.append(cache_hidden[v_idx])
                if cache_hidden_score is not None:
                    video_scores_list.append(cache_hidden_score[v_idx])
            if a_idx is not None and a_idx.numel() > 0:
                audio_list.append(cache_hidden[a_idx])
                if cache_hidden_score is not None:
                    audio_scores_list.append(cache_hidden_score[a_idx])

        def _weighted_mean(feat_list, score_list):
            if not feat_list:
                return torch.zeros(hidden_size, device=device, dtype=dtype)
            feat = torch.cat(feat_list, dim=0)
            if cache_hidden_score is not None and len(score_list) > 0:
                score = torch.cat(score_list, dim=0)
                score = F.softmax(score.squeeze(-1), dim=0).unsqueeze(-1)
                return torch.sum(feat * score, dim=0).to(device=device, dtype=dtype)
            return torch.mean(feat, dim=0).to(device=device, dtype=dtype)

        aw = getattr(self.config, 'audio_mean_weight', 1.0)
        return _weighted_mean(video_list, video_scores_list) + aw * _weighted_mean(audio_list, audio_scores_list)

    def _branch_visibility_mask(self, num_query, kv_len, prefix_len, dtype, device):
        """Branch-isolating 4D additive mask for parallel decoding, shape [1, 1, K, kv_len].
        Visibility rules (matching training-time event-factorized attention):
        - kv index < prefix_len (prompt + serial planning segment; <S> is never fed to the KV): visible to all branches;
        - later kv entries belong to branch (k - prefix_len) % K in round-robin append order
          (anchors and per-round tokens are appended in branch order): visible only to the same branch;
        - among the current K new tokens, each sees only itself (synchronous decoding, no cross-talk)."""
        K = num_query
        neg = torch.finfo(dtype).min
        mask = torch.full((1, 1, K, kv_len), neg, dtype=dtype, device=device)
        mask[:, :, :, :prefix_len] = 0.0
        if kv_len > prefix_len:
            owners = torch.arange(kv_len - prefix_len, device=device) % K       # [num_parallel_kv]
            visible = owners.unsqueeze(0) == torch.arange(K, device=device).unsqueeze(1)  # [K, num_parallel_kv]
            mask[0, 0, :, prefix_len:] = torch.where(
                visible, torch.tensor(0.0, dtype=dtype, device=device), torch.tensor(neg, dtype=dtype, device=device))
        return mask

    @torch.no_grad()
    def generate_parallel(
        self,
        input_ids=None,
        max_new_tokens=1024,
        do_sample=False,
        top_p=0.9,
        eos_token_id=None,
        constrain_planning=False,
        record_separator_id=None,
        deinterleave=True,
        **gen_kwargs,
    ):
        """Event-level parallel decoding (the inference protocol of paper Fig.2/3):
        Phase 1 serial decoding: standard autoregressive generation. Single-event tasks
              (grounding / segment captioning / cross-modal captioning) complete entirely in this
              phase - one <G> followed by the answer text (or nothing), ending on EOS. Dense video
              captioning ends this phase by generating <S> (the switch token), which triggers the
              parallel phases below.
        Phase 2 anchor feeding: on <S>, the K parallel anchor <G> tokens are fed in a single
              forward (each embedding is augmented with its event's aggregated feature; anchors are
              mutually invisible and share the same position);
        Phase 3 parallel decoding: each step forwards K tokens at once (one per branch), with the
              branch-isolating mask + shared positions; each branch terminates independently on EOS.
        Returns a [1, L] flat sequence: prompt + one record per event ("<G_j> caption_j"), records
              separated by record_separator_id. The K anchors are fed to the model but left out of
              the returned sequence, and the planning-end <S> is dropped.
        Also rebuilds self.tgt[0] as an interval list aligned position-by-position with the generated part (directly usable by test post-processing).
        If <S> is never generated (single-event tasks), behaves exactly like plain generate."""
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

        evi_id = self.config.evi_token_id
        evi_end_id = self.config.evi_end_token_id
        anchor_id = evi_id   # anchors reuse the <G> token (fed-in copies, never predicted)
        if eos_token_id is None:
            eos_token_id = self.generation_config.eos_token_id
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0]

        class _StopOnEviEnd(StoppingCriteria):
            def __call__(self, ids, scores, **kwargs):
                return ids[0, -1].item() == evi_end_id

        prompt_len_for_guard = input_ids.shape[1]

        class _ForcePlanningStop(LogitsProcessor):
            """Planning stop guard (inference-side): once the serial <G> count reaches the cap
            (default 30 = max event count in the training set), force <S> so a runaway planning
            phase cannot exhaust the token budget without ever switching to the parallel branches.
            Single-event tasks emit a single <G> and are never affected.
            Disable via model.config.max_planning_events = 0 or None."""

            def __init__(self, cap):
                self.cap = cap

            def __call__(self, ids, scores):
                if self.cap and int((ids[0, prompt_len_for_guard:] == evi_id).sum().item()) >= self.cap:
                    forced = torch.full_like(scores, torch.finfo(scores.dtype).min)
                    forced[:, evi_end_id] = 0.0
                    return forced
                return scores

        class _PlanningVocabConstraint(LogitsProcessor):
            """dvc-only guard (constrain_planning=True): restrict the serial phase to <G>/<S>.
            Under the current protocol 6 of the 7 training tasks continue with plain text right
            after <G>, so an under-trained model tends to slide into that majority mode on dvc
            and never emits <S> -- the parallel phase then never fires. This constraint makes
            the planning->parallel switch inevitable. Single-event tasks must NOT use it
            (their answer text lives in the serial phase)."""

            def __call__(self, ids, scores):
                out = torch.full_like(scores, torch.finfo(scores.dtype).min)
                out[:, evi_id] = scores[:, evi_id]
                out[:, evi_end_id] = scores[:, evi_end_id]
                return out

        max_planning_events = getattr(self.config, 'max_planning_events', 30)
        planning_processors = [_ForcePlanningStop(max_planning_events)]
        if constrain_planning:
            planning_processors.insert(0, _PlanningVocabConstraint())

        # ---------- Phase 1: serial decoding (fully reuses the existing generate machinery and cache/aggregation/tgt recording).
        # Vocabulary is unconstrained: single-event tasks produce their whole answer here and end
        # on EOS; only generating <S> switches into the parallel phases. The full max_new_tokens
        # budget applies because for single-event tasks this phase IS the answer. ----------
        out1 = self.generate(
            input_ids=input_ids,
            **gen_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            stopping_criteria=StoppingCriteriaList([_StopOnEviEnd()]),
            logits_processor=LogitsProcessorList(planning_processors),
            return_dict_in_generate=True,
            use_cache=True,
        )
        seq = out1.sequences  # [1, prompt_len + serial_gen_len]
        cache = out1.past_key_values
        prompt_len = input_ids.shape[1]
        device = seq.device

        if (seq[0, -1].item() != evi_end_id or cache is None
                or not getattr(self, 'evi_count', None) or self.evi_count[0] == 0):
            return seq  # parallel phase not entered: same behavior as plain generate
        K = self.evi_count[0]

        # ---------- Phase 2a: <S> is not written into the KV cache (aligned with training) ----------
        # At training time the anchors' conditioning context = prompt + serial evis (<S> exists only
        # in labels, never in the input), and anchor position = last serial evi + 1. Inference stays
        # consistent: <S> is a pure stop signal (like EOS, generating it triggers the switch but it
        # never conditions the context), so the anchors directly follow the last serial evi.
        prefix_len = cache.get_seq_length()      # prefix visible to all branches: prompt + serial segment (excluding <S>)
        anchor_pos = prefix_len                  # shared logical position of the anchors = last_evi + 1, matching training
        embed = self.get_input_embeddings()
        compute_dtype = self.lm_head.weight.dtype
        rope_delta = self.rope_deltas[0].item() if self.rope_deltas is not None else 0

        def _parallel_step(token_embeds, logical_pos):
            """One parallel forward step: K tokens share position logical_pos, branches isolated. Returns [K, vocab] logits."""
            cur_len = cache.get_seq_length()
            kv_len = cur_len + K
            mask = self._branch_visibility_mask(K, kv_len, prefix_len, compute_dtype, device)
            pos = torch.full((3, 1, K), logical_pos + rope_delta, dtype=torch.long, device=device)
            outputs = self.model(
                inputs_embeds=token_embeds,
                position_ids=pos,
                attention_mask=mask,
                past_key_values=cache,
                cache_position=torch.arange(cur_len, cur_len + K, device=device),
                use_cache=True,
            )
            return self.lm_head(outputs.last_hidden_state[0])  # [K, vocab]

        def _sample_tokens(logits):
            # Constrained decoding for the branch phase: per protocol, branch content must not contain
            # protocol tokens (<G>/<S>/<shift> belong only to planning and boundaries),
            # symmetric to the planning-phase vocab constraint
            for tid in (evi_id, evi_end_id, self.config.shift_token_id):
                logits[:, tid] = torch.finfo(logits.dtype).min
            if do_sample:
                probs = F.softmax(logits.float(), dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cum = torch.cumsum(sorted_probs, dim=-1)
                keep = (cum - sorted_probs) < top_p
                keep[:, 0] = True
                filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
                choice = torch.multinomial(filtered, 1).squeeze(-1)
                return sorted_idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
            return logits.argmax(dim=-1)  # [K]

        # ---------- Phase 2b: feed all K anchors in one forward (embeddings augmented with per-event aggregated features; anchors mutually invisible) ----------
        anchor_embeds = embed(torch.full((1, K), anchor_id, dtype=torch.long, device=device)).to(compute_dtype)
        for j in range(K):
            intervals = self.serial_evi_tgts[0][j] if j < len(self.serial_evi_tgts[0]) else [[0, 0]]
            anchor_embeds[0, j] = anchor_embeds[0, j] + self._aggregate_interval_feature(0, intervals)
        logits = _parallel_step(anchor_embeds, anchor_pos)
        next_tokens = _sample_tokens(logits)     # first content token of each branch

        # ---------- Phase 3: parallel decoding; branches terminate independently ----------
        branches = [[t.item()] for t in next_tokens]
        finished = [t.item() == eos_token_id for t in next_tokens]
        # Per-branch length cap. Deliberately NOT max_new_tokens // K: every loop iteration is one
        # forward that advances all K branches at once, so wall-clock scales with this cap alone and
        # not with K -- dividing by K threw away the whole point of parallel decoding and truncated
        # captions mid-sentence (at K=30 it left 34 tokens, while the median training caption is 75
        # and P95 is 281). The default covers P95 with margin; override via config.max_branch_tokens.
        per_branch_cap = getattr(self.config, 'max_branch_tokens', 320)
        round_idx = 0
        while not all(finished) and round_idx < per_branch_cap:
            token_embeds = embed(next_tokens.view(1, K)).to(compute_dtype)
            logits = _parallel_step(token_embeds, anchor_pos + 1 + round_idx)
            sampled = _sample_tokens(logits)
            for j in range(K):
                if finished[j]:
                    sampled[j] = eos_token_id    # finished branch: feed EOS as a placeholder, stop recording output
                else:
                    branches[j].append(sampled[j].item())
                    if sampled[j].item() == eos_token_id:
                        finished[j] = True
            next_tokens = sampled
            round_idx += 1

        # ---------- Assemble the flat output and the position-aligned tgt ----------
        serial_gen_len = seq.shape[1] - prompt_len
        serial_tokens = seq[0, prompt_len:].tolist()
        serial_tgts = list(self.tgt[0][:serial_gen_len])
        if deinterleave:
            # Readable mode (default): one record per event, "<G_j> caption_j", joined with
            # record_separator_id (a newline in practice; the caller owns the tokenizer).
            assert record_separator_id is not None, \
                "generate_parallel needs record_separator_id to delimit per-event records"
            flat_tokens, flat_tgts, j = [], [], 0
            for tok, tg in zip(serial_tokens, serial_tgts):
                if tok == evi_id and j < K:                      # event token: attach its own branch
                    if j > 0:
                        flat_tokens.append(record_separator_id)
                        flat_tgts.append([[0, 0]])
                    flat_tokens.append(tok)
                    flat_tgts.append(tg)
                    flat_tokens.extend(branches[j])
                    flat_tgts.extend([[[0, 0]]] * len(branches[j]))
                    j += 1
                elif tok == evi_end_id:
                    continue                                     # planning-end <S>: records are already delimited
                else:
                    flat_tokens.append(tok)                      # keep anything else the serial phase produced
                    flat_tgts.append(tg)
        else:
            # Raw mode: emit the prediction in physical generation order -- the serial segment
            # (including <S>), the K fed anchors, then the branch tokens round by round exactly
            # as the parallel steps produced them (branch 0..K-1 within each round; branches
            # that finished earlier simply stop contributing).
            flat_tokens = list(serial_tokens)
            flat_tgts = list(serial_tgts)
            for j in range(K):
                flat_tokens.append(anchor_id)
                flat_tgts.append(self.serial_evi_tgts[0][j] if j < len(self.serial_evi_tgts[0]) else [[0, 0]])
            max_rounds = max(len(b) for b in branches) if branches else 0
            for r in range(max_rounds):
                for j in range(K):
                    if r < len(branches[j]):
                        flat_tokens.append(branches[j][r])
                        flat_tgts.append([[0, 0]])
        self.tgt = [flat_tgts]
        full = torch.cat([
            input_ids[0].to(device),
            torch.tensor(flat_tokens, device=device, dtype=input_ids.dtype),
        ])
        return full.unsqueeze(0)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        audio_feature=None,
        # video_token_indices_list=None,
        # audio_token_indices_list=None,
        video_token_indices_list_src=None,
        audio_token_indices_list_src=None,
        # tgt_timestep_distances=None,
        video_token_indices_per_timestep=None,
        audio_token_indices_per_timestep=None,
        last_text_token_indices=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        # The evidence-related parameters above must appear in the signature, otherwise
        # transformers._validate_model_kwargs raises "not used by the model".
        # They are not passed to super() (the parent doesn't handle them); they are only written into model_inputs here for forward.
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            audio_feature=audio_feature,
            **kwargs,
        )

        # Qwen2-5-VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        # Detect whether this is the first generation step (generating-phase handling)
        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["audio_feature"] = None
        model_inputs["last_text_token_indices"] = last_text_token_indices

        if cache_position[0] == 0:
            batch_size = input_ids.shape[0]

            def _ensure_batch_dim(val):
                if val is None:
                    return None
                if isinstance(val, list) and len(val) == batch_size:
                    return val
                return [val]

            model_inputs["video_token_indices_list_src"] = _ensure_batch_dim(video_token_indices_list_src)
            model_inputs["audio_token_indices_list_src"] = _ensure_batch_dim(audio_token_indices_list_src)
            model_inputs["video_token_indices_per_timestep"] = _ensure_batch_dim(video_token_indices_per_timestep)
            model_inputs["audio_token_indices_per_timestep"] = _ensure_batch_dim(audio_token_indices_per_timestep)

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        vision_start_mask = input_ids == vision_start_token_id
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(input_ids)

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    if not isinstance(dict_to_expand[key], list):
                        raise TypeError(
                            f"Expected value for key '{key}' to be a list, but got {type(dict_to_expand[key])} instead."
                        )
                    tensor = torch.tensor(dict_to_expand[key])
                    lengths = list(video_nums)
                    tensor = _repeat_interleave_samples(tensor, lengths=lengths, repeat_times=expand_size)
                    dict_to_expand[key] = tensor.tolist()
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # input_ids is required for expanding visual inputs
        # If input_ids is unavailable, visual inputs will not be used; therefore, there is no need to expand visual inputs.
        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs
