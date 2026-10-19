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

import os
import logging
import pathlib
import re
import torch
import transformers
import json
import sys
from pathlib import Path
import numpy as np
import torch
import random

from torch.utils.data import DataLoader

# Share worker tensors via shm files to avoid "Too many open files"; must precede worker fork.
torch.multiprocessing.set_sharing_strategy("file_system")

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
from qwenvl.data.dataset import make_supervised_data_module
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, WhisperFeatureExtractor
from transformers import LogitsProcessor, LogitsProcessorList

from qwenvl.train.trainer import QwenVLTrainer

from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

from tqdm import tqdm
import torch.distributed as dist


def collate_fn(batch):
    return batch[0]


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def format_intervals(intervals):
    return [f"{subinterval[0]} - {subinterval[1]} " for subinterval in intervals]


def prompt_payload(prompt):
    """Return the description inside the prompt's trailing < ... >, or ""."""
    m = re.search(r"<\s*(.+?)\s*>\s*[\.\?]?\s*$", prompt, re.S)
    return m.group(1) if m else ""


def echo_overlap(pred, ref):
    """Content-word overlap F1 between two texts (echo criterion for best-of-N resampling)."""
    words = lambda s: set(w for w in re.findall(r"[a-z']+", s.lower()) if len(w) > 3)
    a, b = words(pred), words(ref)
    if not a or not b:
        return 0.0
    i = len(a & b)
    p, r = i / len(a), i / len(b)
    return 2 * p * r / (p + r) if p + r else 0.0


def is_degenerate(body):
    """True if the body is too short, mostly non-ASCII, or leaks the template / modality tags."""
    b = body.strip()
    if len(b.split()) < 6:
        return True
    non_ascii = sum(1 for ch in b if ord(ch) > 127) / max(1, len(b))
    return (non_ascii > 0.3 or "Analyze the" in b or "</audio>" in b or "<audio>" in b
            or "</video>" in b or "<video>" in b)


def trim_to_sentence(body, min_keep=0.5):
    """Drop the fragment after the last sentence end if at least min_keep of the text remains."""
    m = list(re.finditer(r'[.!?]["\u201d]?(?=\s|$)', body))
    if not m:
        return body
    cut = m[-1].end()
    return body[:cut] if cut >= min_keep * len(body) else body



def dedup_dvc_records(text, iou_thresh, drop_degenerate=False):
    """NMS over deinterleaved dvc records ("start - end caption" per line).

    A record is dropped if its IoU with a kept record is >= iou_thresh, or it repeats a kept
    caption with IoU > 0.1. Text with fewer than two records is returned unchanged.
    """
    head = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(.*)$")
    records = []  # [start, end, caption, raw_lines]
    for line in text.split("\n"):
        m = head.match(line)
        if m:
            records.append([float(m.group(1)), float(m.group(2)), m.group(3), line])
        elif records:
            records[-1][2] += " " + line.strip()
            records[-1][3] += "\n" + line
        else:
            return text  # leading non-record text: not a deinterleaved dvc output
    if len(records) < 2:
        return text
    kept = []  # (start, end, normalized_caption, raw_lines)
    for s, e, cap, raw in records:
        if drop_degenerate and is_degenerate(cap):
            continue
        ncap = " ".join(cap.split()).lower()
        redundant = False
        for ks, ke, kcap, _ in kept:
            inter = max(0.0, min(e, ke) - max(s, ks))
            union = max(e, ke) - min(s, ks)
            iou = inter / union if union > 0 else 1.0
            if iou >= iou_thresh or (ncap and ncap == kcap and iou > 0.1):
                redundant = True
                break
        if not redundant:
            kept.append((s, e, ncap, raw))
    return "\n".join(r[3] for r in kept)


def timestep_intervals_to_seconds(intervals, timestep_duration, round_digits=2):
    """Convert timestep-index intervals [[a, b], ...] to seconds."""
    if timestep_duration is None or timestep_duration <= 0:
        return intervals
    out = []
    for a, b in intervals:
        start_sec = a * timestep_duration
        end_sec = (b + 1) * timestep_duration  # half-open interval: end time of step b
        out.append([round(start_sec, round_digits), round(end_sec, round_digits)])
    return out 

def apply_liger_kernel_to_qwen2_5_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2.5-VL models.
    NOTE: Qwen2.5-VL is not available in transformers<4.48.2

    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """

    print("Applying Liger kernels to Qwen2.5-VL model...")

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from qwenvl.model import modeling_qwen2_5_vl

    if rope:
        modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
    if rms_norm:
        modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
    if swiglu:
        modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        model.visual.requires_grad_(True)
    else:
        model.visual.requires_grad_(False)

    if model_args.tune_mm_mlp:
        model.visual.merger.requires_grad_(True)
    else:
        model.visual.merger.requires_grad_(False)

    if model_args.tune_mm_audio:
        model.audio.requires_grad_(True)
    else:
        model.audio.requires_grad_(False)

    if model_args.tune_mm_qformer:
        model.audio.qformer.requires_grad_(True)
        model.audio.q_tokens.requires_grad_(True)
        model.audio.audio_proj.requires_grad_(True)
    else:
        model.audio.qformer.requires_grad_(False)
        model.audio.q_tokens.requires_grad_(False)
        model.audio.audio_proj.requires_grad_(False)

    if model_args.tune_mm_llm:
        if model_args.use_lora:
            raise Exception("tune_mm_llm is not supported when use_lora is True")
        model.model.requires_grad_(True)
        model.lm_head.requires_grad_(True)
    else:
        model.model.requires_grad_(False)
        model.lm_head.requires_grad_(False)

    # Grounding heads are always trainable
    if hasattr(model, 'frm_head'):
        model.frm_head.requires_grad_(True)
    if hasattr(model, 'evi_head'):
        model.evi_head.requires_grad_(True)
    for hname in ('bnd_frm_head', 'bnd_evi_head'):   # boundary heads: trainable only when their loss is on
        if hasattr(model, hname):
            getattr(model, hname).requires_grad_(float(getattr(model.config, 'loss_boundary_weight', 0.0) or 0.0) > 0)


def print_trainable_parameters_summary(
    model, 
    num_added_tokens=0, 
    new_token_ids=None, 
    original_tokenizer_size=None,
    rank=0,
    enabled=True
):
    """Print trainable parameters and the training status of the new token embeddings (rank 0 only)."""
    if not enabled or rank != 0:
        return
    
    print("\n" + "=" * 80)
    print("Trainable Parameters Summary")
    print("=" * 80)
    
    trainable_params = []
    total_trainable = 0
    embedding_trainable = False
    new_token_embedding_trainable = {}
    
    for k, v in model.named_parameters():
        if v.requires_grad:
            trainable_params.append((k, v.shape, v.numel()))
            total_trainable += v.numel()
            
            if "embed" in k.lower() or "lm_head" in k.lower():
                embedding_trainable = True
                print(f"  [EMBEDDING] {k}: {v.shape} ({v.numel():,} params)")
                
                # Keyed on new_token_ids: on resume num_added_tokens is 0
                if new_token_ids:
                    if "embed_tokens" in k or "lm_head" in k:
                        for tid in new_token_ids:
                            if tid < v.shape[0]:
                                if tid not in new_token_embedding_trainable:
                                    new_token_embedding_trainable[tid] = {}
                                new_token_embedding_trainable[tid][k] = v.requires_grad
    
    print(f"\nTotal trainable parameters: {total_trainable:,}")
    print(f"\nAll trainable parameters:")
    for k, shape, numel in trainable_params:
        if "embed" not in k.lower() and "lm_head" not in k.lower():
            print(f"  {k}: {shape} ({numel:,} params)")
    
    if new_token_ids:
        print(f"\n" + "-" * 80)
        print("New Token Embedding Training Status:")
        print("-" * 80)
        
        # Embeddings may live under base_model when LoRA is used
        if hasattr(model, 'get_input_embeddings'):
            input_emb = model.get_input_embeddings()
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'get_input_embeddings'):
            input_emb = model.base_model.get_input_embeddings()
        else:
            input_emb = None
        
        if hasattr(model, 'get_output_embeddings'):
            output_emb = model.get_output_embeddings()
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'get_output_embeddings'):
            output_emb = model.base_model.get_output_embeddings()
        else:
            output_emb = None
        
        if input_emb is None or output_emb is None:
            print("  ⚠️  Cannot access embedding modules")
            print("=" * 80 + "\n")
            return
        
        input_emb_module_trainable = input_emb.weight.requires_grad
        output_emb_module_trainable = output_emb.weight.requires_grad
        
        print(f"  Input embedding module (model.model.embed_tokens):")
        print(f"    Overall requires_grad: {input_emb_module_trainable}")
        model_params = list(model.model.parameters()) if hasattr(model, 'model') else []
        if len(model_params) > 0:
            parent_trainable = model_params[0].requires_grad
            print(f"    Module parent (model.model) requires_grad: {parent_trainable}")
        else:
            print(f"    Module parent (model.model) requires_grad: N/A")
        
        print(f"  Output embedding module (model.lm_head):")
        print(f"    Overall requires_grad: {output_emb_module_trainable}")
        
        all_trainable = True
        for tid in new_token_ids:
            if tid < len(input_emb.weight) and tid < len(output_emb.weight):
                # With a gradient hook requires_grad is True but old-token gradients are masked
                input_trainable = input_emb.weight.requires_grad
                output_trainable = output_emb.weight.requires_grad
                
                has_input_hook = len(input_emb.weight._backward_hooks) > 0 if hasattr(input_emb.weight, '_backward_hooks') else False
                has_output_hook = len(output_emb.weight._backward_hooks) > 0 if hasattr(output_emb.weight, '_backward_hooks') else False
                
                status = "✓ TRAINABLE" if (input_trainable or output_trainable) else "✗ FROZEN"
                print(f"\n  Token ID {tid}:")
                print(f"    Input embedding:  {'✓ trainable' + (' (with selective hook)' if has_input_hook else '') if input_trainable else '✗ frozen'}")
                print(f"    Output embedding: {'✓ trainable' + (' (with selective hook)' if has_output_hook else '') if output_trainable else '✗ frozen'}")
                print(f"    Status: {status}")
                
                if not (input_trainable or output_trainable):
                    all_trainable = False
            else:
                print(f"\n  Token ID {tid}: ⚠️  ID out of range!")
                all_trainable = False
        
        print(f"\n" + "-" * 80)
        if all_trainable:
            print(f"✓ All new token embeddings are trainable!")
            if input_emb.weight.requires_grad:
                print(f"  Using gradient hooks to selectively train only new tokens")
                if original_tokenizer_size is not None:
                    print(f"  Old tokens (0-{original_tokenizer_size-1}) will be frozen via gradient masking")
            print(f"  New tokens will be updated during training.")
        else:
            print(f"⚠️  WARNING: New token embeddings are NOT trainable!")
            print(f"  They will NOT be updated during training.")
            print(f"  Reason: The embedding modules are frozen (requires_grad=False)")
            print(f"  Solution: The code should have set embedding.requires_grad_(True)")
    else:
        print(f"\nNo new tokens added, skipping embedding check")
    
    print("=" * 80 + "\n")


def resize_and_initialize_embeddings(
    model,
    tokenizer,
    special_tokens: list = None,
):
    """Add special tokens, pad the tokenizer to the embedding size, and mean-initialize the new embeddings.

    Returns:
        (original_tokenizer_size, num_added_tokens, new_token_ids)
    """
    original_tokenizer_size = len(tokenizer)
    model_embedding_size = len(model.get_input_embeddings().weight)
    
    # Add special tokens
    num_added_tokens = 0
    new_token_ids = []
    added_desc = []
    
    if special_tokens is not None and len(special_tokens) > 0:
        # Config attribute names are read by the model and checkpoint_utils.py; keep in sync
        token_to_attr_name = {
            "<G>": "evi_token_id",
            "<S>": "evi_end_token_id",
            "<shift>": "shift_token_id",
        }
        
        for token in special_tokens:
            special_tokens_dict = {"additional_special_tokens": [token]}
            num_added = tokenizer.add_special_tokens(special_tokens_dict)
            if num_added > 0:
                num_added_tokens += num_added
                token_id = tokenizer.convert_tokens_to_ids(token)
                new_token_ids.append(token_id)
                added_desc.append(f"added {token} with ID {token_id}")
            else:
                # Token already exists
                token_id = tokenizer.convert_tokens_to_ids(token)
                new_token_ids.append(token_id)
                added_desc.append(f"{token} already present with ID {token_id}")
        print(f"Adding {len(special_tokens)} special tokens ({', '.join(added_desc)})")
    
    # Pad the tokenizer up to the embedding size with reserved tokens
    current_tokenizer_size = len(tokenizer)
    
    if model_embedding_size > current_tokenizer_size:
        need = model_embedding_size - current_tokenizer_size
        new_tokens = [f"<reserved_{i}>" for i in range(need)]
        tokenizer.add_tokens(new_tokens, special_tokens=False)
        
    # Write token ids into config
    if special_tokens is not None and len(special_tokens) > 0:
        for token in special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                if token in token_to_attr_name:
                    attr_name = token_to_attr_name[token]
                    setattr(model.config, attr_name, token_id)
                else:
                    attr_name = token.strip('<>').replace('/', '_') + '_token_id'
                    setattr(model.config, attr_name, token_id)

    current_tokenizer_size = len(tokenizer)
    
    # Resize embeddings only if a new token id is out of range
    if num_added_tokens > 0:
        final_token_ids = []
        if special_tokens is not None:
            for token in special_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    final_token_ids.append(token_id)
        
        max_final_token_id = max(final_token_ids) if final_token_ids else -1
        
        if max_final_token_id >= model_embedding_size:
            new_vocab_size = max(current_tokenizer_size, max_final_token_id + 1)
            model.resize_token_embeddings(new_vocab_size)
            
            # Mean-initialize the new rows
            with torch.no_grad():
                input_emb = model.get_input_embeddings().weight.data
                output_emb = model.get_output_embeddings().weight.data
                
                input_mean = input_emb[:model_embedding_size].mean(dim=0)
                output_mean = output_emb[:model_embedding_size].mean(dim=0)

                num_new_embeddings = new_vocab_size - model_embedding_size
                input_emb[model_embedding_size:] = input_mean.unsqueeze(0).expand(num_new_embeddings, -1)
                output_emb[model_embedding_size:] = output_mean.unsqueeze(0).expand(num_new_embeddings, -1)
            
            # Mean-initialize the special tokens
            if special_tokens is not None and len(special_tokens) > 0:
                with torch.no_grad():
                    input_emb = model.get_input_embeddings().weight.data
                    output_emb = model.get_output_embeddings().weight.data
                    
                    # Initialize with the mean of embeddings within the original tokenizer range
                    input_mean = input_emb[:model_embedding_size].mean(dim=0)
                    output_mean = output_emb[:model_embedding_size].mean(dim=0)
                    
                    for token in special_tokens:
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        if token_id != tokenizer.unk_token_id and token_id < len(input_emb) and token_id < len(output_emb):
                            input_emb[token_id] = input_mean.clone()
                            output_emb[token_id] = output_mean.clone()
        else:
            # No resize needed: still mean-initialize the special tokens
            if special_tokens is not None and len(special_tokens) > 0:
                with torch.no_grad():
                    input_emb = model.get_input_embeddings().weight.data
                    output_emb = model.get_output_embeddings().weight.data
                    
                    # Initialize with the mean of embeddings within the original tokenizer range
                    input_mean = input_emb[:model_embedding_size].mean(dim=0)
                    output_mean = output_emb[:model_embedding_size].mean(dim=0)
                    
                    for token in special_tokens:
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        if token_id != tokenizer.unk_token_id and token_id < len(input_emb) and token_id < len(output_emb):
                            input_emb[token_id] = input_mean.clone()
                            output_emb[token_id] = output_mean.clone()

    return original_tokenizer_size, num_added_tokens, new_token_ids


def setup_new_token_embeddings_for_training(
    model,
    num_added_tokens: int,
    new_token_ids: list,
    rank: int = 0,
):
    """Make only the new token rows of the input/output embeddings trainable (call after LoRA)."""
    if new_token_ids:
        input_emb = model.get_input_embeddings()
        output_emb = model.get_output_embeddings()
        
        if input_emb is not None and output_emb is not None:
            input_emb.weight.requires_grad_(True)
            output_emb.weight.requires_grad_(True)
            
            def create_selective_grad_hook(trainable_token_ids):
                """Gradient hook that zeroes all rows except trainable_token_ids."""
                trainable_set = set(trainable_token_ids)
                def hook(grad):
                    if grad is not None:
                        vocab_size = grad.shape[0]
                        mask = torch.zeros(vocab_size, device=grad.device, dtype=grad.dtype)
                        for tid in trainable_set:
                            if tid < vocab_size:
                                mask[tid] = 1.0
                        # [vocab_size, hidden_size]
                        mask = mask.view(-1, 1).expand_as(grad)
                        return grad * mask
                    return grad
                return hook
            
            input_emb.weight.register_hook(
                create_selective_grad_hook(new_token_ids)
            )
            output_emb.weight.register_hook(
                create_selective_grad_hook(new_token_ids)
            )


def train(attn_implementation="flash_attention_2"):
    global local_rank

    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if unknown_args and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        # Unknown flags are reported and ignored
        print(f"[args] ignoring unknown arguments: {unknown_args}")

    training_args.remove_unused_columns = False

    apply_liger_kernel_to_qwen2_5_vl()

    local_rank = int(os.environ.get("LOCAL_RANK", training_args.local_rank))
    if torch.cuda.is_available() and local_rank is not None and local_rank >= 0:
        torch.cuda.set_device(local_rank)
    os.makedirs(training_args.output_dir, exist_ok=True)

    data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(
        model_args.model_base,
    )
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size, 
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )
    data_args.model_type = "qwen2.5vl"

    try:
        rank = dist.get_rank() if dist.is_initialized() else 0
    except:
        rank = 0
    
    if not data_args.run_test:  # training
        num_added_tokens = 0
        new_token_ids = []
        original_tokenizer_size = None

        if model_args.use_evidence and os.path.isfile(os.path.join(training_args.output_dir, "tokenizer_config.json")):
            tokenizer_path = training_args.output_dir
        else:
            tokenizer_path = model_args.model_name_or_path if model_args.use_evidence and model_args.model_name_or_path and os.path.isdir(model_args.model_name_or_path) else model_args.model_base
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_args=model_args)
        if dist.is_initialized():
            barrier_device_ids = [local_rank] if local_rank is not None and local_rank >= 0 else None
            if barrier_device_ids is not None:
                dist.barrier(device_ids=barrier_device_ids)
            else:
                dist.barrier()
        
        if model_args.use_flashattention:
            attn_impl = "flash_attention_2"
        else:
            attn_impl = "sdpa"

        model = video_SALMONN2_plus.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_impl,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        
        model.config.use_cache = False
        model.config.use_flashattention = model_args.use_flashattention
        model.config.use_evidence = model_args.use_evidence
        if model_args.use_evidence:
            model.config.loss_match_weight = model_args.loss_match_weight
            model.config.loss_stop_weight = model_args.loss_stop_weight
            model.config.loss_stop_symmetric = model_args.loss_stop_symmetric
            model.config.branch_interval_attention = model_args.branch_interval_attention
            model.config.branch_interval_margin = model_args.branch_interval_margin
            model.config.branch_interval_audio_full = model_args.branch_interval_audio_full
            model.config.loss_contrast_weight = model_args.loss_contrast_weight
            model.config.loss_contrast_tau = model_args.loss_contrast_tau
            model.config.loss_contrast_detach = model_args.loss_contrast_detach
            model.config.grounding_target_decay = model_args.grounding_target_decay
            model.config.loss_iou_weight = model_args.loss_iou_weight
            model.config.grounding_coverage_train = model_args.grounding_coverage_train
            model.config.loss_boundary_weight = model_args.loss_boundary_weight
            model.config.grounding_boundary_decay = model_args.grounding_boundary_decay
            model.config.loss_plateau_weight = model_args.loss_plateau_weight
            model.config.grounding_plateau_ratio = model_args.grounding_plateau_ratio
            model.config.loss_boundary_rank_weight = model_args.loss_boundary_rank_weight
            model.config.grounding_boundary_rank_margin = model_args.grounding_boundary_rank_margin
            if (model_args.loss_plateau_weight or 0) > 0 or (model_args.loss_boundary_rank_weight or 0) > 0:
                rank0_print(f"[train] plateau/rank losses: plateau {model_args.loss_plateau_weight} (ratio {model_args.grounding_plateau_ratio}), "
                            f"boundary_rank {model_args.loss_boundary_rank_weight} (margin {model_args.grounding_boundary_rank_margin})")
            model.config.grounding_boundary_heads = bool((model_args.loss_boundary_weight or 0.0) > 0)
            if model.config.grounding_boundary_heads:
                rank0_print(f"[train] boundary heads on: loss_boundary_weight {model_args.loss_boundary_weight}, "
                            f"decay {model_args.grounding_boundary_decay}")
            model.config.self_aggregation_prob = model_args.self_aggregation_prob
            # Readout settings for self-aggregation; must match inference
            model.config.grounding_threshold = model_args.grounding_threshold
            model.config.grounding_merge_gap = model_args.grounding_merge_gap
            model.config.grounding_coverage_mask = model_args.grounding_coverage_mask
            model.config.audio_mean_weight = model_args.audio_mean_weight
            model.config.grounding_snap_window = model_args.grounding_snap_window
            model.config.use_attention_head = model_args.use_attention_head
            model.config.save_gpu_trick = model_args.save_gpu_trick
        
            model.config.vision_audio_layer_index = model_args.vision_audio_layer_index
            original_tokenizer_size, num_added_tokens, new_token_ids = resize_and_initialize_embeddings(
                model=model, tokenizer=tokenizer, special_tokens=["<G>", "<S>", "<shift>"])
        else:
            original_tokenizer_size = len(tokenizer)

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()  
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            if "3" not in training_args.deepspeed:
                if training_args.gradient_checkpointing_kwargs is None:
                    training_args.gradient_checkpointing_kwargs={"use_reentrant": False}
                else:
                    training_args.gradient_checkpointing_kwargs["use_reentrant"] = False

        if model_args.lora_ckpt != "No":
            from peft import PeftModel
            audio_layers = model.audio.layers
            del model.audio.layers
            model = PeftModel.from_pretrained(model, model_args.lora_ckpt)
            model.model.audio.layers = audio_layers
            model = model.merge_and_unload()

        set_model(model_args, model)

        if training_args.no_audio:
            del model.audio

        if model_args.use_lora:
            from peft import LoraConfig, get_peft_model
            module_to_save = [] # fully trained modules besides LoRA
            if model_args.tune_mm_vision:
                module_to_save.append("visual")
            if model_args.tune_mm_mlp:
                module_to_save.append("visual.merger")
            if model_args.tune_mm_audio:
                module_to_save.append("audio")
            if model_args.tune_mm_qformer:  # qformer lives inside the audio module
                module_to_save.append("audio.qformer")
                module_to_save.append("audio.q_tokens")
                module_to_save.append("audio.audio_proj")
            # Grounding heads are set trainable after LoRA, not via modules_to_save (avoids duplicates).
            # Targets are anchored to model.layers.N so Whisper and ViT projections are not wrapped.
            LORA_TARGETS = {
                "qkv": r"^model\.layers\.\d+\.self_attn\.(q|k|v)_proj$",
                "all": r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$",
            }
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=LORA_TARGETS[model_args.lora_target],
                lora_dropout=model_args.lora_dropout,
                bias=model_args.lora_bias,
                task_type="CAUSAL_LM",
                modules_to_save=module_to_save,
            )
            if not training_args.no_audio:
                audio_layers = model.audio.layers
                del model.audio.layers
            model = get_peft_model(model, lora_config)
            if not training_args.no_audio:
                model.model.audio.layers = audio_layers

            for k, v in model.named_parameters():
                if "lora" in k:
                    v.requires_grad_(True)
        
            if model_args.use_evidence:
                # Head location depends on how LoRA wraps the model
                if hasattr(model, 'frm_head'):
                    model.frm_head.requires_grad_(True)
                elif hasattr(model, 'base_model') and hasattr(model.base_model, 'frm_head'):
                    model.base_model.frm_head.requires_grad_(True)
                elif hasattr(model, 'model') and hasattr(model.model, 'frm_head'):
                    model.model.frm_head.requires_grad_(True)
                
                if hasattr(model, 'evi_head'):
                    model.evi_head.requires_grad_(True)
                elif hasattr(model, 'base_model') and hasattr(model.base_model, 'evi_head'):
                    model.base_model.evi_head.requires_grad_(True)
                elif hasattr(model, 'model') and hasattr(model.model, 'evi_head'):
                    model.model.evi_head.requires_grad_(True)

                if (getattr(model_args, 'loss_boundary_weight', 0.0) or 0.0) > 0:
                    for m in (model, getattr(model, 'base_model', None), getattr(model, 'model', None)):
                        if m is not None and hasattr(m, 'bnd_frm_head'):
                            m.bnd_frm_head.requires_grad_(True); m.bnd_evi_head.requires_grad_(True); break
        
        # Make the new token embeddings trainable
        if model_args.use_evidence and not getattr(model_args, 'readout_only', False):
            setup_new_token_embeddings_for_training(model=model, num_added_tokens=num_added_tokens, new_token_ids=new_token_ids, rank=dist.get_rank())
        if getattr(model_args, 'readout_only', False):
            # readout_only: freeze embeddings and lm_head; only the grounding heads train
            for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
                if emb is not None:
                    emb.weight.requires_grad_(False)
            rank0_print("[train] readout_only: embeddings / lm_head frozen; trainable = "
                        + ", ".join(sorted({n.split('.')[0] for n, p in model.named_parameters() if p.requires_grad})))
        
        print_trainable_parameters_summary(
            model=model,
            num_added_tokens=num_added_tokens,
            new_token_ids=new_token_ids,
            original_tokenizer_size=original_tokenizer_size,
            rank=dist.get_rank(),
            enabled=training_args.print_trainable_params
        )
        trainer = QwenVLTrainer(
            model=model, processing_class=tokenizer, args=training_args, **data_module
        )

        class _NonFiniteGuard(transformers.TrainerCallback):
            """Abort at the first non-finite or zero loss / grad_norm (nan_guard=True)."""

            def on_log(self, args, state, control, logs=None, **kwargs):
                import math
                bad = [k for k in ("loss", "grad_norm", "loss_match")
                       if k in logs and isinstance(logs[k], float) and not math.isfinite(logs[k])]
                if bad or (isinstance(logs.get("loss"), float) and logs["loss"] == 0.0 and state.global_step > 0):
                    msg = f"[nan_guard] non-finite/zero training signal at step {state.global_step}: " \
                          f"{ {k: logs.get(k) for k in ('loss', 'grad_norm', 'loss_match')} } -- aborting"
                    print(msg, flush=True)
                    raise RuntimeError(msg)

        if getattr(model_args, "nan_guard", False):
            trainer.add_callback(_NonFiniteGuard())
            rank0_print("[train] nan_guard on: training aborts at the first non-finite loss/grad_norm")
        
        def _ckpt_step(p):
            """Step of a checkpoint-<digits> dir, else None."""
            tail = p.name.split("-")[-1]
            return int(tail) if tail.isdigit() and p.is_dir() else None

        resume_ckpts = sorted(
            [p for p in pathlib.Path(training_args.output_dir).glob("checkpoint-*") if _ckpt_step(p) is not None],
            key=_ckpt_step)
        resume_meta_path = os.path.join(training_args.output_dir, "resume_meta.json")
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        current_meta = {
            "model_name_or_path": model_args.model_name_or_path,
            "lora_ckpt": model_args.lora_ckpt,
            "world_size": world_size,
            # Architecture config not stored in checkpoints; restored by convert_checkpoint_to_model
            "model_config": {
                "grounding_boundary_heads": bool((model_args.loss_boundary_weight or 0.0) > 0),
            },
        }
        if resume_ckpts and os.path.isfile(resume_meta_path):
            with open(resume_meta_path) as f:
                saved_meta = json.load(f)
            for meta_key in ("model_name_or_path", "lora_ckpt"):
                if saved_meta.get(meta_key) != current_meta[meta_key]:
                    raise ValueError(
                        f"Resume consistency check failed: {meta_key} changed from {saved_meta.get(meta_key)!r} "
                        f"to {current_meta[meta_key]!r}. Checkpoints only contain trainable weights; the frozen "
                        f"backbone is rebuilt from these paths, so they must match the original run. "
                        f"To start a new training stage from a trained model, use a new output_dir (RUN_NAME) instead.")
            if saved_meta.get("world_size") != world_size:
                logging.warning(
                    f"world_size changed from {saved_meta.get('world_size')} to {world_size}; "
                    f"optimizer states cannot be loaded and will be reset (model weights only).")
        if not dist.is_initialized() or dist.get_rank() == 0:
            # Atomic write
            tmp_meta_path = resume_meta_path + ".tmp"
            with open(tmp_meta_path, "w") as f:
                json.dump(current_meta, f, indent=2)
            os.replace(tmp_meta_path, resume_meta_path)

        if resume_ckpts:
            logging.info("checkpoint found, resume training")
            latest_checkpoint = str(resume_ckpts[-1])
            resumed_step = None
            ckpt_state_file = os.path.join(latest_checkpoint, "trainer_state.json")
            if os.path.isfile(ckpt_state_file):
                with open(ckpt_state_file) as f:
                    resumed_step = json.load(f).get("global_step")
            did_weight_only_fallback = False
            try:
                trainer.train(resume_from_checkpoint=True)
            except Exception as e:
                error_str = str(e)
                # On world-size mismatch, load model weights only and reset optimizer states
                is_world_size_error = (
                    "world size" in error_str.lower() or
                    "zeroruntimeexception" in error_str.lower() or
                    "dp world size" in error_str.lower() or
                    "automatic adjustment" in error_str.lower()
                )
                if is_world_size_error:
                    logging.warning(f"Checkpoint world size mismatch detected: {e}")
                    logging.warning("Loading model weights only (optimizer states will be reset)")
                    model_to_load = getattr(trainer.model, "module", trainer.model)
                    
                    with open(os.path.join(latest_checkpoint, "latest")) as f:
                        ds_tag = f.read().strip()
                    ds_model_states_path = os.path.join(latest_checkpoint, ds_tag, "mp_rank_00_model_states.pt")
                    checkpoint_state = torch.load(ds_model_states_path, map_location="cpu", weights_only=False)["module"]
                    
                    load_result = model_to_load.load_state_dict(checkpoint_state, strict=False)
                    if load_result.unexpected_keys:
                        raise RuntimeError(
                            f"Checkpoint keys do not match model structure, e.g. {load_result.unexpected_keys[:5]}. "
                            f"Refusing to continue with a partially loaded model.")
                    logging.info(f"Loaded {len(checkpoint_state)} trainable tensors from {ds_model_states_path}")
     
                    did_weight_only_fallback = True
                    trainer.train(resume_from_checkpoint=False)
                else:
                    raise
            if not did_weight_only_fallback and resumed_step is not None and trainer.state.global_step <= resumed_step:
                logging.warning(
                    f"Resume did NOT train any step: checkpoint is already at global_step={resumed_step} with "
                    f"max_steps={trainer.state.max_steps}. To train more epochs, increase --epoch; or start a new "
                    f"stage from the merged model in output_dir with a new RUN_NAME (--model <output_dir> --lora_ckpt No).")
        else:
            trainer.train()
        
        # Save: trainer state, preprocessor/tokenizer, then the merged model
        if dist.get_rank() == 0:
            logging.info(f"Training finished. Total steps = {trainer.state.global_step}, max_steps = {trainer.state.max_steps}")
        trainer.save_state()  


        is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        if is_rank0:
            data_args.image_processor.save_pretrained(training_args.output_dir)
            if model_args.use_evidence:
                tokenizer.save_pretrained(training_args.output_dir)
                print(f"✓ Saved tokenizer to {training_args.output_dir}")

        model.config.use_cache = True

        # use_evidence: save the full merged model
        if model_args.use_evidence:
            zero_stage = None
            if getattr(trainer, "is_deepspeed_enabled", False):
                ds_plugin = getattr(trainer.accelerator.state, "deepspeed_plugin", None)
                if ds_plugin is not None:
                    zero_stage = ds_plugin.zero_stage
            if zero_stage == 3:
                raise RuntimeError(
                    "Saving the merged full model under ZeRO-3 is not supported: weights are sharded across ranks "
                    "and rank0's state_dict is incomplete. Use --deepspeed 2 instead.")
            if is_rank0:
                from peft import PeftModel
                unwrapped = getattr(trainer.model, "module", trainer.model)
                if isinstance(unwrapped, PeftModel):
                    unwrapped = unwrapped.merge_and_unload()  # keys match the inference model
                unwrapped.config.save_pretrained(training_args.output_dir)
                full_state = unwrapped.state_dict()
                cpu_state = {k: v.cpu() for k, v in full_state.items()}
                if getattr(unwrapped.config, "tie_word_embeddings", False):
                    emb = unwrapped.get_input_embeddings()
                    head = unwrapped.get_output_embeddings()
                    if emb is not None and head is not None and head.weight.data_ptr() == emb.weight.data_ptr():
                        cpu_state.pop("lm_head.weight", None)
                torch.save(cpu_state, os.path.join(training_args.output_dir, "pytorch_model.bin"))
                logging.info(f"Saved merged full model to {training_args.output_dir}/pytorch_model.bin for inference")
        else:
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    else:   # testing
        
        # Resolve the model directory (auto-converting checkpoints)
        def _dir_has_file(d, name):
            return bool(d) and os.path.isdir(d) and os.path.isfile(os.path.join(d, name))

        from qwenvl.train.checkpoint_utils import resolve_test_model_path
        model_load_path = resolve_test_model_path(
            model_args.model_name_or_path, training_args.output_dir,
            auto_convert=data_args.auto_convert_checkpoint,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            logging.info(f"[test] resolved model path: {model_load_path}")

        if model_args.use_evidence and _dir_has_file(model_load_path, "tokenizer_config.json"):
            tokenizer_path = model_load_path
        elif model_args.use_evidence and _dir_has_file(training_args.output_dir, "tokenizer_config.json"):
            tokenizer_path = training_args.output_dir
        else:
            tokenizer_path = model_args.model_base
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        pred_rank = training_args.pred_rank
        if torch.cuda.device_count() > 1:
            pred_rank = pred_rank * torch.cuda.device_count() + torch.cuda.current_device()
            data_args.dataset_use = f"dataset/{pred_rank}.json"
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_args=model_args)
        os.makedirs(os.path.join(training_args.output_dir, training_args.run_name), exist_ok=True)

        if torch.cuda.device_count() > 1:
            ds_config = {
                "fp16": {"enabled": False},
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": 3
                },
                "train_micro_batch_size_per_gpu": 1,
            }
        
        if model_args.lora_ckpt == "No" or model_args.use_evidence:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logging.info(f"loading trained model from {model_load_path}")
            model = video_SALMONN2_plus.from_pretrained(
                model_load_path,
                attn_implementation="sdpa",
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        else:
            # Upstream video-SALMONN-2 path: merge a separate LoRA
            attn_impl = "flash_attention_2" if model_args.use_flashattention else "sdpa"
            merged_dir = os.path.join(
                training_args.output_dir,
                "generation" if torch.cuda.device_count() > 1 else f"generation_{pred_rank}",
            )
            if dist.get_rank() == 0:
                model = video_SALMONN2_plus.from_pretrained(
                    model_args.model_name_or_path,
                    attn_implementation=attn_impl,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    device_map="cpu"
                )
                from peft import PeftModel
                if not training_args.no_audio:
                    audio_layers = model.audio.layers
                    del model.audio.layers
                model = PeftModel.from_pretrained(model, model_args.lora_ckpt)
                if not training_args.no_audio:
                    model.model.audio.layers = audio_layers
                model = model.merge_and_unload()
                model.config.use_evidence = model_args.use_evidence
                model.save_pretrained(merged_dir)
            if dist.is_initialized():
                barrier_device_ids = [local_rank] if local_rank is not None and local_rank >= 0 else None
                if barrier_device_ids is not None:
                    dist.barrier(device_ids=barrier_device_ids)
                else:
                    dist.barrier()
            model = video_SALMONN2_plus.from_pretrained(
                merged_dir,
                attn_implementation=attn_impl,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )

        if training_args.no_audio:
            del model.audio

        model.config.grounding_threshold = model_args.grounding_threshold
        model.config.grounding_merge_gap = model_args.grounding_merge_gap
        model.config.grounding_snap_window = model_args.grounding_snap_window
        if model_args.grounding_snap_window > 0:
            if not getattr(model.config, 'grounding_boundary_heads', False):
                raise ValueError("grounding_snap_window > 0 but this model was not trained with boundary heads "
                                 "(config.grounding_boundary_heads is not set)")
            rank0_print(f"[test] boundary snapping on: window {model_args.grounding_snap_window} steps, "
                        f"tasks {model_args.grounding_snap_tasks or 'all'}")
        snap_tasks = {t.strip() for t in (model_args.grounding_snap_tasks or "").split(",") if t.strip()}
        model.config.max_planning_events = model_args.max_planning_events
        model.config.grounding_coverage_mask = model_args.grounding_coverage_mask
        model.config.branch_interval_attention = model_args.branch_interval_attention
        model.config.branch_interval_margin = model_args.branch_interval_margin
        model.config.branch_interval_audio_full = model_args.branch_interval_audio_full

        # Per-task overrides, applied per sample below
        task_audio_weight = {}
        if model_args.task_audio_mean_weight:
            for item in model_args.task_audio_mean_weight.split(","):
                if item.strip():
                    t, w = item.split("=")
                    task_audio_weight[t.strip()] = float(w)
            print(f"[test] per-task audio_mean_weight override: {task_audio_weight}")
        default_audio_weight = getattr(model.config, "audio_mean_weight", 1.0)
        print(f"[test] audio weight (readout / anchor injection): {default_audio_weight}")
        task_body_cap = {}
        if model_args.task_max_body_tokens:
            for item in model_args.task_max_body_tokens.split(","):
                if item.strip():
                    t, n = item.split("=")
                    task_body_cap[t.strip()] = int(n)
            print(f"[test] per-task max body tokens: {task_body_cap}")

        if torch.cuda.device_count() > 1:
            import deepspeed
            ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
            ds_engine.module.eval()
            model = ds_engine.module
        else:
            model.cuda()

        result = []
        # Each sample is appended to a .partial.jsonl; a rerun skips samples already there
        def _to_jsonable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(v) for v in obj]
            return obj
        _run_dir = os.path.join(training_args.output_dir, training_args.run_name)
        os.makedirs(_run_dir, exist_ok=True)
        _partial = os.path.join(_run_dir, f"test_results_rank{pred_rank}.partial.jsonl")
        _done_ids = set()
        if os.path.isfile(_partial):
            with open(_partial) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r0 = json.loads(line)
                    except Exception:
                        continue            # truncated line
                    result.append(r0)
                    v0 = r0.get("video_id"); _done_ids.add(str(v0[0] if isinstance(v0, list) else v0))
            print(f"[test] resume: {len(result)} samples already in {os.path.basename(_partial)}, skipping them")
        _partial_f = open(_partial, "a")
        test_data = data_module["train_dataset"]
        loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=training_args.dataloader_num_workers,
            collate_fn=collate_fn,
            in_order=False
        )
        for inputs in tqdm(loader, desc=f"RANK {pred_rank}"):
            if inputs:
                res_i = {
                    "video": inputs.pop("video", None),
                    "image": inputs.pop("image", None),
                    "prompt": inputs.pop("prompt", None),
                    "ref": inputs.pop("ref", None),
                    "audio": inputs.pop("audio", None),
                    "use_audio": inputs.pop("use_audio", False),
                    "should_use": inputs.pop("should_use", True),
                    "info": inputs.pop("info", None),
                    "task": inputs.pop("task", None),
                    "video_id": inputs.pop("video_id", None),
                    "src_sec": inputs.pop("src_sec", None),
                }
                _v = res_i["video_id"]
                if str(_v[0] if isinstance(_v, list) else _v) in _done_ids:
                    continue
                if model_args.use_evidence:
                    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[0]
                    EVIDENCE_GEN_KEYS = (
                        "video_token_indices_list_src", "audio_token_indices_list_src",
                        "second_per_grid_ts",
                        "last_text_token_indices",
                        "video_token_indices_per_timestep",
                        "audio_token_indices_per_timestep",
                    )
                    for k in EVIDENCE_GEN_KEYS:
                        res_i[k] = inputs.get(k, None)
                    res_i["timestep_duration"] = inputs.get("timestep_duration", None)
                inputs = {k: v.to(f"cuda:{torch.cuda.current_device()}") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                gen_kwargs = dict(inputs)
                if model_args.use_evidence:
                    for k in EVIDENCE_GEN_KEYS:
                        gen_kwargs[k] = res_i[k]
                model.config.audio_mean_weight = task_audio_weight.get(
                    res_i.get("task"), default_audio_weight)
                model.config.grounding_snap_window = (model_args.grounding_snap_window
                    if (not snap_tasks or res_i.get("task") in snap_tasks) else 0)
                # Resample (attempt 0 greedy) on prompt echo (v2a/a2v) or degenerate output; keep the best candidate
                echo_n = (model_args.echo_best_of_n
                          if (model_args.echo_best_of_n > 1 and data_args.num_sample == 1
                              and res_i.get("task") in ("v2a", "a2v")) else 0)
                degen_n = (model_args.degenerate_retry
                           if (model_args.degenerate_retry > 1 and data_args.num_sample == 1
                               and res_i.get("task") in ("t2v", "t2a", "v2a", "a2v")) else 0)
                retry_n = max(echo_n, degen_n)
                n_attempts = retry_n if retry_n else data_args.num_sample
                echo_best = None  # ((is_degenerate, overlap_with_prompt), text)
                max_new = task_body_cap.get(res_i.get("task"), 1024) + 2   # +2: <G> and EOS
                for attempt in range(n_attempts):
                    do_sample_now = bool(data_args.do_sample) or (retry_n > 0 and attempt > 0)
                    with torch.no_grad():
                        if model_args.use_evidence:
                            outputs = model.generate_parallel(
                                **gen_kwargs,
                                max_new_tokens=max_new,
                                do_sample=do_sample_now,
                                top_p=0.9,
                                constrain_planning=(res_i.get("task") == "dvc"),
                                min_body_tokens=(model_args.min_body_tokens
                                                 if res_i.get("task") in ("t2v", "t2a", "v2a", "a2v") else 0),
                                single_event_no_replan=(model_args.single_event_no_replan
                                                        and res_i.get("task") in ("t2v", "t2a", "v2a", "a2v", "v2t", "a2t")),
                                record_separator_id=newline_token_id,
                                deinterleave=data_args.deinterleave_parallel_output)
                        else:
                            # Upstream video-SALMONN-2 path
                            outputs = model.generate(
                                **gen_kwargs,
                                max_new_tokens=max_new,
                                do_sample=do_sample_now,
                                top_p=0.9)
                    output_trimmed = outputs[0, len(inputs["input_ids"][0]):]
                    output_text = tokenizer.decode(output_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    
                    if model_args.use_evidence:
                        tgt = getattr(model, 'tgt', None)
                        assert tgt is not None, "model.tgt is None: generate did not produce grounding results (check use_evidence config of the loaded model)"
                        assert len(output_trimmed) == len(tgt[0]), f"tgt length {len(tgt[0])} != generated length {len(output_trimmed)}"
                       
                        evi_token_id = getattr(model.config, "evi_token_id", None)
                        assert evi_token_id is not None, "model.config.evi_token_id missing: the loaded model was not trained with use_evidence"
                        evi_like_mask = output_trimmed == evi_token_id
                        model.match_inds = torch.where(evi_like_mask)[0].tolist()

                        if len(model.match_inds) > 0:
                           
                            timestep_duration = res_i["timestep_duration"]
                            evi_strs = []
                            for i in model.match_inds:
                                intervals = tgt[0][i]
                                intervals_sec = timestep_intervals_to_seconds(intervals, timestep_duration)
                                evi_strs.append(", ".join(s.strip() for s in format_intervals(intervals_sec)))

                            # t2v/t2a: optionally output the given query interval instead of the readout
                            src_sec = res_i.get("src_sec")
                            if (model_args.src_interval_override
                                    and res_i.get("task") in ("t2v", "t2a")
                                    and isinstance(src_sec, (list, tuple)) and len(src_sec) == 2):
                                evi_strs[0] = f"{src_sec[0]} - {src_sec[1]}"

                            parts = output_text.split('<G>')
                            assert len(parts) == len(evi_strs) + 1, \
                                f"placeholder count {len(parts)-1} != interval count {len(evi_strs)}"
                            pieces = [parts[0]]
                            for i, (iv, rest) in enumerate(zip(evi_strs, parts[1:])):
                                pieces.append(iv)
                                if rest == "":
                                    if i < len(evi_strs) - 1:
                                        pieces.append("; ")
                                elif rest[0] not in " ;,.<":
                                    pieces.append(" ")
                                pieces.append(rest)
                            output_text = "".join(pieces)

                        # dvc: NMS over the records
                        if model_args.dvc_dedup_iou > 0 and res_i.get("task") == "dvc":
                            output_text = dedup_dvc_records(
                                output_text, model_args.dvc_dedup_iou,
                                drop_degenerate=model_args.degenerate_retry > 0)

                        # Length cap hit: trim to the last sentence end
                        if (res_i.get("task") in task_body_cap and len(output_trimmed) >= max_new - 1):
                            m_iv = re.match(r"^[\d\.\s\-,]*", output_text)
                            head, body_ = output_text[:m_iv.end()], output_text[m_iv.end():]
                            output_text = head + trim_to_sentence(body_)

                    if retry_n:
                        body = re.sub(r"^[\d\.\s\-,]*", "", output_text).strip()
                        deg = is_degenerate(body) if degen_n else False
                        ov = 0.0
                        if echo_n:
                            prompt_val = (res_i["prompt"]["value"]
                                          if isinstance(res_i.get("prompt"), dict) else str(res_i.get("prompt")))
                            ov = echo_overlap(body, prompt_payload(prompt_val))
                        score = (1 if deg else 0, ov)
                        if echo_best is None or score < echo_best[0]:
                            echo_best = (score, output_text)
                        if not deg and (not echo_n or ov < model_args.echo_overlap_threshold):
                            break  # accepted
                        continue
                    if data_args.num_sample == 1:
                        res_i["pred"] = output_text
                    else:
                        if "pred" in res_i:
                            res_i["pred"].append(output_text)
                        else:
                            res_i["pred"] = [output_text]
                if retry_n:
                    res_i["pred"] = echo_best[1]
                    res_i["echo_overlap"] = round(echo_best[0][1], 3)
                    res_i["degenerate"] = bool(echo_best[0][0])
                    res_i["echo_attempts"] = attempt + 1
                if not res_i["should_use"]:
                    continue
                result.append(res_i)
                _partial_f.write(json.dumps(_to_jsonable(res_i), ensure_ascii=False) + "\n"); _partial_f.flush()
        _partial_f.close()

        with open(os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.json"), "w") as f:
            json.dump(_to_jsonable(result), f, indent=2, ensure_ascii=False)
        
        return

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
