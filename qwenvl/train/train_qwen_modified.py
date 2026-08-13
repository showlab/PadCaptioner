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
import torch
import transformers
import json
import sys
from pathlib import Path
import numpy as np
import torch
# torch.set_printoptions(profile="full")
import random

from torch.utils.data import DataLoader

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


# local_rank = None

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


def timestep_intervals_to_seconds(intervals, timestep_duration, round_digits=2):
    """Convert timestep-index intervals [[a,b], ...] to real-time intervals in seconds; each step lasts timestep_duration seconds."""
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

    # Keep frm_head and evi_head always trainable (used in evidence mode)
    if hasattr(model, 'frm_head'):
        model.frm_head.requires_grad_(True)
    if hasattr(model, 'evi_head'):
        model.evi_head.requires_grad_(True)
    if hasattr(model, 'score_head'):
        model.score_head.requires_grad_(True)


def print_trainable_parameters_summary(
    model, 
    num_added_tokens=0, 
    new_token_ids=None, 
    original_tokenizer_size=None,
    rank=0,
    enabled=True
):
    """
    Print a summary of trainable parameters.

    Args:
        model: model to inspect
        num_added_tokens: number of newly added tokens
        new_token_ids: list of new token IDs
        original_tokenizer_size: original tokenizer size
        rank: current process rank (only prints when rank == 0)
        enabled: whether printing is enabled (default True)
    """
    if not enabled or rank != 0:
        return
    
    print("\n" + "=" * 80)
    print("Trainable Parameters Summary")
    print("=" * 80)
    
    # Collect trainable parameters
    trainable_params = []
    total_trainable = 0
    embedding_trainable = False
    new_token_embedding_trainable = {}
    
    for k, v in model.named_parameters():
        if v.requires_grad:
            trainable_params.append((k, v.shape, v.numel()))
            total_trainable += v.numel()
            
            # Check whether this is an embedding-related parameter
            if "embed" in k.lower() or "lm_head" in k.lower():
                embedding_trainable = True
                print(f"  [EMBEDDING] {k}: {v.shape} ({v.numel():,} params)")
                
                # Record whether the new tokens' embeddings are trainable (by parameter name).
                # Keyed on new_token_ids rather than num_added_tokens: when resuming, the tokens
                # already exist in the tokenizer so num_added_tokens is 0, but their training
                # status still needs to be checked/printed
                if new_token_ids:
                    if "embed_tokens" in k or "lm_head" in k:
                        for tid in new_token_ids:
                            if tid < v.shape[0]:
                                if tid not in new_token_embedding_trainable:
                                    new_token_embedding_trainable[tid] = {}
                                new_token_embedding_trainable[tid][k] = v.requires_grad
    
    # Print all trainable parameters
    print(f"\nTotal trainable parameters: {total_trainable:,}")
    print(f"\nAll trainable parameters:")
    for k, shape, numel in trainable_params:
        if "embed" not in k.lower() and "lm_head" not in k.lower():
            print(f"  {k}: {shape} ({numel:,} params)")
    
    # Check new tokens' embedding training status (on resume num_added_tokens=0 but new_token_ids is non-empty; still check)
    if new_token_ids:
        print(f"\n" + "-" * 80)
        print("New Token Embedding Training Status:")
        print("-" * 80)
        
        # Get the embeddings (may live under base_model when LoRA is used)
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
        
        # Check the overall state of the embedding modules
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
        
        # Check the embedding rows at the new token positions
        all_trainable = True
        for tid in new_token_ids:
            if tid < len(input_emb.weight) and tid < len(output_emb.weight):
                # Check requires_grad on the embedding weight.
                # Note: with a gradient hook, requires_grad=True but gradients get masked
                input_trainable = input_emb.weight.requires_grad
                output_trainable = output_emb.weight.requires_grad
                
                # Check whether a backward hook is registered
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
    """
    Add special tokens to the tokenizer, then resize and initialize the model's embedding layers.

    Flow:
    1. Pad the tokenizer up to the embedding size (using reserved tokens)
    2. When adding special tokens, prefer mapping them to existing empty slots (reserved tokens)
    3. Resize the model embeddings to hold the new tokens

    Args:
        model: model to modify
        tokenizer: tokenizer instance
        special_tokens: list of special tokens to add, e.g. ["<G>", "<S>", "<shift>"]. If None, no tokens are added.

    Returns:
        tuple: (original_tokenizer_size, num_added_tokens, new_token_ids)
    """
    # Save the original tokenizer size (before adding tokens)
    original_tokenizer_size = len(tokenizer)
    model_embedding_size = len(model.get_input_embeddings().weight)
    
    # step1: add the special tokens
    num_added_tokens = 0
    new_token_ids = []
    added_desc = []
    
    if special_tokens is not None and len(special_tokens) > 0:
        # Mapping from token name to config attribute name
        # NOTE: attr names must match configuration_qwen2_5_vl.py / modeling / checkpoint_utils.py,
        # which all read evi_token_id / evi_end_token_id / shift_token_id.
        token_to_attr_name = {
            "<G>": "evi_token_id",
            "<S>": "evi_end_token_id",
            "<shift>": "shift_token_id",
        }
        
        # Add all special tokens first (HuggingFace assigns IDs automatically)
        for token in special_tokens:
            # Add the special token directly
            special_tokens_dict = {"additional_special_tokens": [token]}
            num_added = tokenizer.add_special_tokens(special_tokens_dict)
            if num_added > 0:
                num_added_tokens += num_added
                token_id = tokenizer.convert_tokens_to_ids(token)
                new_token_ids.append(token_id)
                added_desc.append(f"added {token} with ID {token_id}")
            else:
                # Token already exists, fetch its ID
                token_id = tokenizer.convert_tokens_to_ids(token)
                new_token_ids.append(token_id)
                added_desc.append(f"{token} already present with ID {token_id}")
        print(f"Adding {len(special_tokens)} special tokens ({', '.join(added_desc)})")
    
    # step2: if model embedding size > tokenizer size, pad the tokenizer with reserved tokens
    current_tokenizer_size = len(tokenizer)
    
    if model_embedding_size > current_tokenizer_size:
        need = model_embedding_size - current_tokenizer_size
        new_tokens = [f"<reserved_{i}>" for i in range(need)]
        tokenizer.add_tokens(new_tokens, special_tokens=False)
        
    # Write token IDs into config
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
    
    # step3: resize model embeddings to hold the new tokens (only when necessary)
    if num_added_tokens > 0:
        # Re-fetch the final IDs of all special tokens
        final_token_ids = []
        if special_tokens is not None:
            for token in special_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    final_token_ids.append(token_id)
        
        max_final_token_id = max(final_token_ids) if final_token_ids else -1
        
        # Resizing is only needed when a final token ID exceeds model_embedding_size
        if max_final_token_id >= model_embedding_size:
            # Resize the model to hold the new tokens
            new_vocab_size = max(current_tokenizer_size, max_final_token_id + 1)
            model.resize_token_embeddings(new_vocab_size)
            
            # Initialize the newly added embeddings
            with torch.no_grad():
                input_emb = model.get_input_embeddings().weight.data
                output_emb = model.get_output_embeddings().weight.data
                
                # Mean initialization (mean of embeddings within the original tokenizer range)
                input_mean = input_emb[:model_embedding_size].mean(dim=0)
                output_mean = output_emb[:model_embedding_size].mean(dim=0)

                # Initialize the newly added rows (starting from model_embedding_size)
                num_new_embeddings = new_vocab_size - model_embedding_size
                input_emb[model_embedding_size:] = input_mean.unsqueeze(0).expand(num_new_embeddings, -1)
                output_emb[model_embedding_size:] = output_mean.unsqueeze(0).expand(num_new_embeddings, -1)
            
            # Re-initialize the input/output embeddings of the special tokens
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
            # Even without resizing, still re-initialize the special tokens' embeddings (if within range)
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
    """
    Ensure the newly added tokens' embeddings are trainable.
    After LoRA is applied, the embedding training state must be set again.

    Args:
        model: model to modify
        num_added_tokens: number of newly added tokens
        new_token_ids: list of new token IDs
        rank: current process rank (for printing)
    """
    if new_token_ids:
        input_emb = model.get_input_embeddings()
        output_emb = model.get_output_embeddings()
        
        if input_emb is not None and output_emb is not None:
            input_emb.weight.requires_grad_(True)
            output_emb.weight.requires_grad_(True)
            
            # Create a hook that freezes old tokens' embeddings, keeping gradients only for new tokens
            def create_selective_grad_hook(trainable_token_ids):
                """Create a hook that keeps gradients only for new tokens and freezes old ones."""
                trainable_set = set(trainable_token_ids)
                def hook(grad):
                    if grad is not None:
                        # Build mask: 1 at new token positions, 0 elsewhere
                        vocab_size = grad.shape[0]
                        mask = torch.zeros(vocab_size, device=grad.device, dtype=grad.dtype)
                        for tid in trainable_set:
                            if tid < vocab_size:
                                mask[tid] = 1.0
                        # Broadcast to [vocab_size, hidden_size]
                        mask = mask.view(-1, 1).expand_as(grad)
                        return grad * mask
                    return grad
                return hook
            
            # Register hooks for selective freezing
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
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert data_args.train_type in ["sft", "dpo", "gdpo", "grpo"], f"train_type {data_args.train_type} is not supported"

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

    # Check whether dist has been initialized
    try:
        rank = dist.get_rank() if dist.is_initialized() else 0
    except:
        rank = 0
    
    if not data_args.run_test:  # training
        # Initialize variables (so they are also available in the else branch)
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
        
        # Choose attn_implementation based on use_flashattention
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
            model.config.audio_mean_weight = model_args.audio_mean_weight  # audio weight (0-1) in the video+audio sum
            model.config.use_score_head = model_args.use_score_head
            model.config.use_attention_head = model_args.use_attention_head
            model.config.union_aggregation = model_args.union_aggregation
            model.config.save_gpu_trick = model_args.save_gpu_trick
        
            model.config.vision_audio_layer_index = model_args.vision_audio_layer_index  # only used when use_evidence=True
            original_tokenizer_size, num_added_tokens, new_token_ids = resize_and_initialize_embeddings(
                model=model, tokenizer=tokenizer, special_tokens=["<G>", "<S>", "<shift>"])
        else:
            original_tokenizer_size = len(tokenizer)  # keep the original size even when no tokens are added

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
            # model.save_pretrained(os.path.join(training_args.output_dir, "base/"))

        set_model(model_args, model)

        if training_args.no_audio:
            del model.audio

        if model_args.use_lora:
            from peft import LoraConfig, get_peft_model
            module_to_save = [] # modules to fully train (besides LoRA adapters) and save as complete weights
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
            # Note: frm_head and evi_head are NOT added to modules_to_save to avoid duplicated
            # parameters; they are set trainable directly after LoRA is applied
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj"], # find_all_linear_names(model),
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
                # Try different access paths for the heads (depends on how LoRA wraps the model)
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

                if hasattr(model, 'score_head'):
                    model.score_head.requires_grad_(True)
                elif hasattr(model, 'base_model') and hasattr(model.base_model, 'score_head'):
                    model.base_model.score_head.requires_grad_(True)
                elif hasattr(model, 'model') and hasattr(model.model, 'score_head'):
                    model.model.score_head.requires_grad_(True)
        
        # Ensure the newly added tokens' embeddings are trainable
        if model_args.use_evidence:
            setup_new_token_embeddings_for_training(model=model, num_added_tokens=num_added_tokens, new_token_ids=new_token_ids, rank=dist.get_rank())
        
        # Print trainable-parameter summary
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
        
        def _ckpt_step(p):
            """Accept only checkpoint-<digits> directories; ignore other same-prefix dirs
            (e.g. conversion outputs), whose names would make int() raise and crash the resume."""
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
            # Atomic write so other ranks never read a half-written file
            tmp_meta_path = resume_meta_path + ".tmp"
            with open(tmp_meta_path, "w") as f:
                json.dump(current_meta, f, indent=2)
            os.replace(tmp_meta_path, resume_meta_path)

        if resume_ckpts:   # resume from checkpoint
            logging.info("checkpoint found, resume training")
            latest_checkpoint = str(resume_ckpts[-1])
            resumed_step = None
            ckpt_state_file = os.path.join(latest_checkpoint, "trainer_state.json")
            if os.path.isfile(ckpt_state_file):
                with open(ckpt_state_file) as f:
                    resumed_step = json.load(f).get("global_step")
            did_weight_only_fallback = False
            try:
                # Try normal checkpoint loading (including optimizer states)
                trainer.train(resume_from_checkpoint=True)
            except Exception as e:
                error_str = str(e)
                # Detect world-size mismatch errors (keywords like "DP world size", "ZeRORuntimeException", etc.)
                is_world_size_error = (
                    "world size" in error_str.lower() or
                    "zeroruntimeexception" in error_str.lower() or
                    "dp world size" in error_str.lower() or
                    "automatic adjustment" in error_str.lower()
                )
                if is_world_size_error:
                    # On world-size mismatch, load only model weights, not optimizer states
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
        else:   # train from scratch
            trainer.train()
        
        ############################################################ model saving ############################################################
        # step1: save trainer state for future resume (no model weights)
        if dist.get_rank() == 0:
            logging.info(f"Training finished. Total steps = {trainer.state.global_step}, max_steps = {trainer.state.max_steps}")
        trainer.save_state()  


        # step2: save the image preprocessor config and tokenizer
        is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        if is_rank0:
            data_args.image_processor.save_pretrained(training_args.output_dir)
            if model_args.use_evidence:
                tokenizer.save_pretrained(training_args.output_dir)
                print(f"✓ Saved tokenizer to {training_args.output_dir}")

        # step3: before saving and later inference, enable use_cache in config so generation uses the KV cache
        model.config.use_cache = True

        # step4: when use_evidence=True, save the full model weights and config
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
                unwrapped = getattr(trainer.model, "module", trainer.model)  # strip DDP/DeepSpeed wrapper
                if isinstance(unwrapped, PeftModel):
                    unwrapped = unwrapped.merge_and_unload()  # merge LoRA into base for a plain backbone (keys match the inference model)
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
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)    # saves LoRA weights only
    else:   # testing
        
        # 0. Resolve the model directory to load:
        def _dir_has_file(d, name):
            return bool(d) and os.path.isdir(d) and os.path.isfile(os.path.join(d, name))

        from qwenvl.train.checkpoint_utils import resolve_test_model_path
        model_load_path = resolve_test_model_path(
            model_args.model_name_or_path, training_args.output_dir,
            auto_convert=data_args.auto_convert_checkpoint,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            logging.info(f"[test] resolved model path: {model_load_path}")

        # 1. Load tokenizer
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

        # 2. Load the dataset and create the output folder
        pred_rank = training_args.pred_rank
        if torch.cuda.device_count() > 1:
            pred_rank = pred_rank * torch.cuda.device_count() + torch.cuda.current_device()
            data_args.dataset_use = f"dataset/{pred_rank}.json"
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_args=model_args)
        os.makedirs(os.path.join(training_args.output_dir, training_args.run_name), exist_ok=True)

        # 3. Configure DeepSpeed 
        if torch.cuda.device_count() > 1:
            ds_config = {
                "fp16": {"enabled": False},
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": 3
                },
                "train_micro_batch_size_per_gpu": 1,
            }
        
        # 4. Load the model
        if model_args.lora_ckpt == "No" or model_args.use_evidence:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logging.info(f"loading trained model from {model_load_path}")
            model = video_SALMONN2_plus.from_pretrained(
                model_load_path,
                attn_implementation="sdpa",
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        else:
            # Legacy upstream video-SALMONN-2 path (use_evidence=False with a separate LoRA):
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

        # Grounding salient-segment threshold (paper: LongVALE 0.5 / ChronusAV 0.7), passed per dataset by the script
        model.config.grounding_threshold = model_args.grounding_threshold
        model.config.grounding_merge_gap = model_args.grounding_merge_gap
        model.config.grounding_select = model_args.grounding_select

        if torch.cuda.device_count() > 1:
            import deepspeed
            ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
            ds_engine.module.eval()
            model = ds_engine.module
        else:
            model.cuda()

        result = []
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
                }
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
                for _ in range(data_args.num_sample):
                    with torch.no_grad():
                        if model_args.use_evidence:
                            outputs = model.generate_parallel(
                                **gen_kwargs,
                                max_new_tokens=1024,
                                do_sample=data_args.do_sample,
                                top_p=0.9,
                                constrain_planning=(res_i.get("task") == "dvc"),
                                record_separator_id=newline_token_id,
                                deinterleave=data_args.deinterleave_parallel_output)
                        else:
                            # Legacy upstream video-SALMONN-2 path: plain single-stream generation
                            outputs = model.generate(
                                **gen_kwargs,
                                max_new_tokens=1024,
                                do_sample=data_args.do_sample,
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

                    if data_args.num_sample == 1:
                        res_i["pred"] = output_text
                    else:
                        if "pred" in res_i:
                            res_i["pred"].append(output_text)
                        else:
                            res_i["pred"] = [output_text]
                if not res_i["should_use"]:
                    continue
                result.append(res_i)
        def _to_jsonable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(v) for v in obj]
            return obj

        with open(os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.json"), "w") as f:
            json.dump(_to_jsonable(result), f, indent=2, ensure_ascii=False)
        
        return

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
