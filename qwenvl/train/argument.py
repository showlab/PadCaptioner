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

import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    model_base: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_mm_audio: bool = field(default=False)
    tune_mm_qformer: bool = field(default=False)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    nan_guard: bool = field(default=False, metadata={"help": "Training: abort on the first non-finite or zero loss/grad_norm."})
    lora_target: str = field(default="qkv", metadata={"help": "LLM layers to wrap with LoRA: 'qkv' (q/k/v) or 'all' (q/k/v/o + MLP)."})
    lora_ckpt: str = field(default="No")
    use_evidence: bool = field(default=False)
    use_attention_head: bool = field(default=False, metadata={"help": "With use_evidence: weight token means with an attention head instead of a plain mean."})
    use_flashattention: bool = field(default=True, metadata={"help": "Use flash_attention_2 if True, else sdpa (use_evidence forces sdpa for the parallel 4D mask)."})
    vision_audio_layer_index: Optional[int] = field(default=6)
    loss_match_weight: float = field(default=3.0)
    loss_stop_weight: float = field(default=1.0, metadata={"help": "Training: relative LM-loss weight of the planning-stop <S> token; <=1 disables."})
    loss_stop_symmetric: bool = field(default=False, metadata={"help": "With loss_stop_weight > 1: give dvc planning <G> positions the same weight as <S>."})
    loss_contrast_weight: float = field(default=0.0, metadata={"help": "Training: weight of the timestep InfoNCE term on the grounding curve; 0 disables."})
    loss_contrast_detach: bool = field(default=True, metadata={"help": "Backpropagate the InfoNCE term only into evi_head/frm_head (backbone detached)."})
    loss_contrast_tau: float = field(default=0.1, metadata={"help": "Temperature of the InfoNCE term."})
    grounding_target_decay: float = field(default=1.0, metadata={"help": "Training: decay of the soft grounding target 0.5^(decay * timestep_distance)."})
    loss_boundary_weight: float = field(default=0.0, metadata={"help": "Training: weight of the start/end boundary-peak BCE on the boundary heads; 0 disables."})
    grounding_boundary_decay: float = field(default=1.0, metadata={"help": "Training: decay of the boundary-peak soft target 0.5^(decay * |t - boundary|)."})
    loss_plateau_weight: float = field(default=0.0, metadata={"help": "Training: weight of the plateau-consistency hinge on the grounding curve; 0 disables."})
    grounding_plateau_ratio: float = field(default=0.7, metadata={"help": "Training: plateau ratio for loss_plateau_weight (match the inference grounding_threshold)."})
    loss_boundary_rank_weight: float = field(default=0.0, metadata={"help": "Training: weight of the internal-peak suppression on the boundary heads; 0 disables."})
    grounding_boundary_rank_margin: float = field(default=1.0, metadata={"help": "Training: logit margin for loss_boundary_rank_weight."})
    readout_only: bool = field(default=False, metadata={"help": "Training: train only the grounding readout heads; use with use_lora=False."})
    grounding_snap_window: int = field(default=0, metadata={"help": "Inference: snap interval edges to the boundary-head peaks within +- this many timesteps; 0 disables."})
    grounding_snap_tasks: str = field(default="", metadata={"help": "Comma list of tasks using grounding_snap_window (e.g. 'dvc'); empty = all tasks."})
    grounding_coverage_train: bool = field(default=False, metadata={"help": "Training, dvc: mask timesteps claimed by earlier events in the grounding loss (mirrors grounding_coverage_mask)."})
    loss_iou_weight: float = field(default=0.0, metadata={"help": "Training: weight of the interval soft-IoU loss added to the BCE match loss; 0 disables."})
    branch_interval_attention: bool = field(default=False, metadata={"help": "dvc: each parallel branch attends only to text and the media tokens of its own interval; train and test must agree."})
    branch_interval_audio_full: bool = field(default=False, metadata={"help": "With branch_interval_attention: keep the full audio stream visible and confine only video tokens."})
    branch_interval_margin: int = field(default=1, metadata={"help": "Timesteps of context kept on each side of the branch interval."})
    grounding_coverage_mask: bool = field(default=False, metadata={"help": "Inference, dvc: zero timesteps claimed by earlier <G>s before each planning readout."})
    self_aggregation_prob: float = field(default=0.0, metadata={"help": "Training: per-batch probability of aggregating <G> features from self-predicted intervals (no-grad pass); not ZeRO-3 safe; 0 disables."})
    audio_mean_weight: float = field(default=1, metadata={"help": "Audio weight w in the per-timestep feature video + w * audio."})
    save_gpu_trick: bool = field(default=True, metadata={"help": "Return only the required layer's attention/hidden states to save memory."})
    grounding_threshold: float = field(default=0.8, metadata={"help": "Inference: timesteps with similarity >= threshold * max are salient."})
    grounding_merge_gap: int = field(default=2, metadata={"help": "Inference: bridge salient runs separated by <= this many timesteps."})
    max_planning_events: int = field(default=30, metadata={"help": "Inference, dvc: force <S> after this many planned <G>s; 0 disables."})
    min_body_tokens: int = field(default=0, metadata={"help": "Inference, single-event caption tasks: suppress EOS/<S> until this many body tokens follow the first <G>; 0 disables."})
    dvc_dedup_iou: float = field(default=0.0, metadata={"help": "Inference, dvc: drop records whose interval IoU with a kept record reaches this value; 0 disables."})
    src_interval_override: bool = field(default=False, metadata={"help": "Inference, t2v/t2a: output the query interval given in the prompt instead of the readout."})
    echo_best_of_n: int = field(default=0, metadata={"help": "Inference, v2a/a2v: resample up to N times when the output echoes the prompt description; 0/1 disables."})
    single_event_no_replan: bool = field(default=False, metadata={"help": "Inference, single-event tasks: ban <G>/<S> after the first <G>."})
    task_max_body_tokens: str = field(default="", metadata={"help": "Inference: per-task token cap, e.g. 't2a=90,v2a=90'; empty = 1024 for all."})
    degenerate_retry: int = field(default=0, metadata={"help": "Inference: resample degenerate single-event captions up to N times and drop degenerate dvc records; 0/1 disables."})
    echo_overlap_threshold: float = field(default=0.15, metadata={"help": "echo_best_of_n: accept a candidate whose prompt-overlap F1 is below this."})
    task_audio_mean_weight: str = field(default="", metadata={"help": "Inference: per-task audio_mean_weight override, e.g. 'v2a=3.0'; empty = audio_mean_weight."})



@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    base_interval: float = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    run_test: bool = field(default=False)
    auto_convert_checkpoint: bool = field(default=True, metadata={"help": "Test: convert the latest checkpoint-N into a loadable model when no final model exists."})
    aggregation_jitter_ratio: float = field(default=0.1, metadata={"help": "Training: jitter GT interval boundaries used for <G> aggregation by uniform(-r, r) * length; 0 disables."})
    do_sample: bool = field(default=False)
    num_sample: int = field(default=1)
    deinterleave_parallel_output: bool = field(default=True, metadata={"help": "Test: reassemble parallel dvc output into per-event records; False keeps raw generation order."})
    feature_size: int = field(default=128)
    chunk_length: int = field(default=30)
    hop_length: int = field(default=160)
    sampling_rate: int = field(default=16000)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    pred_rank: int = field(default=0)
    no_audio: bool = field(default=False)
    print_trainable_params: bool = field(
        default=False,
        metadata={
            "help": "Whether to print trainable parameters summary. Default is False."
        },
    )
