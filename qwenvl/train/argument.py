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

from pickle import TRUE
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
    lora_ckpt: str = field(default="No")
    use_evidence: bool = field(default=False)
    use_score_head: bool = field(default=False, metadata={"help": "When use_evidence=True: if True, use score_head to weight token means for evi embedding; if False, use simple mean."})
    use_attention_head: bool = field(default=False, metadata={"help": "When use_evidence=True: if True, use attention_head to weight token means for evi embedding; if False, use simple mean."})
    use_flashattention: bool = field(default=True, metadata={"help": "If True, use flash_attention_2; if False, use sdpa. Note: with use_evidence the inference path loads the model on sdpa regardless of this flag, because the parallel branches are isolated with a custom 4D attention mask that FlashAttention-2 does not support. Training sets it to False for the same reason."})
    vision_audio_layer_index: Optional[int] = field(default=6)
    loss_match_weight: float = field(default=3.0)
    audio_mean_weight: float = field(default=1, metadata={"help": "Weight for audio mean when summing video+audio (0-1). Used in evi embedding and vision info."})
    union_aggregation: bool = field(default=False, metadata={"help": "If True, in evi semantic aggregation, video and audio are conbined first then softmax. If False (default, consistent with the paper and the inference path), they are softmax independently, then added to the evi as video + audio_mean_weight * audio."})
    save_gpu_trick: bool = field(default=True, metadata={"help": "If True, only return required layer attention and hidden state to save GPU memory. If False, return all layers"})
    grounding_threshold: float = field(default=0.8, metadata={"help": "Inference grounding: timesteps with similarity >= threshold * max are salient. Paper: 0.5 for LongVALE, 0.7 for ChronusAV. Legacy default 0.8."})
    grounding_merge_gap: int = field(default=2, metadata={"help": "Inference grounding: salient runs separated by <= this many timesteps are bridged into one segment; larger gaps keep segments apart (one timestep is ~2s at 256-frame sampling of a few-minute video)."})
    grounding_select: str = field(default="mass", metadata={"help": "Inference grounding: how to pick the final interval among bridged segments. 'mass' = highest summed similarity (strength x duration, default), 'longest' = longest duration, 'mean' = highest mean similarity (biases toward short spikes; for ablation)."})
    
    

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
    auto_convert_checkpoint: bool = field(default=True, metadata={"help": "Test time: if the model path has no finished model but contains checkpoint-N dirs, automatically convert the largest-step checkpoint into a loadable model (cached and reused)."})
    aggregation_jitter_ratio: float = field(default=0.1, metadata={"help": "Train-time robustness for interval-feature aggregation: each GT interval boundary used for <G> feature aggregation is jittered by uniform(-r, r) * interval_length (match-loss GT stays clean). Mitigates the train(GT intervals)/inference(predicted intervals) exposure bias. 0 disables."})
    do_sample: bool = field(default=False)
    num_sample: int = field(default=1)
    deinterleave_parallel_output: bool = field(default=True, metadata={"help": "Test time: reassemble the parallel dvc prediction into readable per-event records (interval + its caption, one per line). False keeps the raw physical generation order (serial segment, anchors, then branch tokens round by round) for debugging."})
    train_type: str = field(default="sft")
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
