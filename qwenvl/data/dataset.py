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
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Union
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchcodec.decoders import VideoDecoder, AudioDecoder
import transformers

from .rope2d import get_rope_index_25, get_rope_index_2
from decord import VideoReader, cpu

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_AUDIO_TOKEN = "<audio>"

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def split_into_groups(counts, groups, second_per_grid_ts=None): # split audio token counts across video timesteps
    result = []
    if second_per_grid_ts is None:
        for count, g in zip(counts, groups):
            g = g.item()
            base = count // g
            remainder = count % g
            if remainder == 0:
                group_list = [base] * g
            else:
                group_list = [base] * g
                step = g / remainder
                for i in range(1, remainder + 1):
                    position = i * step
                    index = math.floor(position) - 1
                    if index >= g:
                        index = g - 1
                    group_list[index] += 1
            result.append(group_list)
    else:
        for count, g, second in zip(counts, groups, second_per_grid_ts):
            g = g.item()
            frame_idx = (torch.arange(g) * second * 2).long()   # audio token index per timestep (one token per 0.5s)
            per_grid_t = torch.diff(frame_idx)
            group_list = per_grid_t.tolist()
            group_list.append(count - sum(group_list))  # remainder goes to the last timestep
            result.append(group_list)
    return result

def generate_id_target(
    source,
    grid_thw_image, 
    grid_thw_video, 
    audio_lengths, 
    tokenizer, 
    target_role,
    merge_size: int = 2,
    second_per_grid_ts: List = [],
    gpt_value_segments: Optional[List] = None
):
    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    input_id, target = [], []

    input_id += tokenizer.apply_chat_template(
        [{"role": "system", "content": system_message}]
    )   # system prompt
    target += [IGNORE_INDEX] * len(input_id)
    for conv in source:
        try:
            role = conv["role"]
            content = conv["content"]
        except:
            role = conv["from"]
            content = conv["value"]
        if role not in ["human", target_role]:
            continue

        role = roles.get(role, role)
        if role == "user":
            if "<image>" in content:
                parts = content.split("<image>")
                new_parts = []
                for i in range(len(parts) - 1):
                    new_parts.append(parts[i])
                    replacement = (
                        "<|vision_start|>"
                        + f"<|image_pad|>"
                        * grid_thw_image[i]
                        + "<|vision_end|>"
                    )
                    new_parts.append(replacement)
                new_parts.append(parts[-1])
                content = "".join(new_parts)
            # Interleaved video/audio placeholders
            if "<video>" in content:
                parts = content.split("<video>")
                new_parts = []
                if audio_lengths is None:
                    grid_thw_video = [
                        merged_thw.prod() // merge_size**2
                        for merged_thw in grid_thw_video
                    ]
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|video_pad|>"
                            * grid_thw_video[i]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)
                else: # interleave video and audio tokens per timestep
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        if second_per_grid_ts is None: # distribute audio tokens by frame count
                            per_timestep_audio_len = split_into_groups(audio_lengths, [grid_thw_video[i][0] for i in range(len(grid_thw_video))])
                        else: # distribute audio tokens by time
                            per_timestep_audio_len = split_into_groups(audio_lengths, [grid_thw_video[i][0] for i in range(len(grid_thw_video))], [ts[0] for ts in second_per_grid_ts])
                        replacement = "<|vision_start|>"
                        for timestep in range(grid_thw_video[i][0]):
                            replacement += (
                                f"<|video_pad|>" 
                                * (grid_thw_video[i][1] * grid_thw_video[i][2] // merge_size**2)
                                + f"<|audio_pad|>"
                                * per_timestep_audio_len[i][timestep]
                            )   # frame's video tokens, then its audio tokens
                        replacement += "<|vision_end|>"
                        new_parts.append(replacement)
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            if "<audio>" in content:
                parts = content.split("<audio>")
                new_parts = []
                for i in range(len(parts) - 1):
                    new_parts.append(parts[i])
                    replacement = (
                        "<|vision_start|>" # reuse the vision start token
                        + f"<|audio_pad|>"
                        * audio_lengths[i]
                        + "<|vision_end|>"
                    )
                    new_parts.append(replacement)
                new_parts.append(parts[-1])
                content = "".join(new_parts)
        conv = [{"role": role, "content": content}]

        encode_id = tokenizer.apply_chat_template(conv)
        parallel_pad_flags = None   # True at branch filler positions (label IGNORE)
        if role == "assistant" and gpt_value_segments:
            # Parallel dvc answer: split on <shift>, pad branches to equal length and interleave them
            divide_id = tokenizer.encode("<shift>")[0]

            # parts: [0] serial planning, [1] anchors, [2:] per-event captions
            split_indices = [i for i, t in enumerate(encode_id) if t == divide_id]
            start = 0
            encode_id_parts = []
            for i in split_indices:
                encode_id_parts.append(encode_id[start:i])
                start = i + 1
            encode_id_parts.append(encode_id[start:])

            end_suffix_ids = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
            if len(encode_id_parts) - 1 >= 2:
                for i in range(2, len(encode_id_parts) - 1):
                    encode_id_parts[i] = encode_id_parts[i] + end_suffix_ids   # the last caption already ends with it

            # Pad captions with <|im_end|> (as fed to finished branches at inference); fillers get no loss
            branch_filler_id = end_suffix_ids[0]   # <|im_end|>
            pad_end = min(len(gpt_value_segments) + 2, len(encode_id_parts))    # min() is a bounds guard
            if pad_end > 2:
                segment = encode_id_parts[2:pad_end]
                max_len = max(len(p) for p in segment)
                # Fillers equal real <|im_end|> by value, so flag them by position
                filler_flags = [[False] * len(p) + [True] * (max_len - len(p)) for p in segment]
                encode_id_parts[2:pad_end] = [p + [branch_filler_id] * (max_len - len(p)) for p in segment]

                segment = encode_id_parts[2:pad_end]
                transposed = [t for pos in zip(*segment) for t in pos]
                transposed_flags = [f for pos in zip(*filler_flags) for f in pos]
                encode_id = (encode_id_parts[0] + [divide_id] + encode_id_parts[1] + transposed)
                parallel_pad_flags = [False] * (len(encode_id_parts[0]) + 1 + len(encode_id_parts[1])) + transposed_flags
            else:
                # Single event: concatenate as is
                encode_id = encode_id_parts[0]
                for part in encode_id_parts[1:]:
                    encode_id += [divide_id] + part

        input_id += encode_id
        if role in ["user", "system"]:
            target += [IGNORE_INDEX] * len(encode_id)   # no loss on the question
        else:
            target_mask = encode_id.copy()
            target_mask[:3] = [IGNORE_INDEX] * 3    # <|im_start|>assistant\n
            if parallel_pad_flags is not None:
                target_mask = [IGNORE_INDEX if flag else t for t, flag in zip(target_mask, parallel_pad_flags)]
            target += target_mask
    return input_id, target


def generate_event_id(input_id: List[int], target: List[int], divide_id: int, event_num: int) -> Tuple[torch.Tensor, List[int], List[int]]:
    """Remove the first divide_id (<shift>) and label each token with its event (-1 = shared prefix).

    Returns (event_id LongTensor, input_id, target).
    """
    divide_index = input_id.index(divide_id)
    prefix_length = divide_index

    input_id.pop(divide_index)
    target.pop(divide_index)

    # Interleaved branch tokens cycle through events 0..event_num-1
    n = len(input_id)
    suffix_len = n - prefix_length
    
    ablation_mode = False
    if not ablation_mode:
        event_id = torch.tensor(
            [-1] * prefix_length + [i % event_num for i in range(suffix_len)],
            dtype=torch.long,
        )
    else:
         event_id = torch.tensor(
            [-1] * (prefix_length-event_num) + [i % event_num for i in range(suffix_len+event_num)],
            dtype=torch.long,
        )
    return event_id, input_id, target


def adjust_position_ids_for_parallel(
    position_ids: torch.Tensor,
    event_id: Union[List[int], torch.Tensor],
    event_num: int,
) -> torch.Tensor:
    """Give each parallel branch its own positions continuing from the parallel start.

    position_ids: (3, batch_size, seq_len); event_id: -1 for the prefix, else the branch index.
    """
    device = position_ids.device
    dtype = position_ids.dtype
    event_id_t = torch.as_tensor(event_id, dtype=torch.long).to(device)
    parallel_mask = event_id_t >= 0

    parallel_start_idx = torch.where(parallel_mask)[0][0]
    parallel_start_position = position_ids[:, :, parallel_start_idx]  # (3, batch_size)

    if not parallel_mask.any():
        return position_ids

    # Every event_num consecutive tokens share one position
    parallel_indices = torch.where(parallel_mask)[0]  # (num_parallel,)
    position_in_parallel = (parallel_indices - parallel_start_idx).to(dtype).to(device)
    segment_indices = (position_in_parallel // event_num).view(1, 1, -1)  # (1, 1, num_parallel)
    new_positions = parallel_start_position.unsqueeze(2) + segment_indices  # (3, batch_size, num_parallel)
    position_ids[:, :, parallel_indices] = new_positions

    return position_ids, parallel_start_idx

def find_2th_index(lst, x):
    count = 0
    for i, v in enumerate(lst):
        if v == x:
            count += 1
            if count == 2:
                return i
    return None

def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
    audio_lengths = None,
    merge_size=2,
    second_per_grid_ts: List = [],
    gpt_value_segments: Optional[List] = None
) -> Dict:
    if second_per_grid_ts is not None and isinstance(second_per_grid_ts, list) and not isinstance(second_per_grid_ts[0], list):
        second_per_grid_ts = [second_per_grid_ts]
    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    input_ids, targets, last_text_token_indices = [], [], []
    event_id = None # parallel (dvc) samples only

    for i, source in enumerate(sources):
        try:
            if source[0]["from"] != "human":
                source = source[1:]
        except:
            print(sources)

        input_id, target = generate_id_target(source, grid_thw_image, grid_thw_video, audio_lengths, tokenizer, "gpt", merge_size, second_per_grid_ts, gpt_value_segments)

        if gpt_value_segments:  # parallel dvc sample
            divide_id = tokenizer.encode("<shift>")[0]
            event_id, input_id, target = generate_event_id(input_id, target, divide_id, len(gpt_value_segments))

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)
        last_text_token_indices.append(find_2th_index(input_id, tokenizer.encode("<|im_end|>")[0]))
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    last_text_token_indices = torch.tensor(last_text_token_indices, dtype=torch.long)
    return dict(
        input_ids=input_ids,
        labels=targets,
        event_id=event_id,
        last_text_token_indices=last_text_token_indices,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args, model_args=None):
        super(LazySupervisedDataset, self).__init__()
        # use_evidence lives in ModelArguments
        self.use_evidence = getattr(model_args, 'use_evidence', False) if model_args else False

        dataset = data_args.dataset_use.split(",")
        dataset_list = dataset
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data.split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data)
            else:
                annotations = json.load(open(data, "r"))
            list_data_dict += annotations

        for d in list_data_dict:
            if "<image>" in d["conversations"][0]["value"] and not "image" in d and "video" in d:
                d["conversations"][0]["value"] = d["conversations"][0]["value"].replace(
                    "<image>", "<video>"
                )
            if "<image>" in d["conversations"][0]["value"] and not "image" in d and not "video" in d and "audio" in d:
                d["conversations"][0]["value"] = d["conversations"][0]["value"].replace(
                    "<image>", "<audio>"
                )

        rank0_print(f"Total training samples: {len(list_data_dict)}")


        random.shuffle(list_data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_audio(self, audio_file):
        try:
            audio_kwargs = {
                "sampling_rate": 16000,
                "padding": "max_length",
                "return_attention_mask": False,
            }
            processor = copy.deepcopy(self.data_args.audio_processor)
            if isinstance(audio_file, list):
                audio_data = []
                for file in audio_file:
                    decoder = AudioDecoder(
                        file,
                        sample_rate=audio_kwargs["sampling_rate"],
                        num_channels=1,
                    )
                    audio = decoder.get_all_samples()
                    audio_data.append(audio.data.numpy().squeeze(0))
            else:
                decoder = AudioDecoder(
                    audio_file,
                    sample_rate=audio_kwargs["sampling_rate"],
                    num_channels=1,
                )
                audio = decoder.get_all_samples()
                audio_data = [audio.data.numpy().squeeze(0)]
            audio_inputs = []
            audio_lengths = []
            for idx in range(len(audio_data)):
                if audio_data[idx].shape[0] < audio_kwargs["sampling_rate"]:
                    padding = audio_kwargs["sampling_rate"] - audio_data[idx].shape[0]
                    audio_data[idx] = np.pad(audio_data[idx], (0, padding), mode="constant", constant_values=0)
                audio_lst = [audio_data[idx][k: k + 30 * audio_kwargs["sampling_rate"]] for k in range(0, len(audio_data[idx]), 30 * audio_kwargs["sampling_rate"])]
                spectrogram_lst = [processor(a, sampling_rate=audio_kwargs["sampling_rate"], return_tensors="pt")["input_features"].squeeze() for a in audio_lst]
                audio_inputs.append(torch.stack(spectrogram_lst, dim=0))
                audio_lengths.append(math.ceil(len(audio_data[idx]) / (30 * audio_kwargs["sampling_rate"])) * 60)
            return audio_inputs, audio_lengths
        except:
            return None, None


    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        torchcodec_video = None
        try:    # torchcodec first, decord as fallback
            torchcodec_video = self.video_torchcodec(video_file)
            return torchcodec_video
        except:
            try:
                decord_video = self.read_video_decord(video_file)
                return decord_video
            except Exception as e:
                print(f"torchcodec attempt failed: {e}")

    def read_video_decord(self, video_file):
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        ori_fps = vr.get_avg_fps()
        interval = getattr(self.data_args, "base_interval", 4)
        avg_fps = max(round(ori_fps * interval), 1)
        video_length = total_frame_num / ori_fps
        
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)
        frame_idx = [k for k in range(0, total_frame_num, round(avg_fps))]
        if len(frame_idx) > video_max_frames:
            frame_idx = np.linspace(0, total_frame_num - 1, video_max_frames, dtype=int).tolist()
        video = vr.get_batch(frame_idx).asnumpy().transpose(0, 3, 1, 2)
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file):  # sample frames every base_interval seconds
        device = "cpu"
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)
        new_pixel = self.data_args.video_max_frame_pixels
        if len(frame_idx) < video_max_frames:  # fewer frames: raise the per-frame pixel budget
            new_pixel = 0.95 * video_max_frames / len(frame_idx) * new_pixel
        processor.max_pixels = new_pixel
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"] # [t*h*w, channels]
        grid_thw = video_processed["video_grid_thw"][0] # (t, h, w)
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)  # seconds per temporal block
        return video_tensor, grid_thw, second_per_grid_ts

    def get_video_token_indices_by_time_range(
        self,
        start_sec: float,
        end_sec: float,
        grid_thw: torch.Tensor,
        second_per_grid_ts: List[float],
        spatial_merge_size: int = 2,
    ) -> List[Tuple[int, int]]:
        """Return [(start_idx, end_idx)] ranges of video tokens (in the video token sequence) overlapping [start_sec, end_sec]."""
        token_ranges = []
        current_time_sec = 0.0
        token_offset = 0

        if not isinstance(grid_thw, torch.Tensor):
            grid_thw = torch.tensor(grid_thw)

        num_grids = grid_thw.shape[0] if grid_thw.dim() > 1 else 1

        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)

        for grid_idx in range(num_grids):   # a single grid in practice
            t = grid_thw[grid_idx][0].item() if isinstance(grid_thw[grid_idx][0], torch.Tensor) else grid_thw[grid_idx][0]
            h = grid_thw[grid_idx][1].item() if isinstance(grid_thw[grid_idx][1], torch.Tensor) else grid_thw[grid_idx][1]
            w = grid_thw[grid_idx][2].item() if isinstance(grid_thw[grid_idx][2], torch.Tensor) else grid_thw[grid_idx][2]

            # LLM grid size (after spatial merge)
            llm_grid_t = t
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size

            second_per_grid_t = second_per_grid_ts[grid_idx] if grid_idx < len(second_per_grid_ts) else 1.0

            grid_start_time = current_time_sec
            grid_end_time = current_time_sec + llm_grid_t * second_per_grid_t

            if end_sec < grid_start_time:
                break
            if start_sec > grid_end_time:
                current_time_sec = grid_end_time
                token_offset += llm_grid_t * llm_grid_h * llm_grid_w
                continue

            # Temporal blocks overlapping the interval (floor start, ceil end)
            if second_per_grid_t > 0:
                relative_start = max(0, start_sec - current_time_sec)
                t_idx_start = int(math.floor(relative_start / second_per_grid_t))

                relative_end = max(0, end_sec - current_time_sec)
                t_idx_end = int(math.ceil(relative_end / second_per_grid_t))

                t_idx_start = max(0, min(t_idx_start, llm_grid_t))
                t_idx_end = max(t_idx_start, min(t_idx_end, llm_grid_t))

                if t_idx_start < t_idx_end:
                    start_token_idx = token_offset + t_idx_start * llm_grid_h * llm_grid_w
                    # end index is exclusive
                    end_token_idx = token_offset + t_idx_end * llm_grid_h * llm_grid_w

                    token_ranges.append((start_token_idx, end_token_idx))

            current_time_sec = grid_end_time
            token_offset += llm_grid_t * llm_grid_h * llm_grid_w
        
        return token_ranges

    def get_audio_token_indices_by_time_range(
        self,
        start_sec: float,
        end_sec: float,
        grid_thw: torch.Tensor,
        audio_lengths: List[int],
        second_per_grid_ts: List[float],
        spatial_merge_size: int = 2,
    ) -> List[Tuple[int, int]]:
        """Return [(start_idx, end_idx)] ranges of audio tokens (in the audio token sequence) overlapping [start_sec, end_sec]."""
        token_ranges = []
        current_time_sec = 0.0
        audio_token_offset = 0

        if not isinstance(grid_thw, torch.Tensor):
            grid_thw = torch.tensor(grid_thw)

        num_grids = grid_thw.shape[0] if grid_thw.dim() > 1 else 1

        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)

        for grid_idx in range(num_grids):
            t = grid_thw[grid_idx][0].item() if isinstance(grid_thw[grid_idx][0], torch.Tensor) else grid_thw[grid_idx][0]
            h = grid_thw[grid_idx][1].item() if isinstance(grid_thw[grid_idx][1], torch.Tensor) else grid_thw[grid_idx][1]
            w = grid_thw[grid_idx][2].item() if isinstance(grid_thw[grid_idx][2], torch.Tensor) else grid_thw[grid_idx][2]

            llm_grid_t = t
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size

            second_per_grid_t = second_per_grid_ts[grid_idx] if grid_idx < len(second_per_grid_ts) else 1.0
            audio_len = audio_lengths[grid_idx] if grid_idx < len(audio_lengths) else 0

            grid_start_time = current_time_sec
            grid_end_time = current_time_sec + llm_grid_t * second_per_grid_t

            if end_sec < grid_start_time:
                break
            if start_sec > grid_end_time:
                current_time_sec = grid_end_time
                audio_token_offset += audio_len
                continue

            # Each audio token covers 0.5s from grid_start_time
            if audio_len > 0:
                grid_audio_start_time = grid_start_time
                grid_audio_end_time = grid_start_time + audio_len * 0.5

                relative_start = max(0, start_sec - grid_audio_start_time)
                relative_end = max(0, end_sec - grid_audio_start_time)

                audio_idx_start = int(math.floor(relative_start / 0.5))
                audio_idx_end = int(math.ceil(relative_end / 0.5))

                audio_idx_start = max(0, min(audio_idx_start, audio_len))
                audio_idx_end = max(audio_idx_start, min(audio_idx_end, audio_len))

                if audio_idx_start < audio_idx_end:
                    start_token_idx = audio_token_offset + audio_idx_start
                    end_token_idx = audio_token_offset + audio_idx_end
                    token_ranges.append((start_token_idx, end_token_idx))

            current_time_sec = grid_end_time
            audio_token_offset += audio_len
        
        return token_ranges

    def _jitter_tgt_for_aggregation(self, tgt, timestep_duration, video_grid_thw):
        """Jitter GT interval boundaries by uniform(-r, r) * length for <G> feature aggregation only.

        Mirrored dvc tgt (planning + anchors) shares one jitter per event; the match-loss GT is untouched.
        """
        r = getattr(self.data_args, "aggregation_jitter_ratio", 0.0)
        if not r or tgt is None or not isinstance(tgt, (list, tuple)) or len(tgt) < 2:
            return tgt
        try:
            pairs = [[float(tgt[i]), float(tgt[i + 1])] for i in range(0, len(tgt) - 1, 2)]
        except (ValueError, TypeError):
            return tgt
        # Video duration for clamping, if known
        duration = float("inf")
        try:
            grid = video_grid_thw[0] if isinstance(video_grid_thw, list) else video_grid_thw
            t_dim = int(grid[0].item() if isinstance(grid, torch.Tensor) and grid.dim() == 1 else grid[0][0].item())
            if timestep_duration is not None and timestep_duration > 0:
                duration = t_dim * float(timestep_duration)
        except Exception:
            pass
        n = len(pairs)
        mirrored = n % 2 == 0 and n > 0 and pairs[: n // 2] == pairs[n // 2:]
        base = pairs[: n // 2] if mirrored else pairs
        jittered = []
        for s, e in base:
            length = max(e - s, 1e-6)
            s2 = min(max(s + random.uniform(-r, r) * length, 0.0), duration)
            e2 = min(max(e + random.uniform(-r, r) * length, 0.0), duration)
            if e2 < s2:
                s2, e2 = e2, s2
            jittered.append([s2, e2])
        if mirrored:
            jittered = jittered + [list(p) for p in jittered]
        return [x for p in jittered for x in p]

    def compute_video_token_indices_from_tgt(
        self,
        tgt: List[float],
        video_grid_thw,
        second_per_grid_ts,
        spatial_merge_size: int = 2,
    ) -> List[List[Tuple[int, int]]]:
        """Video token ranges for each [start_sec, end_sec] pair in the flat list tgt."""
        video_token_indices_list = []

        if isinstance(tgt, (list, tuple)) and len(tgt) >= 2:
            idx = 0
            while idx < len(tgt) - 1:
                try:
                    start_sec = float(tgt[idx])
                    end_sec = float(tgt[idx + 1])
                except (ValueError, TypeError):
                    idx += 2
                    continue
                
                if isinstance(video_grid_thw, list):
                    if len(video_grid_thw) > 0:
                        grid_thw_to_use = video_grid_thw[0] if isinstance(video_grid_thw[0], torch.Tensor) else torch.tensor(video_grid_thw[0])
                    else:
                        idx += 2
                        continue
                elif isinstance(video_grid_thw, torch.Tensor):
                    grid_thw_to_use = video_grid_thw[0] if video_grid_thw.dim() > 1 else video_grid_thw
                else:
                    grid_thw_to_use = torch.tensor(video_grid_thw)

                if isinstance(second_per_grid_ts, list):
                    if len(second_per_grid_ts) > 0:
                        if isinstance(second_per_grid_ts[0], list):
                            second_per_grid_ts_to_use = second_per_grid_ts[0]
                        else:
                            second_per_grid_ts_to_use = second_per_grid_ts
                    else:
                        idx += 2
                        continue
                elif isinstance(second_per_grid_ts, torch.Tensor):
                    second_per_grid_ts_to_use = second_per_grid_ts.tolist()
                else:
                    second_per_grid_ts_to_use = [second_per_grid_ts]

                token_ranges = self.get_video_token_indices_by_time_range(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    grid_thw=grid_thw_to_use,
                    second_per_grid_ts=second_per_grid_ts_to_use,
                    spatial_merge_size=spatial_merge_size,
                )
                
                video_token_indices_list.append(token_ranges)
                idx += 2
        
        return video_token_indices_list

    def compute_audio_token_indices_from_tgt(
        self,
        tgt: List[float],
        video_grid_thw,
        audio_lengths,
        second_per_grid_ts,
        spatial_merge_size: int = 2,
    ) -> List[List[Tuple[int, int]]]:
        """Audio token ranges for each [start_sec, end_sec] pair in the flat list tgt."""
        audio_token_indices_list = []

        if isinstance(tgt, (list, tuple)) and len(tgt) >= 2:
            idx = 0
            while idx < len(tgt) - 1:
                try:
                    start_sec = float(tgt[idx])
                    end_sec = float(tgt[idx + 1])
                except (ValueError, TypeError):
                    idx += 2
                    continue
                
                if isinstance(video_grid_thw, list):
                    if len(video_grid_thw) > 0:
                        grid_thw_to_use = video_grid_thw[0] if isinstance(video_grid_thw[0], torch.Tensor) else torch.tensor(video_grid_thw[0])
                    else:
                        idx += 2
                        continue
                elif isinstance(video_grid_thw, torch.Tensor):
                    grid_thw_to_use = video_grid_thw[0] if video_grid_thw.dim() > 1 else video_grid_thw
                else:
                    grid_thw_to_use = torch.tensor(video_grid_thw)
                
                if isinstance(audio_lengths, list):
                    if len(audio_lengths) > 0:
                        if isinstance(audio_lengths[0], list):
                            audio_lengths_to_use = audio_lengths[0]
                        else:
                            audio_lengths_to_use = audio_lengths
                    else:
                        idx += 2
                        continue
                else:
                    audio_lengths_to_use = [audio_lengths] if not isinstance(audio_lengths, (list, tuple)) else audio_lengths
                
                if isinstance(second_per_grid_ts, list):
                    if len(second_per_grid_ts) > 0:
                        if isinstance(second_per_grid_ts[0], list):
                            second_per_grid_ts_to_use = second_per_grid_ts[0]
                        else:
                            second_per_grid_ts_to_use = second_per_grid_ts
                    else:
                        idx += 2
                        continue
                elif isinstance(second_per_grid_ts, torch.Tensor):
                    second_per_grid_ts_to_use = second_per_grid_ts.tolist()
                else:
                    second_per_grid_ts_to_use = [second_per_grid_ts]
                
                token_ranges = self.get_audio_token_indices_by_time_range(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    grid_thw=grid_thw_to_use,
                    audio_lengths=audio_lengths_to_use,
                    second_per_grid_ts=second_per_grid_ts_to_use,
                    spatial_merge_size=spatial_merge_size,
                )
                
                audio_token_indices_list.append(token_ranges)
                idx += 2
        
        return audio_token_indices_list

    def compute_timestep_duration(
        self,
        second_per_grid_ts,
    ) -> Optional[float]:
        """Seconds per timestep from second_per_grid_ts (list, tensor or scalar), or None."""
        if second_per_grid_ts is None:
            return None

        # Same for all grids; take the first
        if isinstance(second_per_grid_ts, list):
            if len(second_per_grid_ts) > 0:
                if isinstance(second_per_grid_ts[0], list):
                    second_per_grid_t = second_per_grid_ts[0][0]
                else:
                    second_per_grid_t = second_per_grid_ts[0]
            else:
                return None
        else:
            second_per_grid_t = second_per_grid_ts
        
        if second_per_grid_t is None:
            return None
        
        if isinstance(second_per_grid_t, torch.Tensor):
            second_per_grid_t = second_per_grid_t.item()
        elif not isinstance(second_per_grid_t, (int, float)):
            try:
                second_per_grid_t = float(second_per_grid_t)
            except (ValueError, TypeError):
                return None
        
        return float(second_per_grid_t)

    def compute_video_audio_token_indices_per_timestep(
        self,
        video_grid_thw,
        second_per_grid_ts,
        audio_lengths: Optional[List[int]] = None,
        spatial_merge_size: int = 2,
    ) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
        """Video and audio token ranges of each temporal block (timestep).

        Returns:
            (video_ranges_per_timestep, audio_ranges_per_timestep); audio entries are [] without audio.
        """
        if isinstance(video_grid_thw, torch.Tensor):
            grid_thw = video_grid_thw.clone()
        elif isinstance(video_grid_thw, list) and video_grid_thw:
            if isinstance(video_grid_thw[0], torch.Tensor):
                grid_thw = torch.cat(
                    [g.unsqueeze(0) if g.dim() == 1 else g for g in video_grid_thw], dim=0
                )
            else:
                grid_thw = torch.tensor(video_grid_thw)
        else:
            grid_thw = torch.tensor(video_grid_thw)
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)

        if isinstance(second_per_grid_ts, list):
            second_list = (
                second_per_grid_ts[0]
                if second_per_grid_ts and isinstance(second_per_grid_ts[0], list)
                else second_per_grid_ts
            )
        else:
            second_list = (
                second_per_grid_ts.tolist()
                if hasattr(second_per_grid_ts, "tolist")
                else [float(second_per_grid_ts)]
            )
        if not isinstance(second_list, list):
            second_list = [second_list]

        audio_list = audio_lengths
        if isinstance(audio_lengths, list) and audio_lengths and isinstance(audio_lengths[0], list):
            audio_list = audio_lengths[0]
        elif audio_lengths is None:
            audio_list = []
        if not isinstance(audio_list, list):
            audio_list = [audio_list] if audio_list else []

        num_grids = grid_thw.shape[0]
        if num_grids > len(second_list):
            second_list = second_list + [second_list[-1] if second_list else 1.0] * (num_grids - len(second_list))
        if num_grids > len(audio_list):
            audio_list = audio_list + [0] * (num_grids - len(audio_list))

        video_ranges_per_timestep = []
        audio_ranges_per_timestep = []
        current_time_sec = 0.0

        for grid_idx in range(num_grids):
            t = int(
                grid_thw[grid_idx][0].item()
                if hasattr(grid_thw[grid_idx][0], "item")
                else grid_thw[grid_idx][0]
            )
            second_per_t = float(
                second_list[grid_idx].item()
                if hasattr(second_list[grid_idx], "item")
                else second_list[grid_idx]
            )
            for t_idx in range(t):
                start_sec = current_time_sec + t_idx * second_per_t
                end_sec = current_time_sec + (t_idx + 1) * second_per_t
                v_ranges = self.get_video_token_indices_by_time_range(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    grid_thw=grid_thw,
                    second_per_grid_ts=second_list,
                    spatial_merge_size=spatial_merge_size,
                )
                video_ranges_per_timestep.append(v_ranges)
                if audio_list and any(a > 0 for a in audio_list):
                    a_ranges = self.get_audio_token_indices_by_time_range(
                        start_sec=start_sec,
                        end_sec=end_sec,
                        grid_thw=grid_thw,
                        audio_lengths=audio_list,
                        second_per_grid_ts=second_list,
                        spatial_merge_size=spatial_merge_size,
                    )
                    audio_ranges_per_timestep.append(a_ranges)
                else:
                    audio_ranges_per_timestep.append([])
            current_time_sec += t * second_per_t

        return video_ranges_per_timestep, audio_ranges_per_timestep

    def token_interval(k: int, n: float) -> Tuple[float, float]:
        s = float(k) * float(n)
        e = float(k + 1) * float(n)
        return s, e
    
    def interval_distance_closed(a: float, b: float, s: float, e: float) -> float:
        """Gap between closed intervals [a, b] and [s, e]; 0 if they overlap or touch."""
        if a > b:
            a, b = b, a
        if s > e:
            s, e = e, s

        return max(0.0, a - e, s - b)
    
    def compute_tgt_timestep_distances(
        self,
        tgt: List[float],
        num_timesteps: int,
        timestep_duration: float,
    ) -> Optional[List[List[float]]]:
        """Distance (seconds) from each [start, end] pair in tgt to every timestep: [[d_0, ..., d_T-1], ...] or None."""
        if tgt is None or not isinstance(tgt, (list, tuple)) or len(tgt) < 2:
            return None
        
        if num_timesteps <= 0 or timestep_duration <= 0:
            return None
        
        distances_list = []
        
        idx = 0
        while idx < len(tgt) - 1:
            try:
                a = float(tgt[idx])
                b = float(tgt[idx + 1])
            except (ValueError, TypeError):
                idx += 2
                continue

            timestep_distances = []
            for k in range(num_timesteps):
                s, e = LazySupervisedDataset.token_interval(k, timestep_duration)
                distance = LazySupervisedDataset.interval_distance_closed(a, b, s, e)
                timestep_distances.append(distance)
            
            distances_list.append(timestep_distances)
            idx += 2
        
        return distances_list if len(distances_list) > 0 else None

    def compute_tgt_timestep_distances_from_grid(
        self,
        tgt: Optional[List[float]],
        video_grid_thw,
        timestep_duration: Optional[float],
    ) -> Optional[List[List[float]]]:
        """compute_tgt_timestep_distances with the timestep count taken from video_grid_thw."""
        if tgt is None or video_grid_thw is None or timestep_duration is None:
            return None

        if not isinstance(video_grid_thw, Sequence):
            video_grid_thw_list = [video_grid_thw]
        else:
            video_grid_thw_list = video_grid_thw
        
        if len(video_grid_thw_list) == 0:
            return None
        
        grid_thw = video_grid_thw_list[0]
        if isinstance(grid_thw, torch.Tensor):
            if grid_thw.dim() > 0:
                num_timesteps = int(grid_thw[0].item())
            else:
                num_timesteps = int(grid_thw.item())
        elif isinstance(grid_thw, (list, tuple)):
            num_timesteps = int(grid_thw[0])
        else:
            num_timesteps = int(grid_thw)
        
        return self.compute_tgt_timestep_distances(
            tgt=tgt,
            num_timesteps=num_timesteps,
            timestep_duration=timestep_duration,
        )

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3

        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        if self.data_args.run_test:
            item_to_return = self.__getitem__(random.randint(0, len(self) - 1))
            item_to_return["should_use"] = False
            return item_to_return
        else:
            print(f"Failed to fetch sample {i}. Try another sample.")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

            grid_thw_merged = None
            video_grid_thw_merged = None
            grid_thw = None
            video_grid_thw = None
            second_per_grid_ts = None
            audio = None
            audio_lengths = None

            if "image" in sources[0]:
                image_file = self.list_data_dict[i]["image"]
                if isinstance(image_file, List):
                    if len(image_file) > 1:
                        image_file = [
                            file for file in image_file
                        ]
                        results = [self.process_image_unified(file) for file in image_file]
                        image, grid_thw = zip(*results)
                    else:
                        image_file = image_file[0]
                        image, grid_thw = self.process_image_unified(image_file)
                        image = [image]
                else:
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
                grid_thw_merged = copy.deepcopy(grid_thw)
                if not isinstance(grid_thw, Sequence):
                    grid_thw_merged = [grid_thw_merged]
                    grid_thw = [grid_thw]
                grid_thw_merged = [
                    merged_thw.prod() // self.data_args.image_processor.merge_size**2
                    for merged_thw in grid_thw_merged
                ]
            if "video" in sources[0]:
                video_file = sources[0]["video"]
                if isinstance(video_file, List):
                    if len(video_file) > 1:
                        video_file = [
                            file for file in video_file
                        ]
                        results = [self.process_video(file) for file in video_file]
                        video, video_grid_thw, second_per_grid_ts = zip(*results)
                    else:
                        video_file = video_file[0]
                        video, video_grid_thw, second_per_grid_ts = self.process_video(
                            video_file
                        )
                        video = [video]
                else:
                    video, video_grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video] # [t*h*w, c]
                if "use_audio" in sources[0] and sources[0]["use_audio"]:
                    audio, audio_lengths = self.process_audio(video_file)    # [num 30s chunks, 128 mels, frames]; one token per 0.5s
                else:
                    audio, audio_lengths = None, None
                video_grid_thw_merged = copy.deepcopy(video_grid_thw)
                if not isinstance(video_grid_thw, Sequence):
                    video_grid_thw_merged = [video_grid_thw_merged]
                    video_grid_thw = [video_grid_thw]
                         
            if "audio" in sources[0]:   # audio-only samples
                audio_file = sources[0]["audio"]
                audio, audio_lengths = self.process_audio(
                    audio_file
                )
            chat_sources = copy.deepcopy([e["conversations"] for e in sources])
            gpt_value_segments = sources[0].get("gpt_value_segments", None)
            data_dict = preprocess_qwen_2_visual(   # input_ids / labels with interleaved placeholders
                chat_sources,
                self.tokenizer,
                grid_thw_image=grid_thw_merged if grid_thw_merged else None,
                grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
                audio_lengths=audio_lengths if audio_lengths else None,
                merge_size=self.data_args.image_processor.merge_size,
                second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
                gpt_value_segments=gpt_value_segments if gpt_value_segments else None
            )
            
            position_ids, _ = self.get_rope_index(  # (3, batch_size, seq_len): time / height / width
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
                video_grid_thw=(
                    torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
                ),
                second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
                audio_lengths=audio_lengths if audio_lengths else None,
            )

            # Parallel positions; the key is always set (None if single-event) since the model indexes it by batch position
            data_dict["parallel_start_idx"] = None
            if data_dict["event_id"] is not None:
                position_ids, parallel_start_idx = adjust_position_ids_for_parallel(position_ids, data_dict["event_id"], len(gpt_value_segments))
                data_dict["parallel_start_idx"] = parallel_start_idx

            if "image" not in sources[0] and "video" not in sources[0] and "audio" not in sources[0]:
                grid_thw_merged = None
                sources = copy.deepcopy([e["conversations"] for e in sources])
                data_dict = preprocess_qwen_2_visual(
                    sources, self.tokenizer, None, None
                )
                position_ids = (
                    torch.arange(0, data_dict["input_ids"].size(1))
                    .view(1, -1)
                    .unsqueeze(0)
                    .expand(3, -1, -1)
                )

            data_dict["position_ids"] = position_ids
            data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]
            if "image" in self.list_data_dict[i]:
                data_dict["pixel_values"] = torch.cat(image, dim=0)
                data_dict["image_grid_thw"] = torch.cat(
                    [thw.unsqueeze(0) for thw in grid_thw], dim=0
                )
            elif "video" in self.list_data_dict[i]:
                data_dict["pixel_values_videos"] = torch.cat(video, dim=0)  # [t*h*w, channels]
                data_dict["video_grid_thw"] = torch.cat(
                    [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
                )
            if audio is not None:
                audio = torch.cat(audio, dim=0)
            data_dict["audio_feature"] = audio
            data_dict["audio_lengths"] = audio_lengths

            # Grounding inputs (<G> path); tgt/src are kept as raw lists
            if self.use_evidence:
                timestep_duration = self.compute_timestep_duration(second_per_grid_ts=second_per_grid_ts)
                data_dict["timestep_duration"] = timestep_duration
                data_dict["tgt"] = sources[0].get("tgt", None)
                data_dict["src"] = sources[0].get("src", None)

                # Per-timestep distance to each GT interval (0 inside); soft target of the grounding loss
                tgt_timestep_distances = self.compute_tgt_timestep_distances_from_grid(
                    tgt=data_dict["tgt"],
                    video_grid_thw=video_grid_thw,
                    timestep_duration=timestep_duration,
                )
                data_dict["tgt_timestep_distances"] = tgt_timestep_distances
                
                # Jittered tgt for <G> aggregation only
                tgt_for_aggregation = self._jitter_tgt_for_aggregation(
                    data_dict["tgt"], timestep_duration, video_grid_thw)

                # Video/audio token ranges per tgt interval (relative to the first media token)
                if data_dict["tgt"] is not None and video_grid_thw is not None and second_per_grid_ts is not None:
                    video_token_indices_list = self.compute_video_token_indices_from_tgt(
                        tgt=tgt_for_aggregation,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        spatial_merge_size=self.data_args.image_processor.merge_size,
                    )
                    data_dict["video_token_indices_list"] = video_token_indices_list
                else:
                    data_dict["video_token_indices_list"] = None

                if data_dict["tgt"] is not None and video_grid_thw is not None and audio_lengths is not None and second_per_grid_ts is not None:
                    audio_token_indices_list = self.compute_audio_token_indices_from_tgt(
                        tgt=tgt_for_aggregation,
                        video_grid_thw=video_grid_thw,
                        audio_lengths=audio_lengths,
                        second_per_grid_ts=second_per_grid_ts,
                        spatial_merge_size=self.data_args.image_processor.merge_size,
                    )
                    data_dict["audio_token_indices_list"] = audio_token_indices_list
                else:
                    data_dict["audio_token_indices_list"] = None

                # Video/audio token ranges per src (query) interval
                if data_dict["src"] is not None and video_grid_thw is not None and second_per_grid_ts is not None:
                    video_token_indices_list = self.compute_video_token_indices_from_tgt(
                        tgt=data_dict["src"],
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        spatial_merge_size=self.data_args.image_processor.merge_size,
                    )
                    data_dict["video_token_indices_list_src"] = video_token_indices_list
                else:
                    data_dict["video_token_indices_list_src"] = None

                if data_dict["src"] is not None and video_grid_thw is not None and audio_lengths is not None and second_per_grid_ts is not None:
                    audio_token_indices_list = self.compute_audio_token_indices_from_tgt(
                        tgt=data_dict["src"],
                        video_grid_thw=video_grid_thw,
                        audio_lengths=audio_lengths,
                        second_per_grid_ts=second_per_grid_ts,
                        spatial_merge_size=self.data_args.image_processor.merge_size,
                    )
                    data_dict["audio_token_indices_list_src"] = audio_token_indices_list
                else:
                    data_dict["audio_token_indices_list_src"] = None

                # Video/audio token ranges per timestep
                if video_grid_thw is not None and second_per_grid_ts is not None:
                    try:
                        v_per_ts, a_per_ts = self.compute_video_audio_token_indices_per_timestep(
                            video_grid_thw=video_grid_thw,
                            second_per_grid_ts=second_per_grid_ts,
                            audio_lengths=audio_lengths,
                            spatial_merge_size=self.data_args.image_processor.merge_size,
                        )
                        data_dict["video_token_indices_per_timestep"] = v_per_ts
                        data_dict["audio_token_indices_per_timestep"] = a_per_ts
                    except Exception:
                        data_dict["video_token_indices_per_timestep"] = None
                        data_dict["audio_token_indices_per_timestep"] = None
                else:
                    data_dict["video_token_indices_per_timestep"] = None
                    data_dict["audio_token_indices_per_timestep"] = None

            # Task name (per-task inference options key on it)
            data_dict["task"] = sources[0].get('task', None)
            # Test: keep only the prompt and attach metadata
            if self.data_args.run_test:
                labels = data_dict.pop("labels", None)
                len_input = sum(labels[0] == IGNORE_INDEX)
                data_dict["input_ids"] = data_dict["input_ids"][:, :len_input]
                data_dict["position_ids"] = data_dict["position_ids"][:, :, :len_input]
                data_dict["attention_mask"] = torch.ones_like(data_dict["input_ids"])

                data_dict["video"] = sources[0].get("video", None)
                data_dict["image"] = sources[0].get("image", None)
                data_dict["audio"] = sources[0].get("audio", None)
                data_dict["use_audio"] = sources[0].get("use_audio", False)
                data_dict["second_per_grid_ts"] = second_per_grid_ts

                data_dict["prompt"] = sources[0]["conversations"][0]
                data_dict["ref"] = sources[0]["conversations"][1]["value"]
                data_dict["id"] = (
                    sources[0].get("id", None)
                    or sources[0].get("video_id", None)
                    or sources[0].get("info", None)
                )
                data_dict["info"] = sources[0].get('info', None)
                data_dict["task"] = sources[0].get('task', None)
                data_dict["video_id"] = sources[0].get('video_id', None)
                data_dict["should_use"] = sources[0].get("should_use", True)
                # Query interval in seconds, for src_interval_override
                data_dict["src_sec"] = sources[0].get("src", None)

                data_dict.pop("audio_lengths", None)

                data_dict.pop("tgt", None)
                data_dict.pop("src", None)

            return data_dict
        except Exception as e:
            print(f"Error: {e}, line: {e.__traceback__.tb_lineno}")
            raise e



def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    use_evidence: bool = False

    def process_ids(self, input_ids, labels, position_ids):
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        return input_ids, labels, position_ids, attention_mask

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids, labels, position_ids, attention_mask = self.process_ids(
            input_ids, labels, position_ids
        ) # pad and truncate to model_max_length
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        audios = list(
            instance["audio_feature"]
            for instance in instances
            if instance["audio_feature"] is not None
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        if len(audios)!= 0:
            concat_audios = torch.cat([audio for audio in audios], dim=0)
            audio_lengths = [
                instance["audio_lengths"]
                for instance in instances
                if "audio_lengths" in instance
            ]
            audio_lengths = [l for length in audio_lengths for l in length]
        else:
            concat_audios = None
            audio_lengths = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["audio_feature"] = concat_audios
        batch["audio_lengths"] = audio_lengths
        
        # Grounding fields (use_evidence)
        if self.use_evidence:
            tgt_list = [instance.get("tgt", None)  for instance in instances if "tgt" in instance]
            
            if len(tgt_list) > 0 and any(x is not None for x in tgt_list):
                batch["tgt"] = tgt_list
            else:
                batch["tgt"] = None
            
            video_token_indices_list = [
                instance.get("video_token_indices_list", None)
                for instance in instances
                if "video_token_indices_list" in instance
            ]
            if len(video_token_indices_list) > 0 and any(x is not None for x in video_token_indices_list):
                batch["video_token_indices_list"] = video_token_indices_list
            else:
                batch["video_token_indices_list"] = None
            
            audio_token_indices_list = [
                instance.get("audio_token_indices_list", None)
                for instance in instances
                if "audio_token_indices_list" in instance
            ]
            if len(audio_token_indices_list) > 0 and any(x is not None for x in audio_token_indices_list):
                batch["audio_token_indices_list"] = audio_token_indices_list
            else:
                batch["audio_token_indices_list"] = None

            video_token_indices_list_src = [
                instance.get("video_token_indices_list_src", None)
                for instance in instances
                if "video_token_indices_list_src" in instance
            ]
            if len(video_token_indices_list_src) > 0 and any(x is not None for x in video_token_indices_list_src):
                batch["video_token_indices_list_src"] = video_token_indices_list_src
            else:
                batch["video_token_indices_list_src"] = None

            audio_token_indices_list_src = [
                instance.get("audio_token_indices_list_src", None)
                for instance in instances
                if "audio_token_indices_list_src" in instance
            ]
            if len(audio_token_indices_list_src) > 0 and any(x is not None for x in audio_token_indices_list_src):
                batch["audio_token_indices_list_src"] = audio_token_indices_list_src
            else:
                batch["audio_token_indices_list_src"] = None

            timestep_duration_list = [
                instance.get("timestep_duration", None)
                for instance in instances
                if "timestep_duration" in instance
            ]
            if len(timestep_duration_list) > 0 and any(x is not None for x in timestep_duration_list):
                batch["timestep_duration"] = timestep_duration_list
            else:
                batch["timestep_duration"] = None
            
            tgt_timestep_distances_list = [
                instance.get("tgt_timestep_distances", None)
                for instance in instances
                if "tgt_timestep_distances" in instance
            ]
            if len(tgt_timestep_distances_list) > 0 and any(x is not None for x in tgt_timestep_distances_list):
                batch["tgt_timestep_distances"] = tgt_timestep_distances_list
            else:
                batch["tgt_timestep_distances"] = None

            video_token_indices_per_timestep_list = [
                instance.get("video_token_indices_per_timestep", None)
                for instance in instances
                if "video_token_indices_per_timestep" in instance
            ]
            if len(video_token_indices_per_timestep_list) > 0 and any(x is not None for x in video_token_indices_per_timestep_list):
                batch["video_token_indices_per_timestep"] = video_token_indices_per_timestep_list
            else:
                batch["video_token_indices_per_timestep"] = None

            audio_token_indices_per_timestep_list = [
                instance.get("audio_token_indices_per_timestep", None)
                for instance in instances
                if "audio_token_indices_per_timestep" in instance
            ]
            if len(audio_token_indices_per_timestep_list) > 0 and any(x is not None for x in audio_token_indices_per_timestep_list):
                batch["audio_token_indices_per_timestep"] = audio_token_indices_per_timestep_list
            else:
                batch["audio_token_indices_per_timestep"] = None

            # Indexed by batch position: keep one entry per instance (None for single-event tasks)
            event_id_list = [instance.get("event_id", None) for instance in instances]
            if any(x is not None for x in event_id_list):
                batch["event_id"] = event_id_list
            else:
                batch["event_id"] = None
            batch["task"] = [instance.get("task", None) for instance in instances]

            parallel_start_idx_list = [instance.get("parallel_start_idx", None) for instance in instances]
            if any(x is not None for x in parallel_start_idx_list):
                batch["parallel_start_idx"] = parallel_start_idx_list
            else:
                batch["parallel_start_idx"] = None

            last_text_token_indices_list = [
                instance.get("last_text_token_indices", None)
                for instance in instances
                if "last_text_token_indices" in instance
            ]
            if len(last_text_token_indices_list) > 0 and any(x is not None for x in last_text_token_indices_list):
                valid_indices = [x for x in last_text_token_indices_list if x is not None]
                if len(valid_indices) > 0:
                    if all(isinstance(x, torch.Tensor) for x in valid_indices):
                        batch["last_text_token_indices"] = torch.stack(valid_indices)
                    else:
                        batch["last_text_token_indices"] = torch.tensor(valid_indices, dtype=torch.long)
                else:
                    batch["last_text_token_indices"] = None
            else:
                batch["last_text_token_indices"] = None

        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, model_args=None
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args, model_args=model_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, use_evidence=getattr(model_args, 'use_evidence', False) if model_args else False)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
