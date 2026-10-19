"""Build PadCaptioner training annotations (7 tasks per video: v2t, a2t, t2v, t2a, v2a, a2v, dvc).

Input: JSON list of videos with id, duration, category, segment_num, seg_time ([start, end] in
seconds or "HH:MM:SS.fff"), audio_captions, video_captions; videos are read from <video_dir>/<id>.mp4.
Usage: python train_data_prepare.py --data_path labels.json --save_path train.json --video_dir /path/to/videos
"""
import os
import json
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", type=str, required=True, help="input JSON with per-video segment labels")
    parser.add_argument("--save_path", type=str, required=True, help="output JSON in the training schema")
    parser.add_argument("--video_dir", type=str, required=True, help="directory holding <id>.mp4 for every video")
    parser.add_argument("--seed", type=int, default=0, help="seed for the per-video segment sampling")
    return parser.parse_args()


def _timestamp_to_seconds(val):
    """Timestamp (seconds, 'MM:SS', 'HH:MM:SS' or 'HH:MM:SS.fff') -> seconds."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return 0.0
    frac = 0.0
    if "." in s:
        s, frac_str = s.split(".", 1)
        try:
            frac = float("0." + frac_str)
        except ValueError:
            pass
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = map(int, parts)
        return h * 3600 + m * 60 + sec + frac
    if len(parts) == 2:
        m, sec = map(int, parts)
        return m * 60 + sec + frac
    try:
        return float(s) + frac
    except ValueError:
        return 0.0


def seg_to_seconds(seg):
    """seg_time entry (e.g. [start, end]) -> list of seconds."""
    if seg is None:
        return []
    if isinstance(seg, (list, tuple)) and len(seg) >= 2:
        return [_timestamp_to_seconds(seg[0]), _timestamp_to_seconds(seg[1])]
    if isinstance(seg, (list, tuple)) and len(seg) == 1:
        return [_timestamp_to_seconds(seg[0])]
    return [_timestamp_to_seconds(seg)]


def ensure_1d(lst):
    """Flatten one level of nesting."""
    if lst is None or len(lst) == 0:
        return [] if lst is not None else []
    out = []
    for x in lst:
        if isinstance(x, (list, tuple)):
            out.extend(x)
        else:
            out.append(x)
    return out


def add_sample(out_list, sample, task, longvale_task, video_dir, *, tgt=None, src=None, human_value, gpt_value, gpt_value_segments=None):
    """Append one training sample; tgt/src are flattened, gpt_value_segments is dvc only."""
    vid = sample["id"]
    entry = {
        "id": vid,
        "duration": sample["duration"],
        "category": sample["category"],
        "video": os.path.join(video_dir, f"{vid}.mp4"),
        "task": task,
        "longvale_task": longvale_task,
        "use_audio": True,
        "conversations": [
            {"from": "human", "value": "<video>\n" + human_value},
            {"from": "gpt", "value": gpt_value},
        ],
    }
    entry["tgt"] = ensure_1d(tgt) if tgt is not None else []
    entry["src"] = ensure_1d(src) if src is not None else []
    if gpt_value_segments is not None:
        entry["gpt_value_segments"] = gpt_value_segments
    out_list.append(entry)


if __name__ == "__main__":
    args = parse_args()

    samples = []

    random.seed(args.seed)
    anno = json.load(open(args.data_path))

    t2a_prompt_templates = [
    "Output the audio information of the audio segment from <G>.",
    "Focus on the video event in <G> and describe what is happening from audio perspective.",
    "What was going on from <G> in the video? Reply should focus on the audio information.",
    "Tell me about the audio events from <G> in the video.",
    "Provide details about the audio events from <G> in the video.",
    "What transpired from <G> in the video, according to the audio information?"
    ]

    t2v_prompt_templates = [
    "Output the visual information of the video segment from <G>.",
    "Focus on the video event in <G> and describe what is happening from the visual perspective.",
    "What was going on from <G> in the video? Reply should focus on visual information.",
    "Tell me about the visual events from <G> in the video.",
    "Provide details about the visual events from <G> in the video.",
    "What transpired from <G> in the video, according to the visual information?"
    ]

    a2v_prompt_templates = [
    "Output the visual information of the video segment corresponding to this audio information: < {query} >.",
    "Localize the event where < {query} > and describe its visual information.",
    "Find the video segment that corresponds to the given textual query < {query} > and tell me what was going from visual perspective.",
    "Based on the given audio cue < {query} >, identify the corresponding moment in the video and describe what is visually happening at that time.",
    "When the audio indicates < {query} >, what visual scene or actions can be observed in the video? Describe the visual content.",
    "Locate the video segment associated with the audio event < {query} > and explain the visual details from an observer’s perspective."
    ]

    v2a_prompt_templates = [
    'Output the audio information of the video segment corresponding to this visual information: < {query} >.',
    'Localize the event where < {query} > and describe its audio information.',
    'Find the video segment that corresponds to the given textual query < {query} > and tell me what was going from audio perspective.',
    'Given the visual content < {query} >, find the corresponding part of the video and describe the audio that can be heard.',
    'What sounds or audio events accompany the visual scene described as < {query} > in the video?',
    'Identify the video segment where < {query} > occurs visually and infer the audio information present during that moment.'
    ]

    a2t_prompt_templates = [
    'Output the start and end time of the video segment corresponding to this audio information: < {query} >.',
    'Between which frames can we find: < {query} >?',
    'Localize the audio content described by the given textual query < {query} > in the video, and output the start and end timestamps in seconds.',
    'Detect and report the start and end timestamps of the video segment that semantically matches the given textual query < {query} >.',
    'Find the video segment that corresponds to the given textual query < {query} > and determine its start and end seconds.',
    'Give you a textual query: < {query} > When does the described content occur in the video? Please return the timestamp in seconds.'
    ]

    v2t_prompt_templates = [
    'Output the start and end time of the video segment corresponding to this video information: < {query} >.',
    'Between which frames can we find: < {query} >?',
    'Localize the visual content described by the given textual query < {query} > in the video, and output the start and end timestamps in seconds.',
    'Detect and report the start and end timestamps of the video segment that semantically matches the given textual query < {query} >.',
    'Find the video segment that corresponds to the given textual query < {query} > and determine its start and end seconds.',
    'Give you a textual query: < {query} > When does the described content occur in the video? Please return the timestamp in seconds.'
    ]

    dvc_prompt_templates = [
    "Could you please detail the events that took place during different time segments in the video? List the events in the format: From xx to xx, event1. From xx to xx, event2. ...",
    "I'd like to know what events transpired during specific time intervals in the video. Could you please elaborate?",
    "Can you go through the video and describe what took place at different time intervals?",
    "Could you outline the incidents that happened during different time periods in the video?",
    "Localize a series of activity events in the video, output the start and end timestamp for each event, and <time> - <time>, <event>. Describe each event with sentences. The output format of each predicted event should be like: “start - <time> - <time>, <event>. end seconds, event description”.",
    "Detect and report the start and end timestamps of activity events in the video, along with descriptions."
    ]

    converted_samples = []

    for sample in anno:
        segment_num = sample['segment_num']

        # Pick three segments (repeats allowed when fewer than three exist)
        if segment_num >= 3:
            rand_idx1, rand_idx2, rand_idx3 = random.sample(range(segment_num), 3)
        elif segment_num >= 1:
            rand_idx1, rand_idx2, rand_idx3 = random.choices(range(segment_num), k=3)
        else:
            rand_idx1 = rand_idx2 = rand_idx3 = 0

        seg1 = seg_to_seconds(sample["seg_time"][rand_idx1])
        v1, a1 = sample["video_captions"][rand_idx1], sample["audio_captions"][rand_idx1]
        seg2 = seg_to_seconds(sample["seg_time"][rand_idx2])
        v2, a2 = sample["video_captions"][rand_idx2], sample["audio_captions"][rand_idx2]
        seg3 = seg_to_seconds(sample["seg_time"][rand_idx3])
        v3, a3 = sample["video_captions"][rand_idx3], sample["audio_captions"][rand_idx3]

        # Single-event tasks: one <G> then serial text; no gpt_value_segments (that marks the parallel dvc path).
        # Grounding: v2t, a2t (the answer is the interval read out from <G>)
        add_sample(converted_samples, sample, "v2t", "grounding", args.video_dir, tgt=seg1,
                   human_value=random.choice(v2t_prompt_templates).format(query=v1), gpt_value="<G>")
        add_sample(converted_samples, sample, "a2t", "grounding", args.video_dir, tgt=seg1,
                   human_value=random.choice(a2t_prompt_templates).format(query=a1), gpt_value="<G>")

        # Segment captioning: t2v, t2a (the prompt's <G> carries the src interval)
        add_sample(converted_samples, sample, "t2v", "seg_captioning", args.video_dir, src=seg2, tgt=seg2,
                   human_value=random.choice(t2v_prompt_templates), gpt_value="<G>" + v2)
        add_sample(converted_samples, sample, "t2a", "seg_captioning", args.video_dir, src=seg2, tgt=seg2,
                   human_value=random.choice(t2a_prompt_templates), gpt_value="<G>" + a2)
        # dvc: tgt = all [start, end] pairs, repeated for planning and anchor <G>s
        caption = ['Visual: ' + v + ' Audio: ' +  a for v, a in zip(sample["video_captions"], sample["audio_captions"])]
        caption_answer = "<G>" * len(caption) + "<shift>" + "<G>" * len(caption) + "<shift>" + "<shift>".join(caption)
        seg_time_sec = [seg_to_seconds(s) for s in sample["seg_time"]]
        tgt_dvc_1d = [x for seg in seg_time_sec for x in seg]
        add_sample(converted_samples, sample, "dvc", "captioning", args.video_dir, tgt=tgt_dvc_1d * 2,
                   human_value=random.choice(dvc_prompt_templates), gpt_value=caption_answer, gpt_value_segments=caption)
        # Cross-modal captioning: v2a, a2v
        add_sample(converted_samples, sample, "v2a", "captioning", args.video_dir, tgt=seg3,
                   human_value=random.choice(v2a_prompt_templates).format(query=v3), gpt_value="<G>" + a3)
        add_sample(converted_samples, sample, "a2v", "captioning", args.video_dir, tgt=seg3,
                   human_value=random.choice(a2v_prompt_templates).format(query=a3), gpt_value="<G>" + v3)


    with open(args.save_path, "w") as fp:
        json.dump(converted_samples, fp, ensure_ascii=False)
    print(f"wrote {len(converted_samples)} samples from {len(anno)} videos -> {args.save_path}")


