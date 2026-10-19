# Dataset

## 1. Download Raw Dataset for Official Project

| Dataset Name | Resources | Usage |
|---|---|---|
| ChronusAV | [Hugging Face](https://huggingface.co/datasets/mxxxxxxxxxxxxxxxxx/ChronusAV) | Train + eval |
| LongVALE | [GitHub](https://github.com/ttgeng233/LongVALE) | Eval |

For training, we use an 18K subset sampled from the ChronusAV training set. The annotation files we used are released on [Hugging Face](https://huggingface.co/datasets/wenzhengzeng/PadCaptioner-Data): `ChronusAV/train18k.json` (training, 18K videos x 7 tasks) and `ChronusAV/test.json` (evaluation, 2000 videos x 7 tasks). `ChronusAV/test_dvc2000.json` is the dense-video-captioning subset of `test.json` (the 2000 `dvc` entries) for evaluating that task alone. All of them are already in the training/evaluation format described below; only the video paths need to be rewritten (Section 2). The videos themselves come from the ChronusAV dataset above.

## 2. Pointing the Annotations at Your Videos

The released `train18k.json` / `test.json` carry absolute `video` paths from the machine
they were built on, so they will not load anywhere else. Rewrite them for your own
download first:

```bash
python my_tools/relocate_video_paths.py \
    --anno /path/to/train18k.json /path/to/test.json \
    --video_dir /path/to/my/videos \
    --out_dir /path/to/my/annotations
```




## 3. More explanation about the structure of the annotation file


### Key fields

| Field | Description |
|---|---|
| `id` | video id; the video file is `<video_dir>/<id>.mp4` |
| `duration` | video length in seconds |
| `category` | category defined within ChronusAV |
| `video` | absolute path to the video file |
| `task` | one of the task names (detailed later) |
| `video_id` | evaluation files only: `<task>_<id>` (e.g. `dvc_BfFu0SDGglA`); test-time predictions are keyed by it (resume, metric computation). Training files do not need it |
| `longvale_task` | coarse task family kept from the original annotations: `grounding`, `seg_captioning`, or `captioning`; not read by the training or inference code |
| `use_audio` | always `true` (omni-modal input) |
| `conversations` | one human turn (prompt, prefixed with `<video>\n`) and one gpt turn (target) |
| `tgt` | flattened ground-truth interval list in seconds. For `dvc` it is **repeated twice** (once for the global planning, once for the planned global tokens `<G>` as anchor condition): `[s1,e1,...,sK,eK, s1,e1,...,sK,eK]`. The six single-event tasks hold the interval once: `[s, e]` |
| `src` | segment-captioning tasks only: the given `[start, end]` interval injected into the prompt's `<G>` |
| `gpt_value_segments` | `dvc` only — per-event caption strings; its presence routes the sample through the parallel-decoding path |

### Special tokens

- `<G>` — global event token. Every task uses it; its embedding carries the event's aggregated interval feature.
- `<S>` — `dvc` only: switch token between the global planning stage and the parallel decoding stage. It never appears in the annotations: during training the model inserts it into the labels after the planning `<G>`s, and at inference the model generates it itself. Single-event tasks never produce it.
- `<shift>` — `dvc` only: preprocessing-time delimiter separating planning stage and parallelized decoding stage. It will be removed from the token sequence.

### Task Definitions

The training annotation covers the DVC task and the other six tasks defined in the ChronusAV training set, resulting in seven tasks in total.

#### Example of `dvc` (dense video captioning)

```json
{
  "id": "<id>",
  "duration": 29.32,
  "category": "example_category",
  "video": "/path/to/<id>.mp4",
  "task": "dvc",
  "longvale_task": "captioning",
  "use_audio": true,
  "conversations": [
    {"from": "human", "value": "<video>\nCould you outline the incidents that happened during different time periods in the video?"},
    {"from": "gpt", "value": "<G><G><shift><G><G><shift>Visual: ... Audio: ...<shift>Visual: ... Audio: ..."}
  ],
  "tgt": [0.0, 19.98, 20.02, 29.32, 0.0, 19.98, 20.02, 29.32],
  "src": [],
  "gpt_value_segments": ["Visual: ... Audio: ...", "Visual: ... Audio: ..."]
}
```

For a video with K events (K = 2 shown), the gpt turn is: K planning `<G>` + `<shift>` + K anchor `<G>` + `<shift>` + the K captions joined by `<shift>`. The `<shift>` token is just a preprocessing-time delimiter separating planning stage and parallelized decoding stage. It will be removed from the token sequence. `<S>` is not part of the annotation; the model adds it to the training labels after the planning `<G>`s.

At inference the model serially plans the K events (`<G>`×K then `<S>`), then decodes all K captions in parallel, conditioned on per-event anchors.

#### Evaluation files

Evaluation annotations use the same fields plus `video_id`. For `dvc` the gpt turn holds only the reference captions, concatenated in event order with each caption starting with `Visual:` and no `<G>` / `<shift>` tokens; `tgt` holds the matching intervals (the same repeated layout as in training). The metric script (`my_tools/etbench_dvc_metrics.py`) splits the reference on `Visual:` and pairs the pieces with `tgt`, so the number of `Visual:` segments must equal the number of events.



------------------------------

##  [Optional] Convert original ChronusAV annotations to our training format

Run `my_tools/code_for_chronusav_data_prepare/train_data_prepare.py` to optionally convert the original ChronusAV annotations into the training annotation format used by PadCaptioner:

```bash
python my_tools/code_for_chronusav_data_prepare/train_data_prepare.py \
    --data_path /path/to/chronusAV_train.json \
    --save_path /path/to/train_annotations.json \
    --video_dir /path/to/videos
```