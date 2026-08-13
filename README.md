<h1 align="center">🎬 PadCaptioner</h1>

<h3 align="center">Parallelized Autoregressive Decoding for Omni-Modal Dense Video Captioning</h3>

<p align="center">
  Wenzheng Zeng, Siyi Jiao, Chen Gao, Hwee Tou Ng, Mike Zheng Shou
</p>

<p align="center">National University of Singapore</p>

<h3 align="center">ECCV 2026</h3>

<p align="center">
  <a href="https://arxiv.org/pdf/2607.02963">Paper</a> &nbsp; | &nbsp;
  <a href="https://huggingface.co/papers/2607.02963">HF Daily Paper</a> &nbsp; | &nbsp;
  <a href="#demo">Demo</a> &nbsp; | &nbsp;
  <a href="#news">News</a> &nbsp; | &nbsp;
  <a href="#overview">Overview</a>
</p>


## Demo

<p align="center">
  <a href="media/padcaptioner_demo.mp4">
    <img src="media/padcaptioner_demo.gif" width="90%" alt="PadCaptioner demo">
  </a>
</p>

<p align="center">
  <a href="media/padcaptioner_demo.mp4">Click here to watch the demo video in MP4 format</a>
</p>


## 📢 News

- [2026-08] The code and model have been released!
- [2026-07] Our work is featured by [DailyPapers](https://x.com/HuggingPapers/status/2076401494295826468)!
- [2026-07] Our paper is available on [arXiv](https://arxiv.org/abs/2607.02963).
- [2026-06] Our work is accepted by ECCV 2026!

## 🔆 Overview

We propose **PadCaptioner**, a 3B model for omni-modal dense video captioning that achieves high efficiency and strong grounded caption quality, outperforming 7B counterparts.

The core idea is to exploit the weak local dependencies among temporally distinct events and restructure the causal token dependency, enabling lossless parallel generation.

We design a latent planning mechanism that automatically determines parallelizable units with non-local awareness, guiding subsequent parallel decoding and improving event grounding and caption quality.


## 🛠️ Installation

1. Clone this repository:

```bash
git clone https://github.com/showlab/PadCaptioner.git
cd PadCaptioner
```

2. Create the conda environment:

```bash
conda create -n padcaptioner python=3.12
conda activate padcaptioner
```

3. Install dependencies:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```




## 📦 Dataset

Please refer to **[DATASET.md](DATASET.md)** for raw dataset links, annotation preparation, the full JSON schema, and task examples.

## 💪 Training

1. Model Preparation

   1. Prepare the [video-SALMONN 2+ 3B checkpoint](https://github.com/bytedance/video-SALMONN-2/tree/a89862ccc6b79c115053c60e3d9f62798d45a42d/video_SALMONN2_plus).

   2. Set `MODEL`, `MODEL_BASE`, and `LORA_CKPT` (download from [here](https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_3B)) in `scripts_run/train.sh` to the downloaded paths. At training start, the LoRA checkpoint is automatically merged into the base model to form the video-SALMONN 2+ starting point.

    - We notice that Video-SALMONN 2+ was subsequently updated, so its newer, already-merged model may potentially serve as a starting point. However, we have not verified the correctness of this setup.

2. Fill in the path block at the top of `scripts_run/train.sh` (`DATASET`, `MODEL`, `MODEL_BASE`, `LORA_CKPT`, `OUTPUT_ROOT`), or pass them as flags.

3. Run:

   ```bash
   bash scripts_run/train.sh
   ```

## 🤖 Inference and Evaluation

1. Download the [pretrained model](https://huggingface.co/wenzhengzeng/PadCaptioner-3B).

2. Fill in the path block at the top of `scripts_run/test.sh` (`DATASET`, `MODEL`, `MODEL_BASE`, `OUTPUT_ROOT`), or pass them as flags.

   For ChronusAV:

   ```bash
   bash scripts_run/test.sh --dataset /path/to/chronusav_test.json
   ```

## 🙏 Acknowledgments

We list below the related works that inspired PadCaptioner:

- [D<sup>2</sup>VLM](https://github.com/nusnlp/d2vlm)
- [video-SALMONN 2+](https://github.com/bytedance/video-SALMONN-2)


## 📌 Citation
If you find our work useful, please kindly cite:
  ```
  @inproceedings{padcaptioner,
    title={Parallelized Autoregressive Decoding for Omni-Modal Dense Video Captioning},
    author={Zeng, Wenzheng and Jiao, Siyi and Gao, Chen and Ng, Hwee Tou and Shou, Mike Zheng},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2026}
  }
  ```
