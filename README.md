<h1 align="center">PadCaptioner</h1>

<h3 align="center">Parallelized Autoregressive Decoding for Omni-Modal Dense Video Captioning</h3>

<p align="center">
  Wenzheng Zeng, Siyi Jiao, Chen Gao, Hwee Tou Ng, Mike Zheng Shou
</p>

<p align="center">National University of Singapore</p>

<h3 align="center">ECCV 2026</h3>

<p align="center">
  <a href="#demo">Demo</a> &nbsp; | &nbsp;
  <a href="#news">News</a> &nbsp; | &nbsp;
  <a href="#overview">Overview</a>
</p>


## Demo

<p align="center">
  <video src="media/padcaptioner_demo.mp4" controls width="90%"></video>
</p>

<p align="center">
  <a href="media/padcaptioner_demo.mp4">Click to play the demo video on GitHub</a>
</p>


## 📢 News

- [2026-06] Our work is accepted by ECCV 2026!
- [2026-06] 🚧 The paper and code will be released soon. Please give us a ⭐ to stay updated!

## 🔆 Overview

We propose **PadCaptioner**, a 3B model for omni-modal dense video captioning that achieves high efficiency and strong grounded caption quality, outperforming 7B counterparts.

The core idea is to exploit the weak local dependencies among temporally distinct events and restructure the causal token dependency, enabling lossless parallel generation.

We design a latent planning mechanism that automatically determines parallelizable units with non-local awareness, guiding subsequent parallel decoding and improving event grounding and caption quality.


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
