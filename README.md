<p align="center">

  <h2 align="center">TeCH: Text-guided Reconstruction of Lifelike Clothed Humans</h2>
  <p align="center">
    <a href="https://github.com/huangyangyi"><strong>Yangyi Huang*</strong></a>
    ·  
    <a href="https://xyyhw.top/"><strong>Hongwei Yi*</strong></a>
    ·
    <a href="http://xiuyuliang.cn/"><strong>Yuliang Xiu*</strong></a>
    ·
    <a href="https://github.com/TingtingLiao"><strong>Tingting Liao</strong></a>
    ·
    <a href="https://me.kiui.moe/"><strong>Jiaxiang Tang</strong></a>
    ·
    <a href="http://www.cad.zju.edu.cn/home/dengcai/"><strong>Deng Cai</strong></a>
    ·
    <a href="https://justusthies.github.io/"><strong>Justus Thies</strong></a>
    <br>
    * Equal contribution
  </p>
  <h2 align="center">3DV 2024</h2>
  <div align="center">
    <video autoplay loop muted src="https://github.com/huangyangyi/TeCH/assets/7944350/f8fc55ed-9cbe-4b5f-bd1d-237396360713" type=video/mp4>
    </video>
  </div>

  <p align="center">
  </br>
    <a href="https://arxiv.org/abs/2308.08545">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
    <a href='https://huangyangyi.github.io/TeCH'>
      <img src='https://img.shields.io/badge/TeCH-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'></a>
    <a href="https://youtu.be/SjzQ6158Pho"><img alt="youtube views" title="Subscribe to my YouTube channel" src="https://img.shields.io/youtube/views/SjzQ6158Pho?logo=youtube&labelColor=ce4630&style=for-the-badge"/></a>
  </p>
</p>

<br/>

TeCH considers image-based reconstruction as a conditional generation task, taking conditions from both the input image and the derived descriptions. It is capable of reconstructing "lifelike" 3D clothed humans. <strong>“Lifelike”</strong> refers to 1) a detailed full-body geometry, including facial features and clothing wrinkles, in both frontal and unseen regions, and 2) a high-quality texture with consistent color and intricate patterns.
<br/>

## Installation

Please follow the [Installation Instruction](docs/install.md) to setup all the required packages.

## Getting Started

We provide a running script at `scripts/run.sh`. Before getting started, you need to set your own environment variables of `CUDA_HOME` and `REPLICATE_API_TOKEN`([get your token here](https://replicate.com/signin?next=/account/api-tokens)) in the script.

After that, you can use TeCH to create a highly detailed clothed human textured mesh from a single image, for example:

```shell
sh scripts/run.sh input/examples/name.img exp/examples/name
```

The results will be saved in the experiment folder `exp/examples/name`, and the textured mesh will be saved as `exp/examples/name/obj/name_texture.obj`

It is noted that in "Step 3", the current version of Dreambooth implementation requires 2\*32G GPU memory. And 1\*32G GPU memory is efficient for other steps. The entire training process for a subject takes ~3 hours on our V100 GPUs.

## TODOs

- [ ] Release of evaluation protocols and results data for comparison (on CAPE & THUman 2.0 datasets).
- [ ] Switch to the diffuser version of DreamBooth to save training memory.
- [ ] Further improvement of efficiency and robustness.

## Citation

```bibtex
@inproceedings{huang2024tech,
  title={{TeCH: Text-guided Reconstruction of Lifelike Clothed Humans}},
  author={Huang, Yangyi and Yi, Hongwei and Xiu, Yuliang and Liao, Tingting and Tang, Jiaxiang and Cai, Deng and Thies, Justus},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2024}
}

```
<br>

## Contributors

Kudos to all of our amazing contributors! TeCH thrives through open-source. In that spirit, we welcome all kinds of contributions from the community.

<a href="https://github.com/huangyangyi/TeCH/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=huangyangyi/TeCH" />
</a>

_Contributor avatars are randomly shuffled._

<br>

## License
This code and model are available only for **non-commercial** research purposes as defined in the LICENSE (i.e., MIT LICENSE). 
Note that, using TeCH, you have to register SMPL-X and agree with the LICENSE of it, and it's not MIT LICENSE, you can check the LICENSE of SMPL-X from https://github.com/vchoutas/smplx/blob/main/LICENSE.

## Acknowledgment
This implementation is mainly built based on [Stable Dreamfusion](https://github.com/ashawkey/stable-dreamfusion), [ECON](https://github.com/YuliangXiu/ECON), [DreamBooth-Stable-Diffusion](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion), and the BLIP API from Salesforce on [Replicate](https://replicate.com/salesforce/blip)
