## Environment setup

1. We have tested our code with this docker environment `pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel` and NVIDIA V100 GPUs.
2. Install PyTorch: `pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0`
3. Install other dependencies: 
```sh
# install libraries
apt-get install -y \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libosmesa6-dev \
# install requirements
pip install -r requirements.txt
# install kaolin
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${YOUR_TORCH_VERSION}_${YOUR_CUDA_VERSION}.html
```
4. Git clone both TeCH and thirdparties
```sh
git clone --recurse-submodules git@github.com:huangyangyi/TeCH.git
```
5. Build modules
```sh
cd core/lib/freqencoder
python setup.py install
cd ../gridencoder
python setup.py install
cd ../../
```

6. Download necessary data for body models: `bash scripts/download_body_data.sh`. If `Username/Password Authentication Failed`, you need firstly register at [ICON](https://icon.is.tue.mpg.de/user.php) and choose "Yes" for all the projects listed in "Register for other projects".
7. Download pretrained models of MODNet: `bash scripits/download_modnets.sh`
8. Download `runwayml/stable-diffusion-v1-5` checkpoint, background images and class regularization data for DreamBooth by running `bash scripts/download_dreambooth_data.sh`, you can also try using another version of SD model, or use other images of `man` and `woman` for regularization (We simply generates these data with the SD model).

### If you still struggle with the package conflict, [YuliangXiu/TeCH](https://github.com/YuliangXiu/TeCH/blob/16188c6e3d5becd811207307b6aeb23024ef258d/docs/install.md) shows a cleaner version to setup TeCH at `Ubuntu 22.04.3 LTS, NVIDIA A100 (80G), CUDA=11.7` with anaconda. This forked repository has removed almost all the package version requirements, see [requirements.txt](https://github.com/YuliangXiu/TeCH/blob/16188c6e3d5becd811207307b6aeb23024ef258d/requirements.txt).
   