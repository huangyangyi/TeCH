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
4. Build modules
```sh
cd core/lib/freqencoder
python setup.py install
cd ../gridencoder
python setup.py install
cd ../../
```
5. Fetch third-partiy code:
```sh
git clone https://github.com/ZHKKKe/MODNet thirdparties/MODNet
```
1. Download necessary data for body models: `bash scripts/download_body_data.sh`
> If you meet `Username/Password Authentication Failed.`, you can solve it by register for other projects in [this link](https://icon.is.tue.mpg.de/user.php).
2. Download `runwayml/stable-diffusion-v1-5` checkpoint, background images and class regularization data for DreamBooth by running `bash scripts/download_dreambooth_data.sh`, you can also try using another version of SD model, or use other images of `man` and `woman` for regularization (We simply generates these data with the SD model).
   