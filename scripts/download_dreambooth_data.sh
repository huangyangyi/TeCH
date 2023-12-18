#!/bin/bash
mkdir -p data/dreambooth_data

# SD v1-5 LDM checkpoint
echo -e "\nDownloading stable diffusion v1.5..."
wget 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt' -O data/dreambooth_data/v1-5-pruned.ckpt

# ECON
echo -e "\nDownloading dreambooth background images and regularization images..."
wget 'https://www.dropbox.com/scl/fi/ucj961vt90hix12up2nyv/dreambooth_data.zip?rlkey=w1frc8hzkjskmnesokextp84r&dl=0' -O 'data/dreambooth_data/dreambooth_data.zip' --no-check-certificate --continue
cd data/dreambooth_data && unzip dreambooth_data.zip
rm -f dreambooth_data.zip