#!/bin/bash

sudo apt-get install -y gcc python3-dev python3-setuptools python3-pip python3-venv git-lfs
sudo pip3 install crcmod
sudo apt-get install -y docker-buildx
curl https://sdk.cloud.google.com > install.sh
bash install.sh --disable-prompts
sudo ln -s $HOME/google-cloud-sdk/bin/gcloud /usr/local/bin/gcloud
sudo ln -s $HOME/google-cloud-sdk/bin/gsutil /usr/local/bin/gsutil

sudo apt-get install -y docker.io
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update -y
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo apt-get install docker-buildx
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
sudo mkdir -p /abacus
