#!/bin/bash

DEFAULT_IMAGE="llm-trt"
IMAGE_NAME=${1:-$DEFAULT_IMAGE}
TAG=${2:-latest}

gcloud auth configure-docker
gcloud auth print-access-token | sudo docker login -u oauth2accesstoken --password-stdin https://us-docker.pkg.dev
existing=`sudo docker buildx ls | grep llm-default-builder`
if [[ x$existing == "x" ]]; then
    sudo docker buildx create --name llm-default-builder --use
fi

sudo docker buildx build --progress plain --platform linux/arm64 -t us-docker.pkg.dev/abacus-llm/llm-images/${IMAGE_NAME}:${TAG} --push --file Dockerfile.trt .
