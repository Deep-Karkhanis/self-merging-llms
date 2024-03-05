#!/bin/bash

# Example:
# Run script using:
#     ~/llm-training/scripts/lambdadocker-default.sh 0,1
#
# To reserve a docker container with gpus 0,1
#

NUM_SYSTEM_GPUS=`nvidia-smi -L | wc -l`
if [[ $NUM_SYSTEM_GPUS == "1" ]]; then
    DEFAULT_GPU=0
fi
GPU=${1:-$DEFAULT_GPU}
TAG=${2:-latest}
MEMORY=${3:-30}
RAY_ENV=""
IPC_ARGS=""
if [[ x$GPU != "xnone" ]]; then
    IFS=',' read -r GPU_0 IGNORE <<< "$GPU"
    GPU_STR=`echo $GPU | tr ',' '_'`
    GPU_BINDINGS=-"-gpus \"device=$GPU\""
else
    GPU_0=20
fi

JUPYTER_PORT=`expr 10000 + $UID + $GPU_0 \* 10`
echo JUPYTER_PORT=$JUPYTER_PORT
SOURCE_DIR="${BASH_SOURCE%/*}"
REPOS_ROOT=$(realpath ${SOURCE_DIR}/../../)
echo REPOS_ROOT=$REPOS_ROOT

source $REPOS_ROOT/llm-training/scripts/_docker_utils

echo "---------------------------------------------------------"
echo "To run notebooks within this docker use:                 "
echo "    /repos/llm-training/scripts/start-docker-jupyter.sh   "
echo "---------------------------------------------------------"

DOCKER_NAME=${USER}-training-gpu-${GPU_STR}

existing=`docker ps | grep $DOCKER_NAME`
if [[ x$existing = "x" ]]; then
    echo "Starting new docker container"
    docker rm /$DOCKER_NAME 2> /dev/null || true

    init_for_docker_start

    MOUNTS=""
    if [[ -d /data ]]; then
	MOUNTS="--mount type=bind,source=/data,target=/data"
    fi
    if [[ -d /sharedfs ]]; then
	MOUNTS="$MOUNTS \
           --mount type=bind,source=/sharedfs,target=/sharedfs"
        IPC_ARGS="--privileged \
                  --ipc host "
    fi

    MOUNTS="$MOUNTS \
        --mount type=bind,source=/abacus,target=/abacus \
	--mount type=bind,source=$REPOS_ROOT,target=/repos"

    if [[ x$GPU = "xray" ]]; then
        GPU_BINDINGS="-gpus \"device=0\""
        RAY_ENV="--env USE_RAY=true"
    fi
    docker run --shm-size=${MEMORY}g \
         --name $DOCKER_NAME \
         $MOUNTS \
         --env PYTHONPATH=/repos/llm-training \
         --env JUPYTER_PORT=$JUPYTER_PORT \
         $RAY_ENV \
         $GPU_BINDINGS \
         --detach \
	 $IPC_ARGS \
         --network host \
         --entrypoint /bin/bash ${DOCKER_IMAGE} /repos/llm-training/scripts/init-docker.sh 2>&1 > /dev/null &
    echo "Wating 5 seconds for startup"
    sleep 5
fi

docker exec -it $DOCKER_NAME /bin/bash
