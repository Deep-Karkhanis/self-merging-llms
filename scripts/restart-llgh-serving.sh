#!/bin/bash

MODEL_NAME=$1
PRIMARY_HOST=$2
shift

# The rest of the arguments are now stored in an array
ALL_HOSTS=("$@")
shift

OTHER_HOSTS=("$@")

NUM_GPUS=${#ALL_HOSTS[@]}
NUM_OTHER_GPUS=${#OTHER_HOSTS[@]}

echo "Model: ${MODEL_NAME}"
echo "Primary host: ${PRIMARY_HOST}"
echo "Number of gpus: ${NUM_GPUS}"
echo "Other hosts: ${OTHER_HOSTS[@]}"
echo "Number of other hosts: ${NUM_OTHER_GPUS}"

pids=()
ssh $PRIMARY_HOST docker stop ${MODEL_NAME}-serving-gpu-ray &
pids+=($!)
if [[ x$NUM_OTHER_GPUS != x0 ]]; then
   for host in "${OTHER_HOSTS[@]}" ; do
       ssh $host docker stop ray-serve-${PRIMARY_HOST}-gpu-0 &
       pids+=($!)
   done
fi
for pid in "${pids[@]}"; do
   wait $pid
done
echo "Stopped existing serving and ray dockers"
sleep 3

if [[ x$NUM_OTHER_GPUS != x0 ]]; then
   pids=()
   echo "Starting secondary docker hosts"
   for host in "${OTHER_HOSTS[@]}" ; do
       ssh $host $HOME/llm-training/serving/docker-start-ray-serve.sh ${PRIMARY_HOST} &
       pids+=($!)
   done
   for pid in "${pids[@]}"; do
       wait $pid
   done
fi

echo "Sleeping before starting final serving job"
sleep 1
echo "Starting vllm for: ${MODEL_NAME} across ${NUM_GPUS} nodes"
ssh $PRIMARY_HOST $HOME/llm-training/serving/docker-start.sh $MODEL_NAME $NUM_GPUS vllm_ray
echo "Started model: ${MODEL_NAME}"
