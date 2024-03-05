#!/bin/bash

DATASET_NAME=${1:-20230123}

mkdir -p training_data/${DATASET_NAME}
cd training_data/${DATASET_NAME}
gsutil -m cp -r gs://llm01/training_data/${DATASET_NAME}/*.pt .
cd
