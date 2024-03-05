#!/bin/bash

PORT=${JUPYTER_PORT:-10000}

jupyter lab --ip 0.0.0.0 --no-browser --allow-root --port=$PORT &
sleep 5
echo Please use the following url to connect to your notebook from a VPN enabled computer:
INTERNAL_SUFFIX=""
EXTERNAL_HOST=`hostname -s`
if [[ $EXTERNAL_HOST != *llgh* ]]; then
    INTERNAL_SUFFIX="-internal"
fi
jupyter notebook list | grep 0.0.0 | tail -n 1 | awk '{print $1}' | sed "s/0.0.0.0/${EXTERNAL_HOST}${INTERNAL_SUFFIX}.reai.io/"
