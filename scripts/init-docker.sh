#!/bin/bash

source /repos/llm-training/scripts/_ray_utils
maybe_init_ray
if [[ -f /usr/sbin/sshd ]] ; then
   /usr/sbin/sshd -D
fi
sleep infinity
