#!/bin/bash

# Run after logging in with ssh forwarding and:
# git clone git@github.com:abacusai/llm-training

cd

ln -s ~/llm-training/config/dotfiles/bin .
mv .bashrc .bashrc.bkp
for f in bashrc bash_profile emacs screenrc ;  do
    ln -s ~/llm-training/config/dotfiles/$f .$f
done
cp ~/llm-training/config/ssh_config.llgh ~/.ssh/config
chmod 600 ~/.ssh/config

PATH=$PATH:/home/ubuntu/google-cloud-sdk/bin
gcloud auth login
gcloud config set project abacus-llm
