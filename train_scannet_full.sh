#!/bin/bash

GPU=$1
scene=$2
# 010 113 38

configs=./configs/scannet/train/scene0${scene}_00_full.txt
NOHUP_FILE=../log/dm-nerf/scannet/scene0${scene}_00_full-GPU_${GPU}.out
export CUDA_VISIBLE_DEVICES=$1
nohup python3 train_scannet.py --config ${configs} >$NOHUP_FILE 2>&1 &

