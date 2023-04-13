#!/bin/bash

GPU=$1
scene=$2
configs=./configs/replica/train/room_${scene}_full.txt
NOHUP_FILE=../log/dm-nerf/room/room_${scene}_full-GPU_${GPU}.out
export CUDA_VISIBLE_DEVICES=$1
nohup python3 train_replica.py --config ${configs} >$NOHUP_FILE 2>&1 &
