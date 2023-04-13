#!/bin/bash

GPU=$1
scene=$2
configs=./configs/replica/train/office_${scene}_sparse_inv.txt
NOHUP_FILE=../log/dm-nerf/office_${scene}_sparse_inv-GPU_${GPU}.out
CUDA_VISIBLE_DEVICES=$1 python3 train_replica.py --config ${configs} >$NOHUP_FILE 2>&1 &
