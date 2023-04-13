#!/bin/bash

GPU=$1
scene=$2
sparse_inv=$3
configs=./configs/replica/train/office_${scene}_sparse_inv.txt
NOHUP_FILE=../log/dm-nerf/office/office_${scene}_sparse_inv-GPU_${GPU}.out
export CUDA_VISIBLE_DEVICES=$1
nohup python3 test_replica.py --config ${configs} --label_sparse_inv ${sparse_inv} >$NOHUP_FILE 2>&1 &
