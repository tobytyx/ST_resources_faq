#!/bin/bash
CUDA_VISIBLE_DEVICES=2 nohup python auto_train.py \
--record_id $1\
--name $2\
--domain $3\
--data_path $4>> /dev/null 2>&1 &
