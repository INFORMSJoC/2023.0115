#!/bin/bash
gpu=$1
load_path=$2
hidden_size=$3
dataset=$4
secret_dataset=$5


python scripts/link_prediction.py --gpu $gpu --dataset $dataset --secret_dataset $secret_dataset --hidden-size $hidden_size --emb-path "$load_path/$dataset.npy"