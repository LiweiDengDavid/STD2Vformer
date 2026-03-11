#!/usr/bin/env bash
set -euo pipefail

# Free-Form prediction (WITH retrain):
# - pretrain at --pred_len
# - finetune one checkpoint per horizon in --pred_len_test

cd "$(dirname "$0")/.."

MODEL_NAME="STD2Vformer"
DATA_NAME="METR-LA"
M=8

for lr in 0.0005 0.0001
 do
  python -u main.py \
    --model_name "$MODEL_NAME" \
    --train True \
    --resume False \
    --exp_name deep_learning \
    --data_name "$DATA_NAME" \
    --points_per_hour 12 \
    --lr "$lr" \
    --M "$M" \
    --pred_len 12 \
    --pred_len_test [6,9,12,24,36] \
    --flexible True \
    --retrain True \
    --alpha 0.0 \
    --is_no_blind False \
    --batch_size 16 \
    --patience 10 \
    --dp_mode False \
    --resume_dir None \
    --output_dir None \
    --info "Flexible-Retrain,is_no_blind=False"
 done
