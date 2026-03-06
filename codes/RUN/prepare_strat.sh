#!/bin/bash
# Prepare data for PAL (strategy + persona) training
# Original: PAL/codes/RUN/prepare_strat.sh
# Changes: replaced hardcoded absolute path with relative path

CUDA_VISIBLE_DEVICES=0 python prepare.py \
    --config_name strat \
    --inputter_name strat \
    --train_input_file ./_reformat/train.txt \
    --max_input_length 512 \
    --max_decoder_input_length 50 \
    --use_all_persona False \
    --encode_context True \
    --single_processing 
