#!/bin/bash
# Prepare data for vanilla BlenderBot baseline (no strategy, no persona)
# Original: ESConv/codes_zcj/RUN/prepare_vanilla.sh (unchanged from ESConv)
# This baseline is NOT used directly in PAL paper comparisons,
# but kept for completeness / additional experiments.

CUDA_VISIBLE_DEVICES=0 python prepare.py \
    --config_name vanilla \
    --inputter_name vanilla \
    --train_input_file ./_reformat/train.txt \
    --max_input_length 160 \
    --max_decoder_input_length 40
