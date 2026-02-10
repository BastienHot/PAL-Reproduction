#!/bin/bash
# Train PAL model (strategy + persona)
# Original: PAL/codes/RUN/train_strat.sh
# Changes: replaced hardcoded absolute path with relative path
#
# NOTE: Paper (Section 4.3) reports lr=2.5e-5 and warmup_steps=100,
#       but the code default is lr=1.5e-5 and warmup_steps=0.
#       We use the code values because they gave better results in our
#       reproduction experiments (see CHANGES.md and GitHub issue #11).

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config_name strat \
    --inputter_name strat \
    --eval_input_file ./_reformat/valid.txt \
    --seed 13 \
    --max_input_length 512 \
    --max_decoder_input_length 50 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 16 \
    --learning_rate 1.5e-5 \
    --num_epochs 10 \
    --warmup_steps 0 \
    --fp16 false \
    --loss_scale 0.0 \
    --pbar true \
    --use_all_persona False \
    --encode_context True
