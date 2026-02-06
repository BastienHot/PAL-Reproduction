#!/bin/bash
# Interactive chat with trained PAL model + persona extractor
# Original: PAL/codes/RUN/interact_strat.sh
# Changes: replaced hardcoded absolute paths with relative placeholders

# UPDATE THESE: set to your actual checkpoint paths
PAL_CHECKPOINT="./DATA/strat.strat_persona_attention_final_rebuttal/YOUR_RUN_DIR/epoch-4.bin"
PERSONA_CHECKPOINT="../persona_extractor/checkpoints/epoch=7-step=8384.ckpt"

CUDA_VISIBLE_DEVICES=0 python interact.py \
    --config_name strat \
    --inputter_name strat_interact \
    --seed 0 \
    --load_checkpoint "$PAL_CHECKPOINT" \
    --fp16 false \
    --max_length 50 \
    --min_length 10 \
    --temperature 0.7 \
    --top_k 30 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1.03 \
    --no_repeat_ngram_size 3 \
    --prepare_persona_ahead False \
    --persona_model_dir_or_name facebook/bart-large-cnn \
    --persona_ckpt "$PERSONA_CHECKPOINT"
