#!/bin/bash
# Run inference with trained PAL model
# Original: PAL/codes/RUN/infer_strat.sh
# Changes: replaced hardcoded absolute checkpoint path with placeholder,
#          replaced hardcoded input path with relative path

# UPDATE THIS: set to your actual trained checkpoint path
CHECKPOINT="./DATA/strat.strat_persona_attention_final_rebuttal/YOUR_RUN_DIR/epoch-4.bin"

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --config_name strat \
    --inputter_name strat \
    --add_nlg_eval \
    --seed 0 \
    --load_checkpoint "$CHECKPOINT" \
    --fp16 false \
    --max_input_length 512 \
    --max_decoder_input_length 15 \
    --max_length 50 \
    --min_length 10 \
    --infer_batch_size 128 \
    --infer_input_file ./_reformat/test.txt \
    --temperature 0.5 \
    --top_k 10 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1.03 \
    --no_repeat_ngram_size 0 \
    --use_all_persona False \
    --encode_context True
