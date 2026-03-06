#!/bin/bash
# Interactive chat with vanilla BlenderBot baseline
# Original: ESConv/codes_zcj/RUN/interact_vanilla.sh (unchanged from ESConv)

# UPDATE THIS: set to your actual trained checkpoint path
CHECKPOINT="./DATA/vanilla.vanilla/YOUR_RUN_DIR/best.bin"

CUDA_VISIBLE_DEVICES=0 python interact.py \
    --config_name vanilla \
    --inputter_name vanilla \
    --seed 0 \
    --load_checkpoint "$CHECKPOINT" \
    --fp16 false \
    --max_length 40 \
    --min_length 10 \
    --temperature 0.7 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1 \
    --no_repeat_ngram_size 3
