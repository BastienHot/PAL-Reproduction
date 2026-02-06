#!/bin/bash
# Compute SimCSE cosine similarity between generated responses and persona
# Original: PAL/codes/RUN/get_gen_sim_cos.sh
# Changes: replaced hardcoded absolute path with CLI argument

# UPDATE THIS: set to the gen.json output from inference
INPUT_FILE="./DATA/strat.strat_persona_attention_final_rebuttal/YOUR_RUN_DIR/YOUR_INFER_DIR/gen.json"

CUDA_VISIBLE_DEVICES=0 python get_cos_similarity.py \
    --input_file "$INPUT_FILE"
