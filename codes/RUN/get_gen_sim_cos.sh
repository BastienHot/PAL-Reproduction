#!/bin/bash
# Compute SimCSE cosine similarity between generated responses and persona
# Original: PAL/codes/RUN/get_gen_sim_cos.sh
# Changes: auto-detects latest gen.json if no argument provided
#
# Usage:
#   bash RUN/get_gen_sim_cos.sh                   # auto-detect latest gen.json
#   bash RUN/get_gen_sim_cos.sh <path_to_gen.json> # use specific file

BASE_DIR="./DATA/strat.strat_persona_attention_final_rebuttal"

if [ -n "$1" ]; then
    INPUT_FILE="$1"
else
    # Find the most recently modified gen.json under the latest run directory
    INPUT_FILE=$(find "$BASE_DIR" -name "gen.json" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -z "$INPUT_FILE" ]; then
        echo "ERROR: No gen.json found under ${BASE_DIR}/. Run inference first or specify path manually."
        exit 1
    fi
    echo "Auto-detected: ${INPUT_FILE}"
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: File not found: ${INPUT_FILE}"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python get_cos_similarity.py \
    --input_file "$INPUT_FILE"
