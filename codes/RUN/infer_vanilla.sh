#!/bin/bash
# Run inference with vanilla BlenderBot baseline
# Original: ESConv/codes_zcj/RUN/infer_vanilla.sh
# Changes: auto-detects latest run directory

BASE_DIR="./DATA/vanilla.vanilla"

# Use $1 as run dir name, otherwise pick the most recent
if [ -n "$1" ]; then
    RUN_DIR="${BASE_DIR}/$1"
else
    RUN_DIR=$(ls -dt "${BASE_DIR}"/*/  2>/dev/null | head -1)
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: No run directories found under ${BASE_DIR}/"
        exit 1
    fi
    RUN_DIR="${RUN_DIR%/}"
    echo "Auto-detected run directory: $(basename "$RUN_DIR")"
fi

CHECKPOINT="${RUN_DIR}/best.bin"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

echo "Using checkpoint: ${CHECKPOINT}"

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --config_name vanilla \
    --inputter_name vanilla \
    --add_nlg_eval \
    --seed 0 \
    --load_checkpoint "$CHECKPOINT" \
    --fp16 false \
    --max_input_length 160 \
    --max_decoder_input_length 40 \
    --max_length 40 \
    --min_length 10 \
    --infer_batch_size 2 \
    --infer_input_file ./_reformat/test.txt \
    --temperature 0.7 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1 \
    --no_repeat_ngram_size 3
