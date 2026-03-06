#!/bin/bash
# Interactive chat with trained PAL model + persona extractor
# Original: PAL/codes/RUN/interact_strat.sh
# Changes: auto-detects latest run directory and best epoch from eval_log.csv
#
# Usage:
#   bash RUN/interact_strat.sh                           # auto-detect PAL checkpoint
#   bash RUN/interact_strat.sh 4                         # auto-detect run dir, use epoch 4
#   bash RUN/interact_strat.sh 4 MY_RUN_DIR              # use specific run dir + epoch
#
# NOTE: Requires a trained persona extractor checkpoint.
#       Set PERSONA_CHECKPOINT below to your checkpoint path.

BASE_DIR="./DATA/strat.strat_persona_attention_final_rebuttal"
PERSONA_CHECKPOINT="../persona_extractor/pl_root/lightning_logs/version_0/checkpoints/epoch=7-step=8384.ckpt"

# Determine run directory: use $2 if provided, otherwise pick the most recent
if [ -n "$2" ]; then
    RUN_DIR="${BASE_DIR}/$2"
else
    RUN_DIR=$(ls -dt "${BASE_DIR}"/*/  2>/dev/null | head -1)
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: No run directories found under ${BASE_DIR}/"
        exit 1
    fi
    RUN_DIR="${RUN_DIR%/}"
    echo "Auto-detected run directory: $(basename "$RUN_DIR")"
fi

# Determine epoch: use $1 if provided, otherwise find best from eval_log.csv
if [ -n "$1" ]; then
    EPOCH="$1"
    echo "Using user-specified epoch: ${EPOCH}"
else
    EVAL_LOG="${RUN_DIR}/eval_log.csv"
    if [ ! -f "$EVAL_LOG" ]; then
        echo "ERROR: eval_log.csv not found in ${RUN_DIR}. Specify epoch manually."
        exit 1
    fi
    EPOCH=$(awk -F',' 'NR>1 { if (best=="" || $4+0 < best+0) { best=$4; epoch=$1 } } END { print epoch }' "$EVAL_LOG")
    BEST_LOSS=$(awk -F',' 'NR>1 { if (best=="" || $4+0 < best+0) { best=$4 } } END { printf "%.4f", best }' "$EVAL_LOG")
    echo "Auto-detected best epoch: ${EPOCH} (val_loss=${BEST_LOSS})"
fi

PAL_CHECKPOINT="${RUN_DIR}/epoch-${EPOCH}.bin"
if [ ! -f "$PAL_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: ${PAL_CHECKPOINT}"
    exit 1
fi

if [ ! -f "$PERSONA_CHECKPOINT" ]; then
    echo "WARNING: Persona extractor checkpoint not found at ${PERSONA_CHECKPOINT}"
    echo "         Update PERSONA_CHECKPOINT in this script to the correct path."
    exit 1
fi

echo "Using PAL checkpoint: ${PAL_CHECKPOINT}"
echo "Using persona checkpoint: ${PERSONA_CHECKPOINT}"

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
