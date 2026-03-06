#!/bin/bash
# Run inference with trained PAL model
# Original: PAL/codes/RUN/infer_strat.sh
# Changes: auto-detects latest run directory and best epoch from eval_log.csv
#
# Usage:
#   bash RUN/infer_strat.sh              # auto-detect latest run + best epoch
#   bash RUN/infer_strat.sh 4            # auto-detect latest run, use epoch 4
#   bash RUN/infer_strat.sh 4 MY_RUN_DIR # use specific run dir + epoch

BASE_DIR="./DATA/strat.strat_persona_attention_final_rebuttal"

# Determine run directory: use $2 if provided, otherwise pick the most recent
if [ -n "$2" ]; then
    RUN_DIR="${BASE_DIR}/$2"
else
    RUN_DIR=$(ls -dt "${BASE_DIR}"/*/  2>/dev/null | head -1)
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: No run directories found under ${BASE_DIR}/"
        exit 1
    fi
    # Remove trailing slash
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
    # Find epoch with lowest validation loss (column 4 = freq_loss)
    EPOCH=$(awk -F',' 'NR>1 { if (best=="" || $4+0 < best+0) { best=$4; epoch=$1 } } END { print epoch }' "$EVAL_LOG")
    BEST_LOSS=$(awk -F',' 'NR>1 { if (best=="" || $4+0 < best+0) { best=$4 } } END { printf "%.4f", best }' "$EVAL_LOG")
    echo "Auto-detected best epoch: ${EPOCH} (val_loss=${BEST_LOSS})"
fi

CHECKPOINT="${RUN_DIR}/epoch-${EPOCH}.bin"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

echo "Using checkpoint: ${CHECKPOINT}"

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
