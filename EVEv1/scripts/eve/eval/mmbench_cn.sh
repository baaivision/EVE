#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT_NAME=$1
CKPT_PATH=$2
LANG="cn"
SPLIT="mmbench_dev_cn_20231003"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m eve.eval.model_vqa_mmbench \
        --model-path ${CKPT_PATH}/${CKPT_NAME} \
        --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --lang $LANG \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/mmbench/answers/$SPLIT/${CKPT_NAME}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mmbench/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

wait

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT
mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT/${CKPT_NAME}

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT/${CKPT_NAME} \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT/${CKPT_NAME} \
    --experiment merge
