#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_NAME=$1
CKPT_PATH=$2
CONV_MODE=$3 

for IDX in $(seq 0 $((CHUNKS-1))); do
CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m eve.eval.model_vqa_science \
    --model-path ${CKPT_PATH}/${CKPT_NAME} \
    --model-type $CONV_MODE \
    --question-file ./playground/data/eval/scienceqa/eve_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --single-pred-prompt \
    --conv-mode $CONV_MODE &
done

wait

output_file=./playground/data/eval/scienceqa/answers/${CKPT_NAME}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/scienceqa/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python eve/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${CKPT_NAME}/merge.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT_NAME}/output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT_NAME}/result.json
