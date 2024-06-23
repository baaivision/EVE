#!/bin/bash
CKPT_NAME=$1
CKPT_PATH=$2
SPLIT="dev"

python -m eve.eval.model_vqa_qbench \
    --model-path ${CKPT_PATH}/${CKPT_NAME} \
    --image-folder ./playground/data/eval/qbench/images_llvisionqa/ \
    --questions-file ./playground/data/eval/qbench/llvisionqa_${SPLIT}.json \
    --answers-file ./playground/data/eval/qbench/llvisionqa_${SPLIT}_${CKPT_NAME}_answers.jsonl \
    --conv-mode eve_v1 \
    --lang en

python playground/data/eval/qbench/format_qbench.py \
    --filepath ./playground/data/eval/qbench/llvisionqa_${SPLIT}_${CKPT_NAME}_answers.jsonl

python playground/data/eval/qbench/qbench_eval.py \
    --filepath ./playground/data/eval/qbench/llvisionqa_${SPLIT}_${CKPT_NAME}_answers.jsonl