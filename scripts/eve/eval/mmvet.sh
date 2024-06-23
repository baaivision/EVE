#!/bin/bash
CKPT_NAME=$1
CKPT_PATH=$2

python -m eve.eval.model_vqa \
    --model-path ${CKPT_PATH}/${CKPT_NAME} \
    --question-file ./playground/data/eval/mm-vet/eve-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${CKPT_NAME}.jsonl \
    --dst ./playground/data/eval/mm-vet/results/${CKPT_NAME}.json

