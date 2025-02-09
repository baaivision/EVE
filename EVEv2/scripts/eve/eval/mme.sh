#!/bin/bash
CKPT_NAME=$1
CKPT_PATH=$2
CONV_MODE=$3 

python -m eve.eval.model_vqa_loader \
    --model-path ${CKPT_PATH}/${CKPT_NAME} \
    --model-type $CONV_MODE \
    --question-file ./playground/data/eval/MME/eve_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${CKPT_NAME}.jsonl \
    --conv-mode $CONV_MODE

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment ${CKPT_NAME}

cd eval_tool

python calculation.py --results_dir answers/${CKPT_NAME}
