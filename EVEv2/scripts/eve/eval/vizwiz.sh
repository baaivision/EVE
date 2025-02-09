#!/bin/bash
CKPT_NAME=$1
CKPT_PATH=$2
CONV_MODE=$3 

python -m eve.eval.model_vqa_loader \
    --model-path ${CKPT_PATH}/${CKPT_NAME} \
    --model-type $CONV_MODE \
    --question-file ./playground/data/eval/vizwiz/eve_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${CKPT_NAME}.jsonl \
    --conv-mode $CONV_MODE

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/eve_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${CKPT_NAME}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${CKPT_NAME}.json
