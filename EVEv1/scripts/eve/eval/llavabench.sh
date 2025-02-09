#!/bin/bash
CKPT_NAME=$1
CKPT_PATH=$2

python -m eve.eval.model_vqa \
    --model-path ${CKPT_PATH}/${CKPT_NAME} \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/resources/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/resources/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python eve/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/resources/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/resources/context.jsonl \
    --rule eve/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/resources/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/${CKPT_NAME}.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/${CKPT_NAME}-eval1.jsonl

python eve/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/${CKPT_NAME}-eval1.jsonl
