#!/bin/bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_HCA=mlx5_2,mlx5_5
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

set -x

wandb offline
apt-get install -y libibverbs1
apt-get install -y libaio-dev

mkdir -p logs

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=16
export MASTER_PORT=12345
export CPUS_PER_TASK=32
export QUOTA=reserved

export DATA_PATH=playground/data/EVE-v2.0-SFT/infinity-mm-llava-ov-index.json
export JSON_PATH=playground/data/EVE-v2.0-SFT/
export IMAGE_PATH=playground/data/EVE-v2.0-SFT/image_path
export VIT_PATH=openai/eve-anyratio-res1600-patch16
export PRE_LEN=playground/data/EVE-v2.0-SFT/conversation_presave_lengths.json # option to save loading time
export PRE_MOLEN=playground/data/EVE-v2.0-SFT/conversation_presave_modality_lengths.json # option to save loading time

export BASE_LR=1e-5
export LEARNIG_RATE=1e-5

export CKPT_PATH=checkpoints/EVEv2-stage2
export SAVE_PATH=EVE-7B-HD-v2.0


torchrun --nproc_per_node=$GPUS_PER_NODE --nnode=$NNODES --node_rank=$1 --master_addr=$2 --master_port=$MASTER_PORT \
    eve/train/train_mem.py \
    --model_name_or_path ${CKPT_PATH} \
    --deepspeed ./scripts/zero3.json \
    --version qwen2 \
    --model_type qwen2 \
    --data_path ${DATA_PATH} \
    --json_path ${JSON_PATH} \
    --image_folder ${IMAGE_PATH} \
    --vision_tower ${VIT_PATH} \
    --group_by_modality_length True \
    --presave_lengths ${PRE_LEN} \
    --presave_modality_lengths ${PRE_MOLEN} \
    --add_moe True \
    --moe_part layernorm-self_attn-mlp \
    --tune_LLM True \
    --tune_VE True \
    --tune_MOE True \
    --bf16 True \
    --output_dir checkpoints/${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 10 \
    --learning_rate ${BASE_LR} \
    --vision_tower_lr ${LEARNIG_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 5000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${SAVE_PATH} \
    2>&1 | tee logs/${SAVE_PATH}-rank$1.log