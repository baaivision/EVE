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
export GPUS_PER_NODE=1
export NNODES=1
export MASTER_PORT=12345
export CPUS_PER_TASK=32
export QUOTA=reserved

export BASE_LR=2e-4
export VE_LR=2e-4

export CKPT_PATH=checkpoints/EVEv2-stage0
export SAVE_PATH=EVEv2-stage0-rep


torchrun --nproc_per_node=$GPUS_PER_NODE --nnode=$NNODES --node_rank=0 --master_addr=$1 --master_port=$MASTER_PORT \
    eve/train/repeat_moe.py \
    --model_name_or_path ${CKPT_PATH} \
    --deepspeed ./scripts/zero2.json \
    --model_type qwen2 \
    --add_moe True \
    --moe_part layernorm-self_attn-mlp \
    --bf16 True \
    --output_dir checkpoints/${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 10 \
    --learning_rate ${BASE_LR} \
    --vision_tower_lr ${VE_LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --run_name ${SAVE_PATH} \
    2>&1 | tee logs/${SAVE_PATH}-rank$1.log