#!/bin/bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_HCA=mlx5_2,mlx5_5

set -x

wandb login
apt-get install -y libibverbs1
mkdir -p logs

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=4
export MASTER_PORT=23456
export CPUS_PER_TASK=32
export QUOTA=reserved

# We have no specific plan to release pretraining data. 
# However, you can download and filter images according to our paper's guidelines. 
# Then, use LLaVA-NEXT to generate high-definition image descriptions, which would provide better results.

export DATA_PATH=playground/data/EVE-Pretrain-33M/eve_pretrain_cap33m.json
export IMAGE_PATH=playground/data/EVE-Pretrain-33M
export VIT_PATH=openai/eve-patch14-anypixel-672
export VIT_PATH_CLIP=openai/clip-vit-large-patch14-336
export BASE_LR=4e-5
export LEARNIG_RATE=4e-4

export CKPT_PATH=lmsys/vicuna-7b-v1.5
export SAVE_PATH=eve-7b-prtr0-672-mse (or EVE-7B-Prealign-v1.0)


torchrun --nproc_per_node=$GPUS_PER_NODE --nnode=$NNODES --node_rank=$1 --master_addr=$2 --master_port=$MASTER_PORT \
    eve/train/train_mem.py \
    --model_name_or_path ${CKPT_PATH} \
    --deepspeed ./scripts/zero3.json \
    --version plain \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_PATH} \
    --vision_tower ${VIT_PATH} \
    --vision_tower_clip ${VIT_PATH_CLIP} \
    --requires_cliploss True \
    --tune_vision_tower True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir checkpoints/${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 7813 \
    --save_total_limit 10 \
    --learning_rate ${BASE_LR} \
    --mm_projector_lr ${LEARNIG_RATE} \
    --vision_tower_lr ${LEARNIG_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${SAVE_PATH} \
    2>&1 | tee logs/${SAVE_PATH}-rank$1-$(date "+%Y-%m-%d|%H:%M:%S").log