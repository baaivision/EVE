CKPT_NAME='EVE-7B-HD-v1_0'
CKPT_PATH='BAAI'
mkdir log_results

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eve/eval/seed.sh ${CKPT_NAME} ${CKPT_PATH}  2>&1 | tee log_results/${CKPT_NAME}_seed
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eve/eval/gqa.sh ${CKPT_NAME} ${CKPT_PATH}  2>&1 | tee log_results/${CKPT_NAME}_gqa
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eve/eval/sqa.sh ${CKPT_NAME} ${CKPT_PATH}  2>&1 | tee log_results/${CKPT_NAME}_sqa
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eve/eval/vqav2.sh ${CKPT_NAME} ${CKPT_PATH}  2>&1 | tee log_results/${CKPT_NAME}_vqav2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eve/eval/textvqa.sh ${CKPT_NAME} ${CKPT_PATH}  2>&1 | tee log_results/${CKPT_NAME}_textvqa
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eve/eval/mmbench_en.sh ${CKPT_NAME} ${CKPT_PATH}  2>&1 | tee log_results/${CKPT_NAME}_mmbench_en
CUDA_VISIBLE_DEVICES=0 bash scripts/eve/eval/mme.sh ${CKPT_NAME} ${CKPT_PATH}  2>&1 | tee log_results/${CKPT_NAME}_mme
CUDA_VISIBLE_DEVICES=1 bash scripts/eve/eval/mmvet.sh ${CKPT_NAME} ${CKPT_PATH}  2>&1 | tee log_results/${CKPT_NAME}_mmvet
CUDA_VISIBLE_DEVICES=2 bash scripts/eve/eval/pope.sh ${CKPT_NAME} ${CKPT_PATH}  2>&1 | tee log_results/${CKPT_NAME}_pope
CUDA_VISIBLE_DEVICES=3 bash scripts/eve/eval/vizwiz.sh ${CKPT_NAME} ${CKPT_PATH}  2>&1 | tee log_results/${CKPT_NAME}_vizwiz
