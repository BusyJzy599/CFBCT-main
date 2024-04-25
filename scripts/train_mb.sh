
BASE_DIR="/mnt/jzy8T/jzy/TCGA"
DATASET="BLCA"

CUDA_VISIBLE_DEVICES=1 python main.py \
--data_root_dir ${BASE_DIR}/${DATASET} \
--split_dir tcga_${DATASET,,} \
--model_type mb \
--which_splits 5foldcv \
--apply_sig false \
--log_data false \