# Get Mask
mkdir -p ${MASK_ROOT}

python moe_fication/generate_mask.py \
    --data_root ${ACT_ROOT} \
    --mask_root ${MASK_ROOT} \
    --N_cluster 7 \
    --remain_rate 0.5 \
    --N_feature 11008