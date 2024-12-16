lr1=$1
lr2=$2
embedding_dim=$3
inter_dim=$3
layer=$4
head=$5

torchrun --nproc-per-node=8 --rdzv-endpoint=localhost:29500 moe_fication/train_red.py \
        --dataset_name togethercomputer/RedPajama-Data-1T-sample \
        --batch_size 2 \
        --N_cluster 7 \
        --mask_root ${MASK_ROOT}/mask.pt \
        --train_predictor \
        --learning_rate ${lr1} \
        --max_train_steps 100 \
        --embedding_dim ${embedding_dim} --inter_dim ${inter_dim} \
        --gradient_accumulation_steps 16 \
        --lr_scheduler_type cosine \
        --save output/train_clusterer/red_embed${embedding_dim}_inter${inter_dim}_${layer}layer_${head}head \
        --n_layer ${layer} --n_head ${head} --attn_normalizer 1 \
        --eval_interval 20 
        --modify_lr_scheduler


python -m accelerate.commands.launch --main_process_port 26000 --config_file default_config.yaml moe_fication/train_red.py \
        --dataset_name togethercomputer/RedPajama-Data-1T-sample \
        --batch_size 1 \
        --N_cluster 7 \
        --mask_root ${MASK_ROOT}/mask.pt \
        --finetune \
        --learning_rate ${lr2} \
        --max_train_steps 200 \
        --embedding_dim ${embedding_dim} --inter_dim ${inter_dim} \
        --gradient_accumulation_steps 32 \
        --lr_scheduler_type cosine \
        --ckpt output/train_clusterer/red_embed${embedding_dim}_inter${inter_dim}_${layer}layer_${head}head \
        --save output/finetune/red_embed${embedding_dim}_inter${inter_dim}_${layer}layer_${head}head_iter0 \
        --n_layer ${layer} --n_head ${head} --attn_normalizer 1 \
        --eval_interval 50 

for iter in 2 3 4 5;
do 
    iter_pre=$(($iter-1))
    torchrun --nproc-per-node=8 --rdzv-endpoint=localhost:29500 moe_fication/train_red.py \
            --dataset_name togethercomputer/RedPajama-Data-1T-sample \
            --batch_size 2 \
            --N_cluster 7 \
            --mask_root ${MASK_ROOT}/mask.pt \
            --train_predictor \
            --learning_rate ${lr1} \
            --max_train_steps 100 \
            --embedding_dim ${embedding_dim} --inter_dim ${inter_dim} \
            --gradient_accumulation_steps 16 \
            --lr_scheduler_type cosine \
            --model output/finetune/red_embed${embedding_dim}_inter${inter_dim}_${layer}layer_${head}head_iter${iter_pre} \
            --save output/train_clusterer/red_embed${embedding_dim}_inter${inter_dim}_${layer}layer_${head}head_iter${iter} \
            --n_layer ${layer} --n_head ${head} --attn_normalizer 1 \
            --eval_interval 50 \
            --modify_lr_scheduler


    python -m accelerate.commands.launch --main_process_port 26000 --config_file default_config.yaml moe_fication/train_red.py \
            --dataset_name togethercomputer/RedPajama-Data-1T-sample \
            --batch_size 1 \
            --N_cluster 7 \
            --mask_root ${MASK_ROOT}/mask.pt \
            --finetune \
            --learning_rate ${lr2} \
            --max_train_steps 200 \
            --embedding_dim ${embedding_dim} --inter_dim ${inter_dim} \
            --gradient_accumulation_steps 32 \
            --lr_scheduler_type cosine \
            --ckpt output/train_clusterer/red_embed${embedding_dim}_inter${inter_dim}_${layer}layer_${head}head_iter${iter} \
            --save output/finetune/red_embed${embedding_dim}_inter${inter_dim}_${layer}layer_${head}head_iter${iter} \
            --n_layer ${layer} --n_head ${head} --attn_normalizer 1 \
            --eval_interval 50 
done