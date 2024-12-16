# Get activations

mkdir -p ${ACT_ROOT}

for config_name in "arxiv" "stackexchange" "wikipedia" "c4" "common_crawl" 'book' 'github';
do
python moe_fication/get_activations.py \
    --dataset_name togethercomputer/RedPajama-Data-1T --dataset_config_name ${config_name} \
    --batch_size 1 \
    --save_dir ${ACT_ROOT}/${config_name}.pt
done
