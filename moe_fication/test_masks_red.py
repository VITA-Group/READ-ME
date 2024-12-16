import os 
import sys

sys.path.insert(1, 'src')

import math
import torch
import datasets
import argparse
from itertools import chain
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, AutoConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_name")
    parser.add_argument("--dataset_config_name")
    parser.add_argument("--N_cluster", type=int)
    parser.add_argument("--cluster_id_test", type=int, help="map all tokens to the cluster id")
    parser.add_argument("--mask_root", type=str)
    parser.add_argument('--eval_iter', type=int, default=None)
    parser.add_argument('--output_file', default=None)

    parser.add_argument("--test_mode", choices=['base', 'moe'])
    parser.add_argument("--ckpt")
    parser.add_argument("--convert_masked_dense_to_moe", action='store_true')

    # arguments for cluster predictor
    parser.add_argument('--hidden_act', default='relu')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--inter_dim', type=int, default=64)

    args = parser.parse_args()
    return args

def main(args):

    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(
                args.model, use_fast=True, trust_remote_code=True,
                token="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"
            )
    
    if args.dataset_name is not None:
        data_arrow_dict = {}

        if args.dataset_config_name in data_arrow_dict.keys():
            data_arrow_file_root = data_arrow_dict[args.dataset_config_name]
            arr_files = [fn for fn in os.listdir(data_arrow_file_root) if fn.endswith('.arrow')]
            data_files = {
                'train': sorted([os.path.join(data_arrow_file_root, fn) for fn in arr_files if '-train' in fn and 'NNNNN' not in fn])[:1]}
            dataset = load_dataset("arrow", data_files=data_files)['train']
        else:
            dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=f'train[:{args.dataset_per}%]')
            dataset = dataset['train']

        if args.dataset_config_name == 'book':
            dataset = dataset.select(range(1))


    column_names = dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

    block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    # DataLoaders creation:
    eval_dataloader = DataLoader(
        lm_datasets, collate_fn=default_data_collator, batch_size=args.batch_size
    )

    config = AutoConfig.from_pretrained(args.model)
    if args.test_mode == 'base':
        # test performance of base models (e.g. llama-2-7b)
        config.add_cluster_predictor = False
        config.train_predictor = False
        model_id = args.model 

    elif args.test_mode == 'moe':
        # test performance of moe

        config.add_cluster_predictor = True
        config.train_predictor = False

        config.embedding_dim = args.embedding_dim
        config.inter_dim = args.inter_dim
        config.N_cluster = args.N_cluster
        config.hidden_act_predictor = args.hidden_act
        config.max_position_embeddings_predictor = 4096

        model_id = args.ckpt if args.ckpt is not None else args.model

    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)

    if args.test_mode == 'moe':
        mask = torch.load(args.mask_root)
        for name, module in model.named_modules():
            if hasattr(module, 'apply_mask'):
                layer_idx = int(name.split('.')[2])
                module.apply_mask(mask[f'module.model.layers.{layer_idx}.mlp.down_proj'], args.N_cluster)
    
        if args.convert_masked_dense_to_moe:
            for name, module in model.named_modules():
                if hasattr(module, 'convert_masked_dense_to_moe'):
                    module.convert_masked_dense_to_moe(args.N_cluster, intermediate_size_expert=int(0.5*11008), remove_dense=True)

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    args.eval_iter = min(args.eval_iter, len(eval_dataloader)) if args.eval_iter is not None else len(eval_dataloader)

    if accelerator.is_main_process:
        if args.output_file is not None:
            with open(args.output_file, "a") as f:
                print(f'Data loader size for eval: {args.eval_iter}/{len(eval_dataloader)}', file=f)
        else:
            print(f'Data loader size for eval: {args.eval_iter}/{len(eval_dataloader)}')

    losses = []
    for step, batch in enumerate(eval_dataloader):
        if args.cluster_id_test is not None:
            batch["cluster_id"] = args.cluster_id_test
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(args.batch_size)))

        if step >= args.eval_iter:
            break

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    if accelerator.is_main_process:
        if args.output_file is not None:

            output_str = f"perplexity: {perplexity} eval_loss: {eval_loss}"
            if args.dataset_config_name is not None:
                output_str = f"Test data domain: {args.dataset_config_name}: " + output_str
            if args.cluster_id_test is not None:
                output_str = f"Expert: {args.cluster_id_test}: " + output_str
            with open(args.output_file, "a") as f:
                print(output_str, file=f) 
        else:
            print(f"{perplexity} eval_loss: {eval_loss}")

        
if __name__ == '__main__':

    args = parse_args()
    main(args)
