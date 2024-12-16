import os 
import sys

sys.path.insert(1, 'src')

import torch
import random
import datasets
import argparse
from itertools import chain
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
        choices=['mistralai/Mistral-7B-Instruct-v0.3', 'meta-llama/Llama-2-7b-chat-hf'])
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_name")
    parser.add_argument("--dataset_config_name")
    parser.add_argument("--validation_file", type=str)
    parser.add_argument("--save_dir")
    parser.add_argument("--dataset_per", type=int, default=2)
    args = parser.parse_args()
    return args

def save_activations(args):

    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(
                args.model, use_fast=True, trust_remote_code=True,
                token="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"
            )
    
    if args.dataset_name is not None:
        # example: dataset = load_dataset('togethercomputer/RedPajama-Data-1T','arxiv', split='train')
        # dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=f'train[:{args.dataset_per}%]')
        data_arrow_dict = {}

        if args.dataset_config_name in data_arrow_dict.keys():
            data_arrow_file_root = data_arrow_dict[args.dataset_config_name]
            arr_files = [fn for fn in os.listdir(data_arrow_file_root) if fn.endswith('.arrow')]
            data_files = {
                'train': sorted([os.path.join(data_arrow_file_root, fn) for fn in arr_files if '-train' in fn and 'NNNNN' not in fn])[:1]}
            dataset = load_dataset("arrow", data_files=data_files)['train']

        else:
            dataset = load_dataset(args.dataset_name, args.dataset_config_name, 
                split=f'train[:{args.dataset_per}%]')
            # dataset = dataset['train']
        
        if args.dataset_config_name == 'book':
            dataset = dataset.select(range(1))

    else:
        # load local C4 data
        dataset = load_dataset("json", data_files=args.validation_file)['train']
    

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
        lm_datasets, collate_fn=default_data_collator, batch_size=args.batch_size, shuffle=True
    )


    model = AutoModelForCausalLM.from_pretrained(args.model)
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    total_activations = get_intermediate(model, eval_dataloader)
    torch.save(total_activations, args.save_dir)

def get_intermediate(model, dataloader):
    model.eval()

    key_names = {
        'mlp': {'hook_module': 'mlp.down_proj'}
    }

    channel_importance_iters = 16
    batch_channel_importance_iters = 2
    sort_type = 'mlp'

    total_activations_list = []
    for hook_iter in range(channel_importance_iters//batch_channel_importance_iters):
        activation_list = []
        def hook(module, fea_in, fea_out):
            # activation_list.append(fea_out[0])
            activation_list.append(fea_in)
        
        handlers, names_with_hook, total_activations = [], [], dict()
        for name, module in model.named_modules():
            if key_names[sort_type]['hook_module'] in name:
                handlers.append(module.register_forward_hook(hook))
                names_with_hook.append(name)
                total_activations[name] = []

        with torch.no_grad():
            iteration = 0
            for step, batch in enumerate(dataloader):
                if iteration >= batch_channel_importance_iters:
                    break
                iteration += 1  
                outputs = model(**batch)
        
        for handler in handlers:
            handler.remove()

        for idx, feat in enumerate(activation_list):
            feat = torch.abs(feat[0]).mean(dim=(0,1))
            total_activations[names_with_hook[idx%len(names_with_hook)]].append(feat)
        
        for key in total_activations:
            total_activations[key] = torch.mean(
                torch.stack(total_activations[key], dim=0), dim=0)

        
        total_activations_list.append(total_activations)
    
    results = dict()
    for key in names_with_hook:
        results[key] = torch.stack([temp[key] for temp in total_activations_list]).mean(0)
    
    return results

        
if __name__ == '__main__':

    args = parse_args()
    save_activations(args)
    sys.exit()
