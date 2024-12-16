import os 
import sys
import copy
import random

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
from huggingface_hub import Repository, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_scheduler, AutoConfig

EVAL_STEP = 50
import datetime
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=108000))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_name")
    parser.add_argument("--dataset_config_name")
    parser.add_argument("--training_file", type=str)
    parser.add_argument("--validation_file", type=str)
    parser.add_argument("--N_cluster", type=int)
    parser.add_argument("--cluster_id_test", type=int, help="map all tokens to the cluster id")
    parser.add_argument("--mask_root", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--save", type=str)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--modify_lr_scheduler", action='store_true')
    parser.add_argument("--revise_block_size", type=int, default=2048)
    parser.add_argument("--round", type=int, default=0)

    # arguments for cluster predictor
    parser.add_argument('--hidden_act', default='silu')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--inter_dim', type=int, default=64)
    parser.add_argument('--n_layer', type=int)
    parser.add_argument('--n_head', type=int)
    parser.add_argument('--attn_normalizer', type=float)
    parser.add_argument('--nonautoregressive', action='store_true')

    parser.add_argument('--train_predictor', action='store_true', help="fix backbone and mask, tune cluster predictor")
    parser.add_argument('--finetune', action='store_true', help="fix predictor and mask, tune backbone")
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--learning_rate_decay', default=0.8, type=float)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--max_train_steps', type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--resume_from_checkpoint', action='store_true')
    parser.add_argument('--checkpointing', action='store_true')

    parser.add_argument('--push_to_hub', default=None)
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    # validation
    assert not (args.train_predictor and args.finetune)

    return args

def main(args):

    args.learning_rate = args.learning_rate * (args.learning_rate_decay ** args.round)
    args.skip_step = int(args.round * 300) + (0 if args.train_predictor else 100)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision='bf16')

    tokenizer = AutoTokenizer.from_pretrained(
                args.model, use_fast=True, trust_remote_code=True,
                token="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"
            )
    
    if args.dataset_name is not None:
        if args.dataset_name == 'togethercomputer/RedPajama-Data-1T-sample':
            # data_arrow_file_root = ''
            # args.validation_split_percentage = 0.01
            # # train_split_size = 200000
            # arr_files = [fn for fn in os.listdir(data_arrow_file_root) if fn.endswith('.arrow')]
            # data_files = {
            #     'train': sorted([os.path.join(data_arrow_file_root, fn) for fn in arr_files if '-train' in fn])}
            # raw_datasets = load_dataset("arrow", data_files=data_files)['train']
            raw_datasets = load_dataset(args.dataset_name)['train']

            random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)

            valid_idxs = random.sample(range(len(raw_datasets)), int(args.validation_split_percentage*len(raw_datasets)))
            train_idxs = list(set(range(len(raw_datasets))) - set(valid_idxs))
            # train_idxs = random.sample(train_idxs, train_split_size)

            train_dataset = raw_datasets.select(train_idxs)
            valid_dataset = raw_datasets.select(valid_idxs)

            
            if 'llama' in args.model:
                tokenize_type = 'llama2_7b_chat'
            elif 'mistral' in args.model:
                tokenize_type = 'mistral_instruct'
            else:
                assert False

        
        
    def process_dataset(dataset, cache_name, shuffle=True):
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
                desc=f"Grouping texts in chunks of {block_size}"
            )

        # DataLoaders creation:
        if shuffle:
            dataloader = DataLoader(
                lm_datasets.shuffle(), collate_fn=default_data_collator, batch_size=args.batch_size)
        else:
            dataloader = DataLoader(
                lm_datasets, collate_fn=default_data_collator, batch_size=args.batch_size)
        return dataloader
    
    eval_dataloader = process_dataset(valid_dataset, cache_name='valid', shuffle=True)
    train_dataloader = process_dataset(train_dataset, cache_name='train', shuffle=True)

    config = AutoConfig.from_pretrained(args.model)
    config.add_cluster_predictor = True
    config.train_predictor = args.train_predictor
    config.embedding_dim = args.embedding_dim
    config.inter_dim = args.inter_dim
    config.N_cluster = args.N_cluster
    config.hidden_act_predictor = args.hidden_act
    config.max_position_embeddings_predictor = 4096

    config.n_head = args.n_head
    config.n_layer = args.n_layer
    config.normalizer = args.attn_normalizer

    config.nonautoregressive = args.nonautoregressive

    
    if args.push_to_hub:
        model = AutoModelForCausalLM.from_pretrained(args.ckpt, config=config)
        if accelerator.is_main_process:
            model.push_to_hub(args.push_to_hub, private=True, safe_serialization=False)
        sys.exit()

    else:
        model_id = args.model if args.train_predictor else args.ckpt
        model = AutoModelForCausalLM.from_pretrained(
            model_id, config=config, attn_implementation='flash_attention_2')
    
        if accelerator.is_main_process:
            print('Load Model: ', model_id)
    
    mask = torch.load(args.mask_root)
    for name, module in model.named_modules():
        if hasattr(module, 'apply_mask'):
            layer_idx = int(name.split('.')[2])
            module.apply_mask(mask[f'module.model.layers.{layer_idx}.mlp.down_proj'], args.N_cluster)
            

    # customize optimzer
    no_decay = ["bias", "layer_norm.weight"]

    params_list = []
    if args.train_predictor:
        for n, p in model.named_parameters():
            if 'cluster_predictor' in n:
                p.requires_grad_(True)
                params_list.append(p)
                # print("Trainable Parameter: ", n)
            else:
                p.requires_grad_(False)
        optimizer = torch.optim.AdamW(params_list, lr=args.learning_rate)

    elif args.finetune:
        params_list_no_decay = []
        for n, p in model.named_parameters():
            if 'cluster_predictor' not in n:
                p.requires_grad_(True)
                if any(nd in n for nd in no_decay):
                    params_list_no_decay.append(p)
                else:
                    params_list.append(p)
            else:
                p.requires_grad_(False)
    
        optimizer_grouped_parameters = [
                        {"params": params_list, "weight_decay": args.weight_decay},
                        {"params": params_list_no_decay, "weight_decay": 0.0},
                    ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)


    if args.modify_lr_scheduler:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=args.max_train_steps * accelerator.num_processes
        )
    else:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=args.max_train_steps
        )


    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader)
    

    if args.train_predictor:
        mask_list = []
        for key, value in mask.items():
            # temp = value['residual'] + torch.stack([value[i] for i in range(args.N_cluster)], dim=0)
            temp = torch.stack([value[i] for i in range(args.N_cluster)], dim=0)
            mask_list.append(temp)
        train_predictor(args, accelerator, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, tokenizer, mask_list)
    
    elif args.finetune:
        finetune(args, accelerator, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, tokenizer)


def loss_func(feat, pred, mask_list, mse_loss, ce_loss, accelerator, rescale=False, idx=0):
    T = 1
    alpha = 0.75
    buffer = 0
    # feat [bsz, seqlen, N_channel], pred [bsz, seqlen, N_cluster]
    # mask_list [8, N_channel]
    bsz, seqlen = feat.shape[0], feat.shape[1]
    N = int(torch.sum(mask_list[0]))
    chunk_list = [N-1, int(N/5*4), int(N/5*3), int(N/5*2), int(N/5)]
    feat_th = torch.zeros_like(feat)
    sorted_feat, _ = torch.sort(feat, dim=-1, descending=True)
    for cidx, N_item in enumerate(chunk_list):
        th = sorted_feat[..., N_item, None]
        feat_th += (feat > th).to(torch.float32) / 3 / N
    
    dist = torch.matmul(feat_th, mask_list.transpose(1,0).to(device=feat_th.device, dtype=feat_th.dtype))
    dist = (dist - torch.mean(dist)) * 10
    
    soft_dist = torch.nn.functional.softmax(dist/T, dim=-1)
    soft_prob = torch.nn.functional.log_softmax(pred/T, dim=-1)
    soft_targets_loss = -torch.sum(soft_dist * soft_prob) / (bsz*seqlen) * (T**2)

    labels = torch.argmax(dist, dim=-1)
    label_loss = ce_loss(pred.reshape(bsz*seqlen, -1), labels.flatten())

    # if idx == 31 and accelerator.is_main_process:
    #     preds_count = torch.bincount(torch.argmax(pred, dim=-1).flatten())
        # print(f'pred Layer Index {idx}', preds_count)

        # labels_count = torch.bincount(labels.flatten())
        # print(f'label Layer Index {idx}', labels_count)

    acc =  torch.eq(torch.argmax(pred, dim=-1), labels).sum() / (seqlen*bsz)

    loss = label_loss * alpha + soft_targets_loss * (1-alpha)

    return loss, acc.item()


    
def train_predictor(args, accelerator, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, tokenizer, mask_list):

    if accelerator.is_main_process:
        print('Dense Model Performance:')
    evaluate(args, accelerator, model, eval_dataloader, 0, eval_step=EVAL_STEP, eval_clusterer=False)
    if accelerator.is_main_process:
        print('MoE Performance:')
    evaluate(args, accelerator, model, eval_dataloader, 0, eval_step=EVAL_STEP)

    mse_loss = torch.nn.MSELoss()
    ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.5)

    if args.skip_step > 0 :
        train_dataloader = accelerator.skip_first_batches(train_dataloader, args.skip_step)

    def add_hook_model(model):
        key_names = {
            'mlp': {'hook_module': 'mlp.down_proj'}
        }
        sort_type = 'mlp'

        def hook(module, fea_in, fea_out):
            activation_list.append(fea_in)
        
        handlers, names_with_hook  = [], []
        for name, module in model.named_modules():
            if key_names[sort_type]['hook_module'] in name:
                handlers.append(module.register_forward_hook(hook))
                names_with_hook.append(name)
        return model, handlers, names_with_hook
    
    def remove_hook(handlers):
        for handler in handlers:
            handler.remove()


    model, handlers, names_with_hook = add_hook_model(model)

    completed_steps = 0
    acc_list, losses = [], []

    while True:
        for step, batch in enumerate(train_dataloader):
            
            model.train()
            with accelerator.accumulate(model):
                activation_list = []
                for key_item in batch:
                    batch[key_item] = batch[key_item][..., :args.revise_block_size]
                batch['return_cluster_pred'] = True
                cluster_predictor_output = model(**batch)
            
                loss = 0
                assert len(activation_list) == 32
                for idx, feat in enumerate(activation_list):
                    loss_temp, acc = loss_func(feat[0], cluster_predictor_output, mask_list[idx], mse_loss, ce_loss, idx=idx, accelerator=accelerator)
                    loss += loss_temp
                    acc_list.append(acc)
                
                del activation_list
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            losses.append(accelerator.gather(loss.repeat(args.batch_size)))

            if accelerator.sync_gradients:
                completed_steps += 1
                
                losses = torch.cat(losses)
                train_loss = torch.mean(losses)
                acc = torch.mean(torch.tensor(acc_list))

                if accelerator.is_main_process:
                    lr = get_lr(optimizer)
                    print(f"step {completed_steps}/{args.max_train_steps}: lr{lr:.8f} loss: {train_loss} acc: {acc}")
                losses, acc_list = [], []

                if (completed_steps+1) % args.eval_interval == 0:
                    remove_hook(handlers)
                    evaluate(args, accelerator, model, eval_dataloader, completed_steps, eval_step=EVAL_STEP)
                    model, handlers, names_with_hook = add_hook_model(model)
            
            if (completed_steps+1) >= args.max_train_steps:
                break

        if (completed_steps+1) >= args.max_train_steps:
                break
    
    remove_hook(handlers)
    
    os.makedirs(args.save, exist_ok=True)
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.save, is_main_process=accelerator.is_main_process, save_function=accelerator.save,
        safe_serialization=False
    )
    if accelerator.is_main_process:
        print('Saving model to ', args.save)
        tokenizer.save_pretrained(args.save)

def finetune(args, accelerator, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, tokenizer):

    if args.resume_from_checkpoint:
        name_list = os.listdir(args.save)
        name_list = [int(name[5:]) for name in name_list if name.startswith('step_')]
        if len(name_list)>0:
            completed_steps = max(name_list)
            args.resume_from_checkpoint = os.path.join(args.save, f'step_{completed_steps}')
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            train_dataloader = accelerator.skip_first_batches(train_dataloader, args.skip_step+completed_steps)
        else:
            accelerator.print("No checkpoint detected!")
            completed_steps = 0
            train_dataloader = accelerator.skip_first_batches(train_dataloader, args.skip_step)

    else:
        completed_steps = 0
        train_dataloader = accelerator.skip_first_batches(train_dataloader, args.skip_step)

    os.makedirs(args.save, exist_ok=True)

    # if accelerator.is_main_process:
    #     print('Dense Model Performance:')
    # evaluate(args, accelerator, model, eval_dataloader, 0, eval_step=EVAL_STEP, eval_clusterer=False)
    if accelerator.is_main_process:
        print('MoE Performance:')
    evaluate(args, accelerator, model, eval_dataloader, 0, eval_step=EVAL_STEP)

    losses = []
    while True:
        for step, batch in enumerate(train_dataloader):

            model.train()
            with accelerator.accumulate(model):
                for key_item in batch:
                    batch[key_item] = batch[key_item][..., :args.revise_block_size]
                output = model(**batch)
                loss = output.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            losses.append(accelerator.gather(loss.repeat(args.batch_size)))

            if accelerator.sync_gradients:
                torch.cuda.empty_cache() 
                completed_steps += 1
                
                losses = torch.cat(losses)
                train_loss = torch.mean(losses)

                if accelerator.is_main_process:
                    lr = get_lr(optimizer)
                    print(f"step {completed_steps}/{args.max_train_steps}: lr{lr:.8f} loss: {train_loss}")
                losses = []

                if (completed_steps+1) % args.eval_interval == 0:
                    evaluate(args, accelerator, model, eval_dataloader, completed_steps, eval_step=EVAL_STEP)

                    if args.checkpointing and (completed_steps+1) < args.max_train_steps:
                        output_dir = f"step_{completed_steps}"
                        output_dir = os.path.join(args.save, output_dir)
                        os.makedirs(output_dir, exist_ok=True)
                        accelerator.save_state(output_dir)

            if (completed_steps+1) >= args.max_train_steps:
                    break
        
        if (completed_steps+1) >= args.max_train_steps:
            break
                
    
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.save, is_main_process=accelerator.is_main_process, save_function=accelerator.save,
        safe_serialization=False
    )
    if accelerator.is_main_process:
        print('Saving model to ', args.save)
        tokenizer.save_pretrained(args.save)



def evaluate(args, accelerator, model, eval_dataloader, completed_steps, eval_step=None, eval_clusterer=True):

    losses = []
    model.eval()
    if eval_step is None:
        eval_step = len(eval_dataloader)

    for step, batch in enumerate(eval_dataloader):

        with torch.no_grad():
            for key_item in batch:
                batch[key_item] = batch[key_item][..., :args.revise_block_size]
            batch['eval_clusterer'] = eval_clusterer
            output = model(**batch)
            loss = output.loss
            losses.append(accelerator.gather(loss.repeat(args.batch_size)))

            train_loss = torch.cat(losses)
            train_loss = torch.mean(train_loss)
            if step >= eval_step:
                break

    if accelerator.is_main_process:
        print(f"[Evaluation] step {completed_steps}: loss: {train_loss} ")



def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def get_locality(pred, collect_pred_list):
    # pred [bsz, seqlen, N_cluster]

    for n in [1, 3, 5, 10, 25]:
        pass
        
if __name__ == '__main__':

    args = parse_args()
    main(args)
