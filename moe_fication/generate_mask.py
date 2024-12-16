import os 
import sys

sys.path.insert(1, 'src')

import torch
import datasets
import argparse
from itertools import chain
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

DATA_DOMAIN_LIST = [
    'arxiv', 'book', 'c4', 'common_crawl', 'github',
    'stackexchange', 'wikipedia'
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--mask_root", type=str)
    parser.add_argument("--N_cluster", type=int, default=8)
    parser.add_argument("--remain_rate", type=float, default=0.5)
    parser.add_argument("--N_feature", type=int, default=11008)
    args = parser.parse_args()
    return args

def main(args):

    act_list, binary_masks = [], []
    for i in range(args.N_cluster):

        # load_path = os.path.join(args.data_root, f'{args.data_domain}_cluster_{i}.pt')
        load_path = os.path.join(args.data_root, f'{DATA_DOMAIN_LIST[i]}.pt')
        data = torch.load(load_path)

        act_list.append(data)
        binary_masks.append(thresholding(data, args.N_feature, args.remain_rate))

    masks = dict()

    for key in binary_masks[0].keys():
        residual = torch.ones_like(binary_masks[0][key])
        for i in range(args.N_cluster): 
            residual = residual * binary_masks[i][key]
        
        masks[key] = {'residual': residual}
        for i in range(args.N_cluster):  
            masks[key][i] = binary_masks[i][key] - residual
    
    torch.save(masks, os.path.join(args.mask_root, 'mask.pt'))
     

def thresholding(data, N_feature, remain_rate):

    name_keys = data.keys()
    binary_masks = dict()
    for key in name_keys:
        vector = data[key]
        N = int(remain_rate*N_feature)
        th, _ = torch.sort(vector, dim=-1, descending=True)
        th = th[N-1]
        vector = (vector > th)
        binary_masks[key] = vector.to(torch.float16)

    return binary_masks


if __name__ == '__main__':
    args = parse_args()
    main(args)