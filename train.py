import numpy as np 
import pandas as pd 
import os
import random
import json
from tqdm import tqdm
import gc
import time
import subprocess
import matplotlib.pyplot as plt
from collections import defaultdict

import argparse
import yaml
from transformers import BertTokenizer, BertModel

from utils.metric import Evaluator, plot_metric
from model.models import FCModel
from data.dataloader import BucketSampler, collate_fn
from data.datasets import FCDataset
from utils.cv import cross_validation

import torch
import optuna
    
def set_seed(seed):
    random.seed(3407)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    ap = argparse.ArgumentParser("arguments for bert-nli training")
    ap.add_argument('--dataset', type=str, default='Snopes', help='[Snopes, Politifact]')
    args = ap.parse_args()
    return args

if __name__ == "__main__":
    set_seed(seed)

    bucket_size = 128
    batch_size = 32
    max_epochs = 50
    seed = 42

    args = parse_args()

    model_args = {
        'hidden_dim': 48,
        'emb_dim': 768
    }

    configs = {
        'lr': 0.00017767799670757544,
        'optimizer_name': 'Adam',
        'max_epoch': max_epochs,
        'batch_size': batch_size,
        'gradient_accumulation_steps': 1,
        'cuda': True
    }

    records = cross_validation(dataset=args.dataset, model_args, configs)

    plot_metric(metrics=['loss', 'accuracy'], sources=['train', 'valid', 'test'], records=records)
    plot_metric(metrics=['auc', 'f1_micro', 'f1_macro'], sources=['train', 'valid', 'test'], records=records)
    plot_metric(metrics=['precision_0', 'recall_0', 'f1_0'], sources=['train', 'valid'], records=records)
    plot_metric(metrics=['precision_1', 'recall_1', 'f1_1'], sources=['train', 'valid'], records=records)