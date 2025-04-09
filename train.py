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
import math
import heapq

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

import warnings
warnings.filterwarnings("ignore")
    
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
    bucket_size = 256
    batch_size = 32
    max_epochs = 100
    seed = 42
    
    set_seed(seed)

    args = parse_args()

    model_args = {
        'hidden_dim': 48,
        'emb_dim': 768
    }

    configs = {
        'lr': 0.00015,
        'optimizer_name': 'Adam',
        'max_epoch': max_epochs,
        'batch_size': batch_size,
        'gradient_accumulation_steps': 1,
        'cuda': True
    }

    if not os.path.exists('img'):
        os.makedirs('img')

    records = cross_validation(args.dataset, model_args, configs)

    plot_metric('avg_1', metrics=['loss', 'accuracy'], sources=['train', 'valid', 'test'], records=records)
    plot_metric('avg_2', metrics=['auc', 'f1_micro', 'f1_macro'], sources=['train', 'valid', 'test'], records=records)
    plot_metric('avg_3', metrics=['precision_0', 'recall_0', 'f1_0'], sources=['train', 'valid'], records=records)
    plot_metric('avg_4', metrics=['precision_1', 'recall_1', 'f1_1'], sources=['train', 'valid'], records=records)