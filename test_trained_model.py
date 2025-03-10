import numpy as np 
import pandas as pd 
import os
import csv
import random
import json
from tqdm import tqdm
import gc
import seaborn as sns
import time
import subprocess
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F 
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
import optuna

def test_model(model_path, dataset_reader, name):
    model_args = {
        # 'hidden_dim': trial.params['hidden_dim'],
        'hidden_dim': 48,
        'emb_dim': 768
    }
    model = FCModel(**model_args)
    model.load_state_dict(torch.load(model_path)['model'])
    
    model.to('cuda')
    model.eval()

    labels = []
    outputs = []
    with torch.no_grad():
        for index, (inputs, targets) in enumerate(dataset_reader):
            targets = targets.float()
            inputs = tuple(input_tensor.to('cuda') for input_tensor in inputs)
            targets = targets.to('cuda')
    
            logits = model(*inputs)
    
            labels += targets.to('cpu').tolist()
            outputs += logits.to('cpu').tolist()
        assert len(outputs) == len(labels)

    probabilities = np.array(torch.sigmoid(torch.tensor(outputs)).tolist())
    labels = np.array(labels)
    predictions= np.where(probabilities > 0.5, 1, 0)
    
    evaluator = Evaluator(predictions, labels)
    print(f'Accuracy: {evaluator.accuracy():.6f} \t AUC: {evaluator.auc():.6f} \t F1_macro: {evaluator.f1_macro():.6f} \t F1_micro: {evaluator.f1_micro():.6f}')
    conf_matrix = confusion_matrix(labels, predictions)
    print(conf_matrix)

    outputs = np.array(outputs)
    indices = np.arange(len(outputs))[(predictions == labels) & (predictions == 1)]
    tp = outputs[indices]
    indices = np.arange(len(outputs))[(predictions != labels) & (predictions == 1)]
    fp = outputs[indices]
    indices = np.arange(len(outputs))[(predictions == labels) & (predictions == 0)]
    tn = outputs[indices]
    indices = np.arange(len(outputs))[(predictions != labels) & (predictions == 0)]
    fn = outputs[indices]
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', ax=axes[0, 0])
    sns.histplot(outputs, ax=axes[0, 1])
    sns.histplot(tp, ax=axes[1, 0])
    sns.histplot(fp, ax=axes[1, 1])
    sns.histplot(tn, ax=axes[2, 0])
    sns.histplot(fn, ax=axes[2, 1])
    # add title and labels
    axes[0, 0].set_title(f'{name} Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted Labels')
    axes[0, 0].set_ylabel('True Labels')
    axes[0, 1].set_title('Histogram of outputs')
    axes[1, 0].set_title('Histogram of tp')
    axes[1, 1].set_title('Histogram of fp')
    axes[2, 0].set_title('Histogram of tn')
    axes[2, 1].set_title('Histogram of fn')
    # show plot
    plt.tight_layout()
    plt.show()

    model.to('cpu')
    del model
    gc.collect()

    return evaluator