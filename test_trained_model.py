import numpy as np 
import pandas as pd 
import os
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import math
import heapq
import argparse

from utils.metric import Evaluator
from utils.cv import read_data
from model.models import FCModel
from data.datasets import FCDataset
from data.dataloader import BucketSampler, collate_fn
from data.preprocess import load_evidences

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import BertTokenizer, BertModel
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, confusion_matrix
import optuna

import warnings
warnings.filterwarnings("ignore")

def test_model(fold_id, model_path, dataset_reader, name):
    model = torch.load(model_path)
    
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
    plt.savefig(f"img/test_{fold_id}.png")
    plt.show()

    model.to('cpu')
    del model
    gc.collect()

    return evaluator

def parse_args():
    ap = argparse.ArgumentParser("arguments for bert-nli training")
    ap.add_argument('--dataset', type=str, default='Snopes', help='[Snopes, Politifact]')
    args = ap.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    bucket_size = 256
    batch_size = 32

    top_evidences = load_evidences(f'reoutput/{dataset}.json')
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    for i in range(5):
        directory_path = f'Datasets/{dataset}/mapped_data/5fold/'
        dataset_path1 = r'train_' + str(i) + ".tsv"
        trainset = read_data(directory_path, top_evidences, dataset_path1)
        train_json_file_path = f'train_{i}.jsonl'
        df2json(trainset, train_json_file_path)
        trainset = FCDataset(train_json_file_path, tokenizer, bert_model, cuda=True)
        
        dataset_path2 = r'test_' + str(i) + ".tsv"
        testset = read_data(directory_path, top_evidences, dataset_path2)
        test_json_file_path = f'test_{i}.jsonl'
        df2json(testset, test_json_file_path)
        testset = FCDataset(test_json_file_path, tokenizer, bert_model, cuda=True)
        if len(testset.examples) % 32 == 1:
            testset.examples = testset.examples[:-1]

        test_sampler = BucketSampler(testset, bucket_size=bucket_size, shuffle=True)
        testset_reader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, collate_fn=collate_fn(trainset.claim_src_vocab, trainset.evidence_src_vocab))

        test_model(i, model_path, testset_reader, 'testset')