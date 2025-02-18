import numpy as np 
import pandas as pd 
import os
import csv
import random
import json

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F 
from transformers import BertTokenizer, BertModel

def tok2emb_sent(sentence, tokenizer, model):
    """
    Convert the input sequence into its word embedding with the pretrained bert model
    Arguments:
        sentence {string}: input sentence
        tokenizer: pretrained bert tokenizer
        model: pretrained bert model
    Returns:
        word_embeddings {torch.Tensor}: shape(1, seq_len, emb_dim)
    """
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased')

    # tokenizer split input sequence into sub-word and encode them into IDs
    encoding = tokenizer.batch_encode_plus(
        [sentence],
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True
    )
    input_ids = encoding['input_ids'] # token IDs
    attention_mask = encoding['attention_mask']
    # pass the encoded input through BERT model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state[0, :, :] # batch=0
    return word_embeddings

def tok2emb_list(sentences, tokenizer, model):
    """
    Convert the input sequences into its word embedding with the pretrained bert model
    Arguments:
        sentences {list}: input sentences
        tokenizer: pretrained bert tokenizer
        model: pretrained bert model
    Returns:
        word_embeddings {torch.Tensor}: shape(sentence_num, seq_len, emb_dim)
    """
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased')
    encodings = tokenizer.batch_encode_plus(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True
    )
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state
    return word_embeddings

class FCDataset(Dataset):
    """
    Fake News Detection Dataset
    """

    def __init__(self, data_path, tokenizer, pretrained_model, test=False, cuda=True):
        """
        Arguments:
            data_path {string} : path of the news dataset (.jsonl file)
            tokenizer: pretrained bert tokenizer
            pretrained_model: pretrained bert model
        """
        self.cuda = cuda
        self.tokenizer = tokenizer
        self.model = pretrained_model
        self.examples = self.read_file(data_path)
        self.max_len_claim = self.examples
        self.test = test

    def read_file(self, data_path):
        """
        Load examples from the .jsonl file
        Each example's format : [<claim>{torch.Tensor}, <claim_source>{torch.Tensor}, evidences{torch.Tensor}, evidences_source{torch.Tensor}, <cred_label>{int}]
        Arguments:
            data_path {string}: the .jsonl file where preprocessed dataset reserved
        Returns:
            examples {list}: a list containing all examples
        """
        examples = []
        with open(data_path, 'r') as f:
            for _, line in enumerate(f):
                data = json.loads(line.strip())

                label = 1 if data['cred_label'] == 'True' else 0
                # convert sequences into torch.Tensor
                claim_emb = tok2emb_sent(data['claim'], self.tokenizer, self.model)
                claim_source_emb = tok2emb_sent(data['claim_source'], self.tokenizer, self.model)

                evidences = []
                evidences_source = []
                for evidence in data['evidences']:
                    evidences.append(evidence['evidence'])
                    evidences_source.append(evidence['evidence_source'])
                
                evidences_emb = tok2emb_list(evidences, self.tokenizer, self.model)
                evidences_source_emb = tok2emb_list(evidences_source, self.tokenizer, self.model)
                example = [claim_emb, claim_source_emb, evidences_emb, evidences_source_emb]

                if self.cuda:
                    example = [item.cuda() for item in example] + [label]
                example = example + [label]
                examples.append(example)
        return examples

    def __len__(self):
        """
        return the quantity of examples
        """
        return len(self.examples)
        
    def __getitem__(self, idx):
        """
        return the idxth example 
        Arguments:
            idx {int}: the position of the example
        Returns:
            claim_emb {torch.Tensor}: shape(seq_len, emb_dim)
            claim_source_emb {torch.Tensor}: shape(seq_len, emb_dim)
            evidences_emb {torch.Tensor}: shape(evi_num, seq_len, emb_dim)
            evidences_source_emb {torch.Tensor}: shape(evi_num, seq_len, emb_dim)
            
            target {int}
        """
        example = self.examples[idx]
        target = example[4]
        return (example[0], example[1], example[2], example[3], target)