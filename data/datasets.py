import numpy as np 
import pandas as pd 
import os
import csv
import random
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F 
from transformers import BertTokenizer, BertModel

def tok2emb_sent(sentence, tokenizer, model, cuda):
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
    # input_ids = encoding['input_ids'] # token IDs
    # attention_mask = encoding['attention_mask']
    if cuda:
        encoding = encoding.to('cuda')
        # input_ids = input_ids.to('cuda')
        # attention_mask = attention_mask.to('cuda')
        
    # pass the encoded input through BERT model
    with torch.no_grad():
        outputs = model(**encoding)
        word_embeddings = outputs.last_hidden_state[0, :, :] # batch=0
        if cuda:
            word_embeddings = word_embeddings.to('cpu')
            # input_ids = input_ids.to('cpu')
            # attention_mask = attention_mask.to('cpu')
    return word_embeddings

def tok2emb_list(sentences, tokenizer, model, cuda):
    """
    Convert the input sequences into its word embedding with the pretrained bert model
    Arguments:
        sentences {list}: input sentences
        tokenizer: pretrained bert tokenizer
        model: pretrained bert model
    Returns:
        word_embeddings {torch.Tensor}: shape(sentence_num, seq_len, emb_dim)
    """
    encodings = tokenizer.batch_encode_plus(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True
    )
    # input_ids = encodings['input_ids']
    # attention_mask = encodings['attention_mask']
    if cuda:
        encodings = encodings.to('cuda')
        # input_ids = input_ids.to('cuda')
        # attention_mask = attention_mask.to('cuda')
        
    with torch.no_grad():
        outputs = model(**encodings)
        word_embeddings = outputs.last_hidden_state
        if cuda:
            word_embeddings = word_embeddings.to('cpu')
            # input_ids = input_ids.to('cpu')
            # attention_mask = attention_mask.to('cpu')
    return word_embeddings

class FCDataset(Dataset):
    """
    Fake News Detection Dataset
    """

    def __init__(self, data_path, tokenizer, pretrained_model, cuda=False, test=False):
        """
        Arguments:
            data_path {string} : path of the news dataset (.jsonl file)
            tokenizer: pretrained bert tokenizer
            pretrained_model: pretrained bert model
        """
        self.tokenizer = tokenizer
        self.model = pretrained_model
        # self.max_len_claim = self.examples
        self.test = test
        self.cuda = cuda
        if cuda:
            self.model.to('cuda')

        self.claim_src_vocab = {}
        self.evidence_src_vocab = {}
        
        self.examples = self.read_file(data_path)

        self.class_idx_samples = {i: [] for i in [0, 1]}
        for i, example in enumerate(self.examples):
            self.class_idx_samples[example[-1]].append(i)
        self.class_weight = {key: len(value) / len(self.examples) for key, value in self.class_idx_samples.items()}
        
        if cuda:
            self.model.to('cpu')
            torch.cuda.empty_cache()

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
        claim_src_vocab = {}
        evidence_src_vocab = {}
        claim_src_idx = 1
        evidence_src_idx = 1
        with open(data_path, 'r') as f:
            for i, line in enumerate(tqdm(f)):
                # if i > 3 :
                #     break
                data = json.loads(line.strip())
                label = 1 if data['cred_label'] == 'True' else 0
                # convert sequences into torch.Tensor
                claim_emb = tok2emb_sent(data['claim'], self.tokenizer, self.model, self.cuda)
                claim_source = data['claim_source']
                if claim_source not in claim_src_vocab:
                    claim_src_vocab[claim_source] = claim_src_idx
                    claim_src_idx += 1

                evidences = []
                evidences_source = []
                for evidence in data['evidences']:
                    evidences.append(evidence['evidence'])
                    evidences_source.append(evidence['evidence_source'])
                    if evidence['evidence_source'] not in evidence_src_vocab:
                        evidence_src_vocab[evidence['evidence_source']] = evidence_src_idx
                        evidence_src_idx += 1
                
                evidences_emb = tok2emb_list(evidences, self.tokenizer, self.model, self.cuda)
                example = [claim_emb, claim_source, evidences_emb, evidences_source]
                example = example + [label]
                examples.append(example)

            self.claim_src_vocab = claim_src_vocab
            self.evidence_src_vocab = evidence_src_vocab
        return examples

    def get_labels(self):
        _, _, _, _, labels = zip(*(self.examples))
        return list(labels)

    def get_claim_src_vocab(self):
        return self.claim_src_vocab

    def get_evidence_src_vocab(self):
        return self.evidence_src_vocab

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