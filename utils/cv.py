import sys
sys.path.append('..')
import pandas as pd 
import os
from tqdm import tqdm
import gc
import re
from collections import defaultdict

from model.models import FCModel
from data.dataloader import BucketSampler, BalancedBatchSampler, collate_fn
from data.datasets import FCDataset
from data.preprocess import df2json, load_evidences
from utils.metric import Evaluator, plot_metric

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F 
from transformers import BertTokenizer, BertModel
import optuna

def read_data(data_dir, all_evidences, files_name):
    pattern = re.compile(files_name)
    files = os.listdir(data_dir)
    files = [data_dir + file for file in files if pattern.match(file)]
    main_df = pd.read_csv(files[0], sep='\t', usecols=['id_left', 'id_right', 'cred_label', 'claim_text', 'claim_source', 'evidence', 'evidence_source'])
    for file in files[1: ]:
        data = pd.read_csv(file, sep='\t', usecols=['id_left', 'id_right', 'cred_label', 'claim_text', 'claim_source', 'evidence', 'evidence_source'])
        main_df = pd.concat([main_df, data], axis=0)
        
    main_df['correlation'] = -1
    def get_correlation(row):
        evidences = all_evidences[str(row['id_left'])]
        for i, e in enumerate(evidences):
            if e[0] in row['evidence']:
                row['correlation'] = e[1]
                break
        return row
                
    main_df = main_df.apply(get_correlation, axis=1)
    return main_df

class EarlyStopping:
    def __init__(self, save_path, patience=5, verbose=False, delta=-0.01):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.best_f1_ma = 0
        self.auc = 0
        self.delta = delta

    def __call__(self, epoch, f1_macro, auc, f1_macro_test, model):
        score = f1_macro
        print("Test f1_macro: ", f1_macro_test)
        if score < self.best_f1_ma + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            self.early_stop = False

        if self.best_f1_ma < f1_macro:
            self.save_checkpoint(epoch, f1_macro, auc, model)
            self.best_f1_ma = f1_macro
        elif self.best_f1_ma + self.delta <= f1_macro and auc >= self.auc:
            self.save_checkpoint(epoch, f1_macro, auc, model)

    def save_checkpoint(self, epoch, f1_macro, auc, model):
        if self.verbose:
            print(f'F1_macro increased ({self.best_f1_ma:.6f} --> {f1_macro:.6f}). Saving model ...')
    
        # torch.save({
        #             'epoch': epoch,
        #             'model': model.state_dict(),
        #             'f1_macro': f1_macro
        #         }, self.save_path)
        torch.save(model, self.save_path)
        if self.auc < auc:
            self.auc = auc

def default_value():
    return {
        'train': [],
        'valid': [],
        'test': []
    }

class Trainer():
    def __init__(self, model, lr, optimizer_name, weight_decay, max_epoch, batch_size, alpha=0, gradient_accumulation_steps=1, cuda=False):
        self.model = model
        self.cuda = cuda
        if cuda:
            self.model = self.model.to('cuda')
        self.lr = lr
        
        weight_p, bias_p = [], []
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
                
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(
                [{'params': weight_p, 'weight_decay': weight_decay},
                 {'params': bias_p, 'weight_decay': 0}
                 ], lr=lr
            )
        elif optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
            
            
        self.max_epoch = max_epoch
        self.loss_func = FactCheckingLoss(alpha=alpha)
        self.classify_loss = torch.nn.BCEWithLogitsLoss()
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
    def train(self, trial, outdir, trainset_reader, validset_reader, testset_reader):
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        writer = SummaryWriter(log_dir='./log')
        # scheduler = StepLR(self.optimizer, step_size=30, gamma=0.5)
        early_stopping = EarlyStopping(outdir + 'checkpoint.pt', patience=10, verbose=True)
    
        global_step = 0
        steps = 0
        records = defaultdict(default_value)
        train_loss = 0
        classify_loss = 0
        predictions = []
        labels = []
        for epoch in range(self.max_epoch):
            # training
            self.model.train()
            # optimizer.zero_grad()
            for batch_idx, (inputs, correlations, targets) in enumerate(trainset_reader):
                steps += 1
                targets = targets.float()
                if self.cuda:
                    inputs = tuple(input_tensor.to('cuda') for input_tensor in inputs)
                    targets = targets.to('cuda')
                    correlations = correlations.to('cuda')
                    
                    
                cosine_sim, logits = self.model(*inputs)
                loss = self.loss_func(cosine_sim, logits, correlations, targets)
                classify_loss += self.classify_loss(logits, targets).item()
                train_loss += loss.item()
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                self.optimizer.zero_grad()
                loss.backward()
                global_step += 1
                if global_step % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                
                predictions += torch.sigmoid(logits).to('cpu').tolist()
                labels += targets.to('cpu').tolist()
            # update learning rate
            # scheduler.step()
                
                if steps % int(len(trainset_reader)) == 0:
                #     for name, param in self.model.named_parameters():
                #         if param.grad is not None:
                #             writer.add_histogram(tag=name+'_grad', values=param.grad.clone().cpu().data.numpy(), global_step=epoch + 1)
                #             writer.add_histogram(tag=name+'_data', values=param.clone().cpu().data.numpy(), global_step=epoch + 1)
                        
                    classify_loss = classify_loss / int(len(trainset_reader))
                    train_loss = train_loss / int(len(trainset_reader))
                    rank_loss = train_loss - classify_loss
                    records['loss']['train'].append(train_loss)
                    records['classify_loss']['train'].append(classify_loss)
                    records['rank_loss']['train'].append(rank_loss)
                    train_evaluator = Evaluator(predictions, labels)
                    records['auc']['train'].append(train_evaluator.auc())
                    records['precision_0']['train'].append(train_evaluator.precision(pos_class=0))
                    records['precision_1']['train'].append(train_evaluator.precision(pos_class=1))
                    records['recall_0']['train'].append(train_evaluator.recall(pos_class=0))
                    records['recall_1']['train'].append(train_evaluator.recall(pos_class=1))
                    records['f1_0']['train'].append(train_evaluator.f1(pos_class=0))
                    records['f1_1']['train'].append(train_evaluator.f1(pos_class=1))
                    records['f1_macro']['train'].append(train_evaluator.f1_macro())
                    records['f1_micro']['train'].append(train_evaluator.f1_micro())
                    records['accuracy']['train'].append(train_evaluator.accuracy())
                    if self.cuda:
                        torch.cuda.empty_cache()
                
                    # validating
                    valid_loss = 0
                    classify_loss = 0
                    predictions = []
                    labels = []
                    self.model.eval()
                    with torch.no_grad():
                        for index, (inputs, correlations, targets) in enumerate(validset_reader):
                            targets = targets.float()
                            if self.cuda:
                                inputs = tuple(input_tensor.to('cuda') for input_tensor in inputs)
                                targets = targets.to('cuda')
                                correlations = correlations.to('cuda')
                
                            cosine_sim, logits = self.model(*inputs)
                            loss = self.loss_func(cosine_sim, logits, correlations, targets)
                            valid_loss += loss.item()
                            classify_loss += self.classify_loss(logits, targets).item()
                            
                            labels += targets.to('cpu').tolist()
                            predictions += torch.sigmoid(logits).to('cpu').tolist()
                        assert len(predictions) == len(labels)
                
                    valid_loss = valid_loss / len(validset_reader)
                    classify_loss = classify_loss / len(validset_reader)
                    rank_loss = valid_loss - classify_loss
                    records['loss']['valid'].append(valid_loss)
                    records['classify_loss']['valid'].append(classify_loss)
                    records['rank_loss']['valid'].append(rank_loss)
                    valid_evaluator = Evaluator(predictions, labels)
                    records['auc']['valid'].append(valid_evaluator.auc())
                    records['precision_0']['valid'].append(valid_evaluator.precision(pos_class=0))
                    records['precision_1']['valid'].append(valid_evaluator.precision(pos_class=1))
                    records['recall_0']['valid'].append(valid_evaluator.recall(pos_class=0))
                    records['recall_1']['valid'].append(valid_evaluator.recall(pos_class=1))
                    records['f1_0']['valid'].append(valid_evaluator.f1(pos_class=0))
                    records['f1_1']['valid'].append(valid_evaluator.f1(pos_class=1))
                    records['f1_macro']['valid'].append(valid_evaluator.f1_macro())
                    records['f1_micro']['valid'].append(valid_evaluator.f1_micro())
                    records['accuracy']['valid'].append(valid_evaluator.accuracy())
                
                    # testing
                    test_loss = 0
                    classify_loss = 0
                    predictions = []
                    labels = []
                    self.model.eval()
                    with torch.no_grad():
                        for index, (inputs, correlations, targets) in enumerate(testset_reader):
                            targets = targets.float()
                            if self.cuda:
                                inputs = tuple(input_tensor.to('cuda') for input_tensor in inputs)
                                targets = targets.to('cuda')
                                correlations = correlations.to('cuda')
                
                            cosine_sim, logits = self.model(*inputs)
                            loss = self.loss_func(cosine_sim, logits, correlations, targets)
                            test_loss += loss.item()
                            classify_loss += self.classify_loss(logits, targets).item()
                            
                            labels += targets.to('cpu').tolist()
                            predictions += torch.sigmoid(logits).to('cpu').tolist()
                        assert len(predictions) == len(labels)
                
                    test_loss = test_loss / len(testset_reader)
                    records['loss']['test'].append(test_loss)
                    classify_loss = classify_loss / len(testset_reader)
                    rank_loss = test_loss - classify_loss
                    records['classify_loss']['test'].append(classify_loss)
                    records['rank_loss']['test'].append(rank_loss)
                    test_evaluator = Evaluator(predictions, labels)
                    records['auc']['test'].append(test_evaluator.auc())
                    records['f1_macro']['test'].append(test_evaluator.f1_macro())
                    records['f1_micro']['test'].append(test_evaluator.f1_micro())
                    records['accuracy']['test'].append(test_evaluator.accuracy())
            
                    auc_score = records['auc']['valid'][-1]
                    print(f"epoch={epoch}\ttrain_loss={train_loss:.6f}\tvalid_loss={valid_loss:.6f}\tvalid_accuray={records['accuracy']['valid'][-1]:.6f}\tF1_macro={records['f1_macro']['valid'][-1]:.6f}\tF1_micro={records['f1_micro']['valid'][-1]:.6f}\tauc={auc_score:.6f}")
                    if trial != None:
                        trial.report(records['accuracy']['test'][-1], epoch)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()

                    if records['loss']['valid'][-1] > records['loss']['valid'][0]:
                        print("Early stopping")
                        writer.close()
                        return records
                        
                    early_stopping(epoch, records['f1_macro']['valid'][-1], records['auc']['valid'][-1], records['f1_macro']['test'][-1], self.model)
                    smallest_loss = heapq.nsmallest(2, records['loss']['valid'])
                    if records['loss']['valid'][-1] in smallest_loss:
                        early_stopping.early_stop = False
                        early_stopping.counter = 0
                    if early_stopping.early_stop:
                        if abs(records['loss']['valid'][-1] - records['loss']['train'][-1]) <= 0.02:
                            early_stopping.early_stop = False
                            early_stopping.counter = 0
                        else:
                            print("Early stopping")
                            writer.close()
                            return records
                            # break

                    train_loss = 0
                    classify_loss = 0
                    predictions = []
                    labels = []

        writer.close()
        return records

def cross_validation(dataset_name, model_args, configs, k_fold=5):
    """
    Arguments:
        dataset: 'Snopes' or 'PolitiFact'
    """
    bucket_size = 256
    batch_size = 32
    max_epochs = 1

    fold_num = []

    top_evidences = load_evidences(f'reoutput/{dataset_name}.json')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    validset = read_data(f'Datasets/{dataset_name}/mapped_data/', top_evidences, r'dev_ori.tsv')
    valid_json_file_path = 'valid.jsonl'
    df2json(validset, valid_json_file_path)
    validset = FCDataset(valid_json_file_path, tokenizer, bert_model, cuda=True)
    validset.examples = validset.examples[:-1]

    for i in range(k_fold):
        directory_path = f'Datasets/{dataset_name}/mapped_data/5fold/'
        dataset_path1 = r'train_' + str(i) + ".tsv"
        dataset_path2 = r'test_' + str(i) + "_ori.tsv"
        
        trainset = read_data(directory_path, top_evidences, dataset_path1)
        testset = read_data(directory_path, top_evidences, dataset_path2)
        
        train_json_file_path = f'train_{i}.jsonl'
        test_json_file_path = f'test_{i}.jsonl'
        
        df2json(trainset, train_json_file_path)
        df2json(testset, test_json_file_path)
        
        trainset = FCDataset(train_json_file_path, tokenizer, bert_model, cuda=True)
        testset = FCDataset(test_json_file_path, tokenizer, bert_model, cuda=True)
        if len(testset.examples) % 32 == 1:
            testset.examples = testset.examples[:-1]

        if 'PolitiFact' in dataset_name:
            train_sampler = BalancedBatchSampler(trainset, batch_size=batch_size, shuffle=True)
            trainset_reader = DataLoader(trainset, batch_sampler=train_sampler, collate_fn=collate_fn(trainset.claim_src_vocab, trainset.evidence_src_vocab))
        else:
            train_sampler = BucketSampler(trainset, bucket_size=bucket_size, shuffle=True)
            trainset_reader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn(trainset.claim_src_vocab, trainset.evidence_src_vocab))
        test_sampler = BucketSampler(testset, bucket_size=bucket_size, shuffle=True)
        testset_reader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, collate_fn=collate_fn(trainset.claim_src_vocab, trainset.evidence_src_vocab))
        valid_sampler = BucketSampler(validset, bucket_size=bucket_size, shuffle=True)
        validset_reader = DataLoader(validset, batch_size=batch_size, sampler=valid_sampler, collate_fn=collate_fn(trainset.claim_src_vocab, trainset.evidence_src_vocab))

        model_args['claim_src_num'] = len(trainset.claim_src_vocab) + 1
        model_args['evidence_src_num'] = len(trainset.evidence_src_vocab) + 1
        
        model = FCModel(**model_args)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    
        trainer = Trainer(model, **configs)
        records_tmp = trainer.train(None, f'checkpoint/{dataset_name}/{i}/', trainset_reader, validset_reader, testset_reader)
        plot_metric(f'i_1', metrics=['loss', 'accuracy'], sources=['train', 'valid', 'test'], records=records_tmp)
        plot_metric(f'i_2', metrics=['auc', 'f1_micro', 'f1_macro'], sources=['train', 'valid', 'test'], records=records_tmp)
        
        if i == 0:
            records = records_tmp
            fold_num += [1] * len(records['loss']['train'])
        else:
            target_length = max(len(records['loss']['train']), len(records_tmp['loss']['train']))
            fold_num += [0] * (target_length - len(fold_num))
            fold_num[: len(records_tmp['loss']['train'])] = [fold_num[i] + 1 for i in range(len(records_tmp['loss']['train']))]
            for metric in records:
                for source in ['train', 'valid', 'test']:
                    records[metric][source] = records[metric][source] + [0] * (target_length - len(records[metric][source]))
                    records_tmp[metric][source] = records_tmp[metric][source] + [0] * (target_length - len(records_tmp[metric][source]))
                    records[metric][source] = [records[metric][source][i] + records_tmp[metric][source][i] for i in range(len(records[metric][source]))]

        model.to('cpu')
        del model
        gc.collect()
    
    for metric in records:
        for source in ['train', 'valid', 'test']:
            records[metric][source] = [records[metric][source][i] / fold_num[i] for i in range(len(records[metric][source]))]
            
    return records