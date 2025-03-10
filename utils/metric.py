import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, confusion_matrix


class Evaluator(object):
    def __init__(self, predictions, labels, threshold=0.5):
        self.predictions = np.array(predictions)
        self.labels = np.array(labels)
        self.threshold = threshold
        predictions = np.where(self.predictions > 0.5, 1, 0)
        self.tp_0 = np.sum((predictions == 0) & (predictions == self.labels))
        self.fp_0 = np.sum((predictions == 0) & (predictions != self.labels))
        self.fn_0 = np.sum((predictions != 0) & (predictions != self.labels))
        self.tp_1 = np.sum((predictions == 1) & (predictions == self.labels))
        self.fp_1 = np.sum((predictions == 1) & (predictions != self.labels))
        self.fn_1 = np.sum((predictions != 1) & (predictions != self.labels))
        
    def precision(self, pos_class=0):
        tp = self.tp_0 if pos_class == 0 else self.tp_1
        fp = self.fp_0 if pos_class == 0 else self.fp_1
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        return precision

    def recall(self, pos_class):
        tp = self.tp_0 if pos_class == 0 else self.tp_1
        fn = self.fn_0 if pos_class == 0 else self.fn_1
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        return recall
        
    def f1(self, pos_class):
        precision = self.precision(pos_class)
        recall = self.recall(pos_class)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def accuracy(self):
        predictions = np.where(self.predictions > 0.5, 1, 0)
        accuracy = accuracy_score(self.labels, predictions)
        return accuracy

    def auc(self):
        fpr, tpr, thresholds = roc_curve(self.labels, self.predictions)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def f1_macro(self):
        f1_macro = (self.f1(pos_class = 0) + self.f1(pos_class=1)) / 2
        return f1_macro

    def f1_micro(self):
        predictions = np.where(self.predictions > 0.5, 1, 0)
        tp_micro = self.tp_0 + self.tp_1
        fp_micro = self.fp_0 + self.fp_1
        fn_micro = self.fn_0 + self.fn_1
                
        precision_micro = tp_micro / (tp_micro + fp_micro)
        recall_micro = tp_micro / (tp_micro + fn_micro)
        f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)
        return f1_micro

def plot_metric(metrics, sources, records, colors=['#1f77b4', 'orange', 'green']):
    fig, axs = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    for i, metric in enumerate(metrics):
        epochs = range(1, len(records[metric][sources[0]]) + 1)
        for j, source in enumerate(sources):
            axs[i].plot(epochs, records[metric][source], color=colors[j], label=f'{source} {metric}')
        axs[i].set_title(metric)
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(metric)
        axs[i].legend()
        axs[i].grid()
    
    plt.tight_layout()
    plt.show()