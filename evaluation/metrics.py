import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch

def compute_metrics(eval_pred):
    """
    Computes accuracy, precision, recall, and F1-score for HF Trainer.
    """
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
        
    preds = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def get_confusion_matrix(y_true, y_pred, labels=None):
    """
    Returns confusion matrix.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)
