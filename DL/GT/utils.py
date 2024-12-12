import numpy as np
from scipy.stats import pearsonr

def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true)**2))

def sd(y_pred, y_true):
    return np.std(y_pred - y_true)

def pearson(y_pred, y_true):
    return pearsonr(y_pred, y_true)[0]

def label_normalization(labels, min_value=0, max_value=1):
    labels = np.array(labels)
    min_label = 0.01
    max_label = 19.05
    normalized_labels = (labels-min_label)/(max_label-min_label) * (max_value-min_value) + min_label
    return normalized_labels
