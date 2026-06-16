import numpy as np

def compute_accuracy(predictions, labels):
    return np.mean(np.array(predictions) == np.array(labels))

def compute_fpr(tp, fp, tn, fn):
    if (fp + tn) == 0:
        return 0.0
    return fp / (fp + tn)

def compute_fnr(tp, fp, tn, fn):
    if (tp + fn) == 0:
        return 0.0
    return fn / (tp + fn)