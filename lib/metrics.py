import numpy as np
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true.tolist())) > 1 else 0.0,
        "per_class_precision": p.tolist(),
        "per_class_recall": r.tolist(),
        "per_class_f1": f.tolist(),
    }
