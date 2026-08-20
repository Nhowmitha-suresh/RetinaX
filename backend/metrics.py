"""
RetinaX Evaluation & Metrics Calculation Module.
Computes Loss, Accuracy, Quadratic Weighted Kappa (QWK), Macro/Per-class Precision, Recall, F1, and Confusion Matrix.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

def compute_metrics(y_true: list, y_pred: list, loss_val: float = 0.0) -> dict:
    """
    Computes comprehensive evaluation metrics for 5-class Diabetic Retinopathy classification.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]).tolist()

    return {
        "val_loss": float(loss_val),
        "accuracy": float(round(acc, 2)),
        "qwk": float(round(qwk, 4)),
        "macro_precision": float(round(macro_p, 4)),
        "macro_recall": float(round(macro_r, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "per_class": {
            int(c): {
                "precision": float(round(per_class_p[c], 4)),
                "recall": float(round(per_class_r[c], 4)),
                "f1": float(round(per_class_f1[c], 4)),
            }
            for c in range(5)
        },
        "confusion_matrix": cm,
    }
