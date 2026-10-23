from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def classification_metrics(prediction: np.ndarray, target: np.ndarray, n_classes: int) -> dict:
    mask = target >= 0
    target = target[mask]
    prediction = prediction[mask]
    cm = confusion_matrix(target, prediction, labels=range(n_classes))
    total = np.sum(cm)
    oa = float(np.trace(cm) / total) if total else 0.0
    class_acc = np.zeros(len(cm), dtype=np.float64)
    for i in range(len(cm)):
        row_sum = np.sum(cm[i, :])
        class_acc[i] = cm[i, i] / row_sum if row_sum else 0.0
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total) if total else 0.0
    kappa = (oa - pe) / (1 - pe) if (1 - pe) else 0.0
    return {
        "confusion_matrix": cm.tolist(),
        "oa": oa * 100.0,
        "aa": float(np.mean(class_acc) * 100.0),
        "kappa": float(kappa * 100.0),
        "class_acc": (class_acc * 100.0).tolist(),
    }


def aggregate_metrics(results: list[dict]) -> dict:
    aggregate = {}
    for key in ["oa", "aa", "kappa"]:
        values = np.asarray([item[key] for item in results], dtype=np.float64)
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values))
    return aggregate
