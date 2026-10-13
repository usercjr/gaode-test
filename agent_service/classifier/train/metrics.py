"""分类指标：风格（多标签）与路由（单标签）统一口径，基线、微调、LLM 零样本共用。"""
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, object]:
    """y_* 为 0/1 矩阵 [n, L]。exact_match = 全部标签都对（含“无标签=综合”）。"""
    per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
    none_true, none_pred = y_true.sum(1) == 0, y_pred.sum(1) == 0
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match": float((y_true == y_pred).all(1).mean()),
        "none_f1": float(f1_score(none_true, none_pred, zero_division=0)),
        "per_label_f1": {l: round(float(f), 3) for l, f in zip(labels, per_label)},
    }


def singlelabel_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_label_f1": {l: round(float(f), 3) for l, f in
                         zip(labels, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0))},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def to_multi_hot(label_lists: List[List[str]], labels: List[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(label_lists), len(labels)), dtype=int)
    for r, ls in enumerate(label_lists):
        for l in ls:
            if l in idx:
                m[r, idx[l]] = 1
    return m


def tune_thresholds(y_true: np.ndarray, probs: np.ndarray) -> List[float]:
    """每个标签在验证集上选使 F1 最大的阈值（0.2~0.8）。"""
    grid = np.arange(0.2, 0.81, 0.05)
    out = []
    for j in range(y_true.shape[1]):
        best = max(grid, key=lambda t: f1_score(y_true[:, j], probs[:, j] >= t, zero_division=0))
        out.append(round(float(best), 2))
    return out
