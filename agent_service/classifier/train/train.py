"""训练与评测意图分类模型。

python -m classifier.train.train --task style --method baseline
python -m classifier.train.train --task style --method finetune --base BAAI/bge-small-zh-v1.5
python -m classifier.train.train --task route --method finetune --base hfl/chinese-roberta-wwm-ext

协议：训练集按 9:1 切出验证集（选最优 epoch、调多标签阈值），测试集只在最后评测一次。
结果写入 classifier/results/{task}_{name}.json，模型写入 models/{task}_{name}/（不入库）。
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from classifier.labels import ROUTE_LABELS, STYLE_LABELS
from classifier.model import Predictor, TextClassifier
from classifier.train.metrics import multilabel_metrics, singlelabel_metrics, to_multi_hot, tune_thresholds

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "classifier" / "data"
RESULTS_DIR = ROOT / "classifier" / "results"
MODELS_DIR = ROOT / "models"
SEED = 42


def load(task: str, split: str) -> List[dict]:
    with (DATA_DIR / f"{task}_{split}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def split_val(items: List[dict], ratio: float = 0.1) -> Tuple[List[dict], List[dict]]:
    items = items[:]
    random.Random(SEED).shuffle(items)
    n_val = max(1, int(len(items) * ratio))
    return items[n_val:], items[:n_val]


def targets(task: str, items: List[dict]):
    if task == "style":
        return to_multi_hot([it["labels"] for it in items], STYLE_LABELS)
    return np.array([ROUTE_LABELS.index(it["label"]) for it in items])


def evaluate(task: str, probs: np.ndarray, items: List[dict], thresholds=None) -> Dict[str, object]:
    if task == "style":
        th = np.array(thresholds if thresholds is not None else [0.5] * len(STYLE_LABELS))
        return multilabel_metrics(targets(task, items), (probs >= th).astype(int), STYLE_LABELS)
    pred = [ROUTE_LABELS[i] for i in probs.argmax(1)]
    return singlelabel_metrics([it["label"] for it in items], pred, ROUTE_LABELS)


def main_metric(task: str, m: Dict[str, object]) -> float:
    return float(m["micro_f1"] if task == "style" else m["macro_f1"])


# ---------------- 基线：句向量 + 逻辑回归 ----------------
def run_baseline(task: str, train, val, test, base: str) -> Dict[str, object]:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    enc = SentenceTransformer(base)
    emb = lambda items: enc.encode([it["text"] for it in items], normalize_embeddings=True, batch_size=64)
    t0 = time.perf_counter()
    x_tr, x_va, x_te = emb(train), emb(val), emb(test)
    if task == "style":
        clf = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced"))
    else:
        clf = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced")
    clf.fit(x_tr, targets(task, train))
    train_s = time.perf_counter() - t0
    p_va, p_te = clf.predict_proba(x_va), clf.predict_proba(x_te)
    thresholds = tune_thresholds(targets(task, val), p_va) if task == "style" else None

    lat = []
    for it in test[:50]:
        s = time.perf_counter()
        clf.predict_proba(enc.encode([it["text"]], normalize_embeddings=True))
        lat.append((time.perf_counter() - s) * 1000)
    return {"val": evaluate(task, p_va, val, thresholds), "test": evaluate(task, p_te, test, thresholds),
            "thresholds": thresholds, "train_seconds": round(train_s, 1), "latency_ms_p50": round(statistics.median(lat), 1)}


# ---------------- 微调：编码器 + 线性头 ----------------
def batches(items, size, shuffle, rng):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), size):
        yield [items[j] for j in idx[i:i + size]]


@torch.inference_mode()
def predict_probs(model, tok, items, task, max_len, bs=64) -> np.ndarray:
    model.eval()
    out = []
    for b in batches(items, bs, False, None):
        enc = tok([it["text"] for it in b], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        logits = model(enc["input_ids"], enc["attention_mask"])
        out.append((torch.sigmoid(logits) if task == "style" else torch.softmax(logits, -1)).numpy())
    return np.concatenate(out)


def run_finetune(task: str, train, val, test, base: str, out_dir: Path, epochs: int, lr: float, bs: int,
                 max_len: int) -> Dict[str, object]:
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    labels = STYLE_LABELS if task == "style" else ROUTE_LABELS
    tok = AutoTokenizer.from_pretrained(base)
    model = TextClassifier(base, len(labels))

    y_tr = targets(task, train)
    if task == "style":
        pos = y_tr.sum(0).clip(min=1)
        pos_weight = torch.tensor(((len(y_tr) - pos) / pos).clip(max=5.0), dtype=torch.float)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 正样本少的标签加大权重
    else:
        loss_fn = nn.CrossEntropyLoss()

    params = [
        {"params": model.encoder.parameters(), "lr": lr},
        {"params": model.head.parameters(), "lr": lr * 30},  # 随机初始化的分类头用更大学习率
    ]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = epochs * ((len(train) + bs - 1) // bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best, best_epoch, history = -1.0, 0, []
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for b in batches(train, bs, True, rng):
            enc = tok([it["text"] for it in b], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            y = targets(task, b)
            logits = model(enc["input_ids"], enc["attention_mask"])
            loss = loss_fn(logits, torch.tensor(y, dtype=torch.float if task == "style" else torch.long))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        m_val = evaluate(task, predict_probs(model, tok, val, task, max_len), val)
        score = main_metric(task, m_val)
        history.append({"epoch": epoch, "train_loss": round(total / len(train), 4), "val_main": round(score, 4)})
        print(f"  epoch {epoch}: loss={total / len(train):.4f} val={score:.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if score > best:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), out_dir / "model.pt")
    train_s = time.perf_counter() - t0

    model.load_state_dict(torch.load(out_dir / "model.pt", map_location="cpu"))
    p_va = predict_probs(model, tok, val, task, max_len)
    thresholds = tune_thresholds(targets(task, val), p_va) if task == "style" else None
    tok.save_pretrained(out_dir)
    meta = {"task": task, "base_model": base, "labels": labels, "multi_label": task == "style",
            "thresholds": thresholds, "max_len": max_len, "best_epoch": best_epoch}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    predictor = Predictor(out_dir)  # 用线上推理的同一代码路径做最终评测与测速
    p_te = np.array(predictor.predict_proba([it["text"] for it in test]))
    np.save(out_dir / "test_probs.npy", p_te)  # 供级联（本地 + LLM 兜底）分析使用
    lat = [predictor.predict(it["text"])["latency_ms"] for it in test[:50]]
    size_mb = round((out_dir / "model.pt").stat().st_size / 1e6, 1)
    return {"val": evaluate(task, p_va, val, thresholds), "test": evaluate(task, p_te, test, thresholds),
            "thresholds": thresholds, "best_epoch": best_epoch, "history": history,
            "train_seconds": round(train_s, 1), "latency_ms_p50": round(statistics.median(lat), 1), "model_mb": size_mb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["style", "route"], required=True)
    ap.add_argument("--method", choices=["baseline", "finetune"], required=True)
    ap.add_argument("--base", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--tag", default="", help="区分同一模型的不同训练配置，如 e10")
    args = ap.parse_args()

    train_all, test = load(args.task, "train"), load(args.task, "test")
    train, val = split_val(train_all)
    short = args.base.split("/")[-1] + (f"-{args.tag}" if args.tag else "")
    name = f"{args.method}_{short}"
    print(f"[{args.task}/{name}] train={len(train)} val={len(val)} test={len(test)}", flush=True)
    if args.method == "baseline":
        res = run_baseline(args.task, train, val, test, args.base)
    else:
        res = run_finetune(args.task, train, val, test, args.base, MODELS_DIR / f"{args.task}_{short}",
                           args.epochs, args.lr, args.bs, args.max_len)
    res.update({"task": args.task, "method": args.method, "base": args.base,
                "n_train": len(train), "n_val": len(val), "n_test": len(test)})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.task}_{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    t = res["test"]
    main_key = "micro_f1" if args.task == "style" else "macro_f1"
    print(f"[{args.task}/{name}] test {main_key}={t[main_key]:.4f} "
          f"{'exact=%.4f' % t['exact_match'] if args.task == 'style' else 'acc=%.4f' % t['accuracy']} "
          f"latency_p50={res['latency_ms_p50']}ms train={res['train_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
