"""文本分类模型：预训练编码器（CLS 向量）+ 线性分类头。训练与在线推理共用，保证预处理一致。

- multi_label=True：sigmoid + 每个标签独立阈值（在验证集上调出），用于风格多标签
- multi_label=False：softmax，用于路由单标签
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # 屏蔽加载权重时“未使用的预训练头”提示


class TextClassifier(nn.Module):
    def __init__(self, base_model: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # bge 系列以 CLS 作为句向量
        return self.head(self.dropout(cls))


class Predictor:
    """加载训练产物目录（model.pt + tokenizer + meta.json）做推理。"""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "meta.json").read_text(encoding="utf-8"))
        self.labels: List[str] = self.meta["labels"]
        self.multi_label: bool = self.meta["multi_label"]
        self.thresholds: List[float] = self.meta.get("thresholds") or [0.5] * len(self.labels)
        self.max_len: int = self.meta["max_len"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = TextClassifier(self.meta["base_model"], len(self.labels))
        self.model.load_state_dict(torch.load(self.model_dir / "model.pt", map_location="cpu"))
        self.model.eval()

    @torch.inference_mode()
    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        logits = self.model(enc["input_ids"], enc["attention_mask"])
        probs = torch.sigmoid(logits) if self.multi_label else torch.softmax(logits, dim=-1)
        return probs.tolist()

    def predict(self, text: str) -> Dict[str, object]:
        started = time.perf_counter()
        probs = self.predict_proba([text])[0]
        scores = {l: round(p, 4) for l, p in zip(self.labels, probs)}
        if self.multi_label:
            labels = [l for l, p, t in zip(self.labels, probs, self.thresholds) if p >= t]
            result: Dict[str, object] = {"labels": labels, "scores": scores}
        else:
            best = max(range(len(probs)), key=probs.__getitem__)
            result = {"label": self.labels[best], "score": round(probs[best], 4), "scores": scores}
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return result


def load_predictor(model_dir: Path) -> Optional[Predictor]:
    return Predictor(model_dir) if (Path(model_dir) / "meta.json").exists() else None
