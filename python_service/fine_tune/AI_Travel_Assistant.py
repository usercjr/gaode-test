# -*- coding: utf-8 -*-
"""
Use your fine-tuned Wenxin model together with the local PaddleNLP classifier.
This script is standalone and DOES NOT run inside the FastAPI service.

Environment variables (PowerShell examples):
  $env:BAIDU_LLM_API_KEY="YOUR_KEY"
  $env:BAIDU_BASE_URL="https://aistudio.baidu.com/llm/lmapi"   # optional
  $env:BAIDU_API_VERSION="v3"                                   # optional (v1 or v3)
  $env:BAIDU_MODEL="ernie-3.5-8k"                               # or your fine-tuned model ID
  $env:BEST_MODEL_DIR="D:\gaode_test\best_model"                # local classifier dir

Run:
  cd D:\gaode_test\python_service
  .\.venv310\Scripts\Activate.ps1
  python .\fine_tune\AI_Travel_Assistant.py
"""

import os
import json
import time
import numpy as np

from paddlenlp.transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    # OpenAI Python SDK (works with Wenxin's OpenAI-compatible endpoint)
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


# ========== 1) Wenxin LLM (OpenAI-compatible) ==========
BAIDU_LLM_API_KEY = (os.getenv("BAIDU_LLM_API_KEY") or os.getenv("BAIDU_API_KEY") or "").strip()
BAIDU_BASE_URL = (os.getenv("BAIDU_BASE_URL") or "https://aistudio.baidu.com/llm/lmapi").rstrip("/")
BAIDU_API_VERSION = (os.getenv("BAIDU_API_VERSION") or "v3").strip()
MODEL_NAME = (os.getenv("BAIDU_MODEL") or "ernie-3.5-8k").strip()

if OpenAI is None:
    raise RuntimeError("openai package not installed. Install it via: pip install openai")

client = OpenAI(api_key=BAIDU_LLM_API_KEY, base_url=f"{BAIDU_BASE_URL}/{BAIDU_API_VERSION}")


def generate_itinerary(origin: str, destination: str, date_str: str, preference_summary: str) -> str:
    """Call Wenxin (OpenAI-compatible) to generate itinerary text."""
    sys = (
        "你是旅行行程规划助手。严格按如下中文格式输出行程文本，不要额外解释或添加 markdown 代码块：\n\n"
        "第1天 标题\n- [城市] 景点（备注）\n- [城市] 景点（备注）\n\n"
        "第2天 标题\n- [城市] 景点（备注）\n...\n\n"
        "要求：\n- 标题以“第N天 …”开头；\n- 每个景点一行，以“- ”开头；\n"
        "- 城市写在中括号内，如[上海]；备注写在中文括号内；\n- 不要输出除行程文本以外的任何内容。"
    )
    user = f"出发地：{origin or '未填写'}；目的地：{destination or '未填写'}；日期：{date_str or '未填写'}；偏好：{preference_summary or '未选择'}。请生成 2-3 天行程。"

    # Some providers require the first message to be user or assistant; merge system+user to user
    combined = f"{sys}\n\n【用户需求】\n{user}"

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": combined}],
        temperature=0.7,
    )

    # Try multiple shapes
    def _extract_text(obj):
        if not obj:
            return ""
        # standard OpenAI
        try:
            text = obj.choices[0].message.content
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception:
            pass
        # Baidu result nesting
        try:
            data = resp.model_dump() if hasattr(resp, "model_dump") else resp
            if isinstance(data, dict):
                if isinstance(data.get("result"), str) and data["result"].strip():
                    return data["result"].strip()
                inner = data.get("result") or {}
                if isinstance(inner, dict):
                    if isinstance(inner.get("result"), str) and inner["result"].strip():
                        return inner["result"].strip()
                    if isinstance(inner.get("output_text"), str) and inner["output_text"].strip():
                        return inner["output_text"].strip()
        except Exception:
            pass
        return ""

    text = _extract_text(resp)
    if not text:
        raise RuntimeError("LLM 未返回文本")
    return text


# ========== 2) Local PaddleNLP classifier ==========
BEST_MODEL_DIR = os.getenv("BEST_MODEL_DIR", os.path.join(os.getcwd(), "best_model"))
clf_tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
clf_model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
clf_model.eval()


def classify_preference(text: str):
    import paddle

    if not text or not text.strip():
        return {"label": "Unknown", "score": 0.0}
    enc = clf_tokenizer(
        text,
        max_seq_len=256,
        pad_to_max_seq_len=True,
        return_tensors="pd",
    )
    with paddle.no_grad():
        logits = clf_model(enc["input_ids"], enc["token_type_ids"])
    logits_np = np.array(logits)
    exp = np.exp(logits_np - np.max(logits_np, axis=-1, keepdims=True))
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    pred_id = int(np.argmax(probs))
    score = float(probs[0, pred_id])
    return {"label": str(pred_id), "score": score}


if __name__ == "__main__":
    # Example usage
    user_desc = "目的地：上海；偏好：独自出行、经典必去"
    cls = classify_preference(user_desc)
    print("Preference classify:", cls)

    itinerary = generate_itinerary(
        origin="上海",
        destination="上海",
        date_str=time.strftime("%Y-%m-%d"),
        preference_summary="独自出行、经典必去",
    )
    print("\n=== Generated Itinerary ===\n")
    print(itinerary)


