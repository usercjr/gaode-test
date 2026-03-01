import os
import json
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 懒加载：在没有模型文件时也能启动服务（返回 503 提示）
MODEL_READY = False
LAST_ERROR = ""
# 默认从项目根目录查找 best_model，如果当前在 python_service 目录，则向上查找
_default_model_dir = os.path.join(os.getcwd(), "best_model")
if not os.path.exists(_default_model_dir) and os.path.basename(os.getcwd()) == "python_service":
    # 如果在 python_service 目录下，尝试从上级目录查找
    _default_model_dir = os.path.join(os.path.dirname(os.getcwd()), "best_model")
BEST_MODEL_DIR = os.environ.get("BEST_MODEL_DIR", _default_model_dir)
tokenizer = None
model = None
ID2LABEL = None

app = FastAPI(title="Local PaddleNLP Classifier", version="0.1.0")


def _load_label_map() -> Dict[int, str]:
    lm_json = os.path.join(BEST_MODEL_DIR, "label_map.json")
    if os.path.exists(lm_json):
        with open(lm_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            id2label = data.get("id2label") or {}
            ret = {}
            for k, v in id2label.items():
                try:
                    ret[int(k)] = v
                except Exception:
                    pass
            if ret:
                return ret
    # 占位默认标签，待你提供真实 label_map.json 后会自动替换
    return {
        0: "Rush_Trip",
        1: "Deep_Cultural_Trip",
        2: "Food_Adventure",
        3: "Family_Friendly",
        4: "Nightlife",
    }


def _softmax(x):
    import numpy as np
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


@app.on_event("startup")
def _maybe_load_model():
    global MODEL_READY, tokenizer, model, ID2LABEL, LAST_ERROR
    try:
        from paddlenlp.transformers import AutoTokenizer, AutoModelForSequenceClassification
        import paddle
        # 确认目录
        if not os.path.isdir(BEST_MODEL_DIR):
            print(f"[python_service] best_model directory not found: {BEST_MODEL_DIR}. Service will start in placeholder mode.")
            MODEL_READY = False
            LAST_ERROR = "best_model_not_found"
            return
        tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
        model.eval()
        ID2LABEL = _load_label_map()
        MODEL_READY = True
        LAST_ERROR = ""
        print(f"[python_service] Model loaded from {BEST_MODEL_DIR}. Labels: {len(ID2LABEL)}")
    except Exception as e:
        print(f"[python_service] Failed to load model: {e}. Service will start in placeholder mode.")
        MODEL_READY = False
        LAST_ERROR = str(e)


class ClassifyReq(BaseModel):
    text: str


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_ready": MODEL_READY,
        "model_dir": BEST_MODEL_DIR,
        "last_error": LAST_ERROR,
    }


@app.post("/classify")
def classify(req: ClassifyReq):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="empty text")
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="model_not_loaded: 请放置 best_model/ 并重启服务")

    # 真正推理
    from paddlenlp.transformers import AutoTokenizer  # noqa: F401  (确保环境依赖被声明)
    import paddle
    import numpy as np

    enc = tokenizer(
        req.text,
        max_seq_len=256,
        pad_to_max_seq_len=True,
        return_tensors="pd"
    )
    with paddle.no_grad():
        logits = model(enc["input_ids"], enc["token_type_ids"])
    logits_np = np.array(logits)  # [1, num_labels]
    probs = _softmax(logits_np)[0]
    pred_id = int(np.argmax(probs))
    label = ID2LABEL.get(pred_id, str(pred_id))
    score = float(probs[pred_id])
    return {"label": label, "score": score}


