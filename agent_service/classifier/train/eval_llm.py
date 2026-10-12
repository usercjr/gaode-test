"""LLM 零样本分类（对照组）：与本地模型用同一测试集、同一标签定义、同一指标。

python -m classifier.train.eval_llm --task style
"""
import argparse
import asyncio
import json
import statistics
import time

from classifier.labels import ROUTE_DESC, ROUTE_LABELS, STYLE_DESC, STYLE_LABELS
from classifier.train.metrics import multilabel_metrics, singlelabel_metrics, to_multi_hot
from classifier.train.train import RESULTS_DIR, load, split_val
from config import get_settings
from core.errors import AppError
from core.llm import LLMClient
from core.logging import setup_logging
from rag.pipeline import extract_json

STYLE_PROMPT = """判断旅行需求体现了哪些风格，可多选，也可以一个都不选（表示综合/无明显风格）。
风格定义：
{defs}
只返回 JSON：{{"labels": ["标签英文名", ...]}}
需求：{text}"""

ROUTE_PROMPT = """判断用户消息属于哪一类，只选一个。
类别定义：
{defs}
只返回 JSON：{{"label": "类别英文名"}}
消息：{text}"""


async def main(task: str, split: str = "test") -> None:
    settings = get_settings()
    setup_logging("WARNING")
    llm = LLMClient(settings)
    test = load(task, "test") if split == "test" else split_val(load(task, "train"))[1]
    defs = "\n".join(f"- {k}：{v}" for k, v in (STYLE_DESC if task == "style" else ROUTE_DESC).items())
    template = STYLE_PROMPT if task == "style" else ROUTE_PROMPT

    async def one(item):
        started = time.perf_counter()
        try:
            raw = await llm.chat([{"role": "user", "content": template.format(defs=defs, text=item["text"])}],
                                 temperature=0.0, max_tokens=60, purpose="clf.llm_eval")
            obj = extract_json(raw) or {}
        except AppError:
            obj = {}
        return obj, (time.perf_counter() - started) * 1000

    results = await asyncio.gather(*[one(it) for it in test])
    # 并发跑分时的耗时包含排队等待，不能代表单次延迟；另外顺序调用 20 条测真实单次延迟
    latencies = [(await one(it))[1] for it in test[:20]]
    await llm.aclose()
    if task == "style":
        pred = [[l for l in (o.get("labels") or []) if l in STYLE_LABELS] for o, _ in results]
        preds_out = pred
        metrics = multilabel_metrics(to_multi_hot([it["labels"] for it in test], STYLE_LABELS),
                                     to_multi_hot(pred, STYLE_LABELS), STYLE_LABELS)
    else:
        pred = [o.get("label") if o.get("label") in ROUTE_LABELS else "chitchat" for o, _ in results]
        preds_out = pred
        metrics = singlelabel_metrics([it["label"] for it in test], pred, ROUTE_LABELS)
    res = {"task": task, "method": "llm_zero_shot", "base": settings.llm.model, "n_test": len(test),
           "test": metrics, "latency_ms_p50": round(statistics.median(latencies), 1),
           "note": "顺序调用 20 条测得的单次耗时，含网络往返", "predictions": preds_out}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if split == "val":  # 验证集上的 LLM 预测只用于选级联阈值，不进主表
        (RESULTS_DIR.parent / "data" / f"{task}_llm_val_predictions.json").write_text(
            json.dumps({"predictions": preds_out, "metrics": metrics}, ensure_ascii=False), encoding="utf-8")
        print(f"[{task}/llm_zero_shot/val] saved")
        return
    (RESULTS_DIR / f"{task}_llm_zero_shot.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    key = "micro_f1" if task == "style" else "macro_f1"
    print(f"[{task}/llm_zero_shot] test {key}={metrics[key]:.4f} latency_p50={res['latency_ms_p50']}ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["style", "route"], required=True)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    args = ap.parse_args()
    asyncio.run(main(args.task, args.split))
