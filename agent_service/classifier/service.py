"""在线分类服务：优先用本地微调模型（毫秒级、零成本），模型不存在或置信度低时退回 LLM 零样本。

返回结构兼容前端旧接口 {label, score}，另外给出多标签 labels、路由 route 与来源 source。
"""
from pathlib import Path
from typing import Dict, Optional

from starlette.concurrency import run_in_threadpool

from classifier.labels import LEGACY_DEFAULT, LEGACY_LABEL, ROUTE_LABELS
from classifier.llm_zero_shot import classify_style as llm_classify_style
from classifier.model import Predictor, load_predictor
from core.llm import LLMClient
from core.logging import get_logger

log = get_logger("classifier")

# 本地路由模型最高概率低于该值时返回 route=None，交给 LLM 判断（P5 动态路由使用）。
# 级联阈值在验证集上选择，结果是“全部用本地模型”最优（eval/classifier_report.md），所以默认 0。
ROUTE_CONFIDENCE = 0.0


class ClassifierService:
    def __init__(self, llm: LLMClient, models_dir: Path, style_model: str, route_model: str):
        self.llm = llm
        self.style: Optional[Predictor] = None
        self.route: Optional[Predictor] = None
        self._dirs = (models_dir / style_model, models_dir / route_model)

    def load(self) -> None:
        for attr, d in zip(("style", "route"), self._dirs):
            try:
                setattr(self, attr, load_predictor(d))
                log.info("classifier loaded" if getattr(self, attr) else "classifier model missing",
                         extra={"model_dir": str(d)})
            except Exception:
                log.exception("classifier load failed", extra={"model_dir": str(d)})

    async def classify(self, text: str, with_route: bool = False) -> Dict[str, object]:
        """with_route 只对聊天原话开启：路由模型只在单句消息上训练，表单拼接格式不在其分布内。"""
        if self.style is None:
            result = await llm_classify_style(self.llm, text)
            result.setdefault("labels", [])
            return result
        pred = await run_in_threadpool(self.style.predict, text)
        labels = pred["labels"]
        scores = pred["scores"]
        primary = max(labels, key=lambda l: scores[l]) if labels else None
        result: Dict[str, object] = {
            "label": LEGACY_LABEL[primary] if primary else LEGACY_DEFAULT,  # 兼容前端的 8 类风格提示词
            "score": scores[primary] if primary else round(1 - max(scores.values()), 4),
            "labels": labels,
            "scores": scores,
            "source": "local",
            "latency_ms": pred["latency_ms"],
        }
        if with_route and self.route is not None:
            r = await run_in_threadpool(self.route.predict, text)
            result["route"] = r["label"] if r["score"] >= ROUTE_CONFIDENCE else None
            result["route_score"] = r["score"]
        return result

    def status(self) -> Dict[str, object]:
        return {"style_model": bool(self.style), "route_model": bool(self.route),
                "route_labels": ROUTE_LABELS}
