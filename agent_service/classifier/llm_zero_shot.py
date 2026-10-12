"""旅行风格分类：LLM 零样本版本（本地微调模型上线前的默认实现，之后作为兜底）。"""
from typing import Dict

from classifier.labels import ROUTE_DESC, ROUTE_LABELS
from core.errors import AppError
from core.llm import LLMClient
from core.logging import get_logger
from rag.pipeline import extract_json

log = get_logger("classifier")

STYLE_LABELS = [
    "Rush_Trip",
    "Deep_Cultural_Trip",
    "Food_Adventure",
    "Nature_Relaxation",
    "City_Leisure_Tour",
    "Customized_AI_Tour",
    "Vlog_Creator_Trip",
    "Photo_Experience",
]
DEFAULT_LABEL = "Customized_AI_Tour"


async def classify_style(llm: LLMClient, text: str) -> Dict[str, object]:
    """返回 {label, score, source}。LLM 失败时降级为默认标签而不是报错。"""
    prompt = (
        "你是旅行偏好分类器。从以下标签中选出最符合用户描述的一个："
        + ", ".join(STYLE_LABELS)
        + '。只返回 JSON：{"label": "<标签>", "score": <0到1的置信度>}。\n用户描述：'
        + text
    )
    try:
        raw = await llm.chat(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=60, purpose="classify"
        )
    except AppError as e:
        log.warning("classify degraded to default", extra={"code": e.code.value})
        return {"label": DEFAULT_LABEL, "score": 0.0, "source": "default", "degraded": ["classify"]}

    parsed = extract_json(raw) or {}
    label = parsed.get("label") if parsed.get("label") in STYLE_LABELS else DEFAULT_LABEL
    try:
        score = min(1.0, max(0.0, float(parsed.get("score"))))
    except (TypeError, ValueError):
        score = 0.5
    return {"label": label, "score": score, "source": "llm"}


async def classify_route(llm: LLMClient, text: str) -> Dict[str, object]:
    """路由的 LLM 零样本兜底（本地路由模型缺失或不确定时使用）；失败时按“经验问答”处理。"""
    defs = "\n".join(f"- {k}：{v}" for k, v in ROUTE_DESC.items())
    prompt = f"判断用户消息属于哪一类，只选一个。\n类别定义：\n{defs}\n只返回 JSON：{{\"label\": \"类别英文名\"}}\n消息：{text}"
    try:
        raw = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=30,
                             purpose="route.llm")
    except AppError:
        return {"route": "notes_qa", "source": "default"}
    label = (extract_json(raw) or {}).get("label")
    return {"route": label if label in ROUTE_LABELS else "notes_qa", "source": "llm"}
