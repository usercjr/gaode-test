"""从用户消息中抽取“值得长期记住”的信息。

LLM 只提出候选，是否写入由确定性规则把关：
1. 键必须在固定集合内
2. evidence 必须是用户原话的子串（防止模型凭空“记住”用户没说过的话）
3. 含手机号 / 身份证 / 邮箱 / 银行卡等敏感信息的候选一律丢弃
"""
import re
from typing import Dict, List

from core.errors import AppError
from core.llm import LLMClient
from core.logging import get_logger
from memory.store import ALL_KEYS
from rag.pipeline import extract_json

log = get_logger("memory.extractor")

SENSITIVE = [
    re.compile(r"1[3-9]\d{9}"),                       # 手机号
    re.compile(r"\d{17}[\dXx]"),                      # 身份证
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),           # 邮箱
    re.compile(r"\d{16,19}"),                         # 银行卡
]

PROMPT = """你是旅行助手的记忆模块。判断用户这句话里有没有值得**长期记住**、对以后规划行程有用的个人信息。

只记录满足全部条件的信息：
1. 用户明确说出来的，不是你推测的
2. 可复用：以后规划别的行程也用得上（同行人、饮食禁忌、体力限制、预算水平、节奏偏好、喜欢/讨厌的类型、常住城市）
3. 稳定：不是这一次的临时想法（“这次想吃火锅”“明天想去宽窄巷子”“第二天轻松点”都不记）

键只能从下面选：
- companion：同行人（如“带父母”“和女朋友”“带 5 岁孩子”）
- mobility：体力或行动限制（如“膝盖不好不能爬山”“推婴儿车”）
- budget：预算水平（如“学生党预算低”“预算宽裕”）
- pace：节奏偏好（如“喜欢慢节奏”“喜欢特种兵式”）
- home_city：常住或经常出发的城市
- diet：饮食禁忌或偏好（如“不吃辣”“素食”“海鲜过敏”）
- likes：喜欢的景点或活动类型（如“喜欢博物馆”）
- dislikes：不喜欢、想避开的类型（如“不喜欢人挤人的网红景点”）
likes / dislikes 的 value 只写对象本身（如“博物馆”“网红景点”），不要带“喜欢/不喜欢/逛”等词。

不要记录：手机号、证件号、具体住址等隐私；对助手的评价；闲聊。
evidence 必须原样摘抄用户原话中的片段。confidence：用户说得明确且是长期习惯给 0.9，带“可能/一般/有时”等模糊词给 0.5。

只返回 JSON：{{"memories": [{{"key": "...", "value": "简短概括，不超过 15 字", "evidence": "原话片段", "confidence": 0.9}}]}}
没有就返回 {{"memories": []}}

用户说：{message}"""


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def is_sensitive(text: str) -> bool:
    return any(p.search(text or "") for p in SENSITIVE)


def validate_candidates(message: str, candidates: List[Dict]) -> List[Dict]:
    """确定性过滤（可单独测试）：键合法、evidence 出自原话、无敏感信息。"""
    msg = _compact(message)
    out = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        key, value = str(c.get("key", "")), str(c.get("value", "")).strip()
        evidence = str(c.get("evidence", "")).strip()
        try:
            confidence = max(0.0, min(1.0, float(c.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        if key not in ALL_KEYS or not value or not evidence:
            continue
        if _compact(evidence) not in msg:
            log.info("memory candidate dropped: evidence not in message", extra={"key": key, "value": value})
            continue
        if is_sensitive(value) or is_sensitive(evidence):
            log.info("memory candidate dropped: sensitive", extra={"key": key})
            continue
        out.append({"key": key, "value": value[:15], "evidence": evidence[:100], "confidence": confidence})
    return out


async def extract(llm: LLMClient, message: str) -> List[Dict]:
    """失败时返回空列表：记忆是增强功能，不能影响主流程。"""
    if not message.strip() or len(message) > 1000:
        return []
    try:
        raw = await llm.chat([{"role": "user", "content": PROMPT.format(message=message)}],
                             temperature=0.0, max_tokens=300, purpose="memory.extract")
    except AppError as e:
        log.warning("memory extraction skipped", extra={"code": e.code.value})
        return []
    return validate_candidates(message, (extract_json(raw) or {}).get("memories") or [])
