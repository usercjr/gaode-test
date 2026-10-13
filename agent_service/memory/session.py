"""会话上下文：最近几轮原文 + 更早对话的滚动摘要；以及把依赖上下文的追问改写成独立问题。"""
import re
from typing import Dict, List

from core.errors import AppError
from core.llm import LLMClient
from core.logging import get_logger
from memory.store import MemoryStore

log = get_logger("memory.session")

KEEP_RECENT = 6        # 保留原文的最近轮数（user/assistant 各算一轮）
SUMMARIZE_AFTER = 10   # 未被摘要的轮数超过该值时，把较早的部分压缩进摘要

SUMMARY_PROMPT = """把下面的旅行对话压缩成摘要，供之后的对话参考。按固定字段输出纯文本，没有的字段写“无”：
目的地与日期：
已确定的安排：
用户否决过的方案：
待定事项：
其他偏好：

{previous}
对话：
{dialog}"""

REWRITE_PROMPT = """根据对话上下文，把用户最后一句话改写成一个不依赖上下文、能单独理解的问题（补全省略的地点、天数等）。
只输出改写后的问题，不要解释。如果原句已经完整，原样输出。

上下文：
{context}

用户最后一句：{message}"""

# 含这些指代/省略词，或者很短的消息，才需要结合上下文改写
_CONTEXTUAL = re.compile(r"那里|那边|这里|这边|那个|这个|它|附近|旁边|那家|这家|刚才|上面|前面说|第[一二三四五1-5]天|还有呢|呢$")


def needs_rewrite(message: str) -> bool:
    msg = message.strip()
    return len(msg) <= 8 or bool(_CONTEXTUAL.search(msg))


def _dialog(turns: List[Dict]) -> str:
    return "\n".join(f"{'用户' if t['role'] == 'user' else '助手'}：{t['content'][:300]}" for t in turns)


class SessionContext:
    def __init__(self, store: MemoryStore, llm: LLMClient):
        self.store = store
        self.llm = llm

    def context_text(self, session_id: str) -> str:
        """摘要 + 最近几轮原文，用于提示词。"""
        if not session_id:
            return ""
        s = self.store.summary(session_id)
        recent = self.store.turns(session_id, after_id=s["covered_until"])[-KEEP_RECENT:]
        parts = []
        if s["summary"]:
            parts.append(f"【之前对话摘要】\n{s['summary']}")
        if recent:
            parts.append(f"【最近对话】\n{_dialog(recent)}")
        return "\n\n".join(parts)

    async def standalone_question(self, session_id: str, message: str) -> str:
        """多轮追问改写成独立问题；没有上下文、不需要改写或改写失败时返回原句。"""
        context = self.context_text(session_id)
        if not context or not needs_rewrite(message):
            return message
        try:
            rewritten = await self.llm.chat(
                [{"role": "user", "content": REWRITE_PROMPT.format(context=context, message=message)}],
                temperature=0.0, max_tokens=80, purpose="memory.rewrite")
        except AppError:
            return message
        rewritten = rewritten.strip().strip("“”\"")
        return rewritten if 0 < len(rewritten) <= 120 else message

    async def maybe_summarize(self, session_id: str) -> bool:
        """未摘要的轮数过多时，把除最近 KEEP_RECENT 轮以外的部分合并进摘要。"""
        s = self.store.summary(session_id)
        pending = self.store.turns(session_id, after_id=s["covered_until"])
        if len(pending) <= SUMMARIZE_AFTER:
            return False
        to_fold = pending[:-KEEP_RECENT]
        previous = f"已有摘要：\n{s['summary']}\n" if s["summary"] else ""
        try:
            summary = await self.llm.chat(
                [{"role": "user", "content": SUMMARY_PROMPT.format(previous=previous, dialog=_dialog(to_fold))}],
                temperature=0.0, max_tokens=400, purpose="memory.summary")
        except AppError:
            return False
        self.store.save_summary(session_id, summary.strip(), to_fold[-1]["id"])
        log.info("session summarized", extra={"session_id": session_id, "folded_turns": len(to_fold)})
        return True
