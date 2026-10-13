"""记忆服务：给路由与规划提供“读记忆 / 记录轮次 / 抽取并写入记忆 / 会话摘要”。

所有方法在 user_id / session_id 缺失时都退化为无操作，记忆是增强能力，不影响主流程。
"""
import asyncio
from typing import Dict, List

from core.llm import LLMClient
from core.logging import get_logger
from memory import extractor
from memory.session import SessionContext
from memory.store import KEY_CN, MemoryStore

log = get_logger("memory")


class MemoryService:
    def __init__(self, store: MemoryStore, llm: LLMClient):
        self.store = store
        self.llm = llm
        self.session = SessionContext(store, llm)
        self._bg: set = set()

    def prompt_lines(self, user_id: str) -> List[str]:
        return self.store.prompt_lines(user_id) if user_id else []

    async def learn(self, user_id: str, message: str, source: str = "") -> List[Dict]:
        """从用户原话抽取记忆并写入，返回发生的变化（供前端提示“已记住…”）。"""
        if not user_id:
            return []
        updates = []
        for c in await extractor.extract(self.llm, message):
            change = self.store.upsert(user_id, c["key"], c["value"], c["confidence"], c["evidence"], source)
            if change:
                m = change["memory"]
                updates.append({"action": change["action"], "id": m["id"], "key": m["key"],
                                "key_cn": KEY_CN[m["key"]], "value": m["value"], "status": m["status"]})
        if updates:
            log.info("memory updated", extra={"updates": [(u["action"], u["key"], u["value"]) for u in updates]})
        return updates

    def record(self, session_id: str, user_id: str, role: str, content: str, route: str = "") -> None:
        if session_id and content:
            self.store.add_turn(session_id, user_id, role, content, route)

    def summarize_later(self, session_id: str) -> None:
        """摘要放到后台，不阻塞当前回复。"""
        if not session_id:
            return
        task = asyncio.create_task(self.session.maybe_summarize(session_id))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    def list(self, user_id: str) -> List[Dict]:
        return [{"id": m["id"], "key": m["key"], "key_cn": KEY_CN[m["key"]], "value": m["value"],
                 "status": m["status"], "evidence": m["evidence"], "updated_at": m["updated_at"]}
                for m in self.store.list(user_id)]
