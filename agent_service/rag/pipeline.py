"""RAG 流水线：意图改写 → 混合检索（地名过滤 + 向量/BM25 + RRF）→ 重排 → 引用回答。

LLM 步骤失败时逐级降级（不让整个请求失败），并把被跳过的能力记入 degraded，
由接口返回给前端提示用户。
"""
import json
import re
from typing import Any, Dict, List, Optional

from starlette.concurrency import run_in_threadpool

from core.errors import AppError, ErrorCode
from core.llm import LLMClient
from core.logging import get_logger
from rag.retriever import Retriever

log = get_logger("rag.pipeline")

STYLE_KEYWORDS = {
    "Rush_Trip": "打卡 高效 快游",
    "Deep_Cultural_Trip": "文化 历史 博物馆 古迹",
    "Food_Adventure": "美食 餐厅 小吃 探店",
    "Nature_Relaxation": "自然 风景 徒步 放松",
    "City_Leisure_Tour": "城市 漫步 咖啡 街区",
    "Customized_AI_Tour": "旅游 行程 攻略",
    "Vlog_Creator_Trip": "拍摄 vlog 打卡 出片",
    "Photo_Experience": "拍照 机位 出片 打卡",
}


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    for candidate in (raw, (re.search(r"\{[\s\S]*\}", raw) or [None])[0]):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    return None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _similarity(hit: Dict[str, Any]) -> Optional[float]:
    dist = hit.get("distance")
    return None if dist is None else round(max(0.0, 1.0 - float(dist)), 6)


class RagPipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient, llm_model: str):
        self.retriever = retriever
        self.llm = llm
        self.llm_model = llm_model

    async def _retrieve(self, query: str, top_k: int, rerank: str, degraded: List[str]) -> Dict[str, Any]:
        """调用同步检索器；交叉编码器出错时降级为不重排，向量库出错时报 RAG_UNAVAILABLE。"""
        try:
            return await run_in_threadpool(self.retriever.retrieve, query, top_k, rerank=rerank)
        except Exception as e:
            if rerank == "cross_encoder":
                log.exception("rerank failed, retrying without rerank")
                degraded.append("rerank")
                try:
                    return await run_in_threadpool(self.retriever.retrieve, query, top_k, rerank="none")
                except Exception as e2:
                    e = e2
            log.exception("retrieval failed")
            raise AppError(ErrorCode.RAG_UNAVAILABLE, detail=str(e)) from e

    async def intent_understand(self, query: str, default_top_k: int, degraded: List[str]) -> Dict[str, Any]:
        fallback = {
            "intent": "general_search",
            "query_rewrite": query,
            "top_k": max(6, min(30, default_top_k)),
            "must_have": [],
            "exclude": [],
        }
        prompt = (
            "你是检索意图解析器。请将用户问题改写为适合向量检索的查询词，并给出召回参数。"
            "只返回 JSON，字段必须包含：intent, query_rewrite, top_k, must_have, exclude。\n"
            "top_k 范围 6~30。\n"
            f"用户问题：{query}"
        )
        try:
            raw = await self.llm.chat(
                [{"role": "user", "content": prompt}],
                model=self.llm_model, temperature=0.0, max_tokens=400, purpose="rag.intent",
            )
        except AppError:
            degraded.append("query_rewrite")
            return fallback
        obj = extract_json(raw) or {}
        return {
            "intent": str(obj.get("intent") or fallback["intent"]),
            "query_rewrite": str(obj.get("query_rewrite") or query),
            "top_k": max(6, min(30, _safe_int(obj.get("top_k"), fallback["top_k"]))),
            "must_have": obj.get("must_have") if isinstance(obj.get("must_have"), list) else [],
            "exclude": obj.get("exclude") if isinstance(obj.get("exclude"), list) else [],
        }

    async def llm_rerank(
        self, user_query: str, candidates: List[Dict[str, Any]], top_k: int, degraded: List[str]
    ) -> List[Dict[str, Any]]:
        by_distance = sorted(candidates, key=lambda c: c["distance"] if c.get("distance") is not None else 1e9)
        if len(candidates) <= 1:
            return by_distance[:top_k]
        lines = [f"[{c['idx']}] 标题: {c.get('title', '')}\n内容: {c.get('text', '')[:260]}" for c in candidates]
        candidate_block = "\n\n".join(lines)
        prompt = (
            "你是检索重排器。请根据用户问题从候选片段里选出最相关结果，返回 JSON。\n"
            '格式: {"ranked_ids": [idx1, idx2, ...]}\n'
            f"用户问题: {user_query}\n"
            f"候选片段:\n\n{candidate_block}"
        )
        try:
            raw = await self.llm.chat(
                [{"role": "user", "content": prompt}],
                model=self.llm_model, temperature=0.0, max_tokens=500, purpose="rag.rerank",
            )
        except AppError:
            degraded.append("llm_rerank")
            return by_distance[:top_k]

        ranked_ids = (extract_json(raw) or {}).get("ranked_ids")
        if not isinstance(ranked_ids, list):
            degraded.append("llm_rerank")
            return by_distance[:top_k]
        rank_map = {}
        for pos, v in enumerate(ranked_ids):
            idx = _safe_int(v, -1)
            if idx >= 0 and idx not in rank_map:
                rank_map[idx] = pos
        # 未被 LLM 选中的候选排在后面，并保持向量距离顺序
        ordered = sorted(by_distance, key=lambda c: rank_map.get(c["idx"], 10_000))
        return ordered[:top_k]

    async def answer_with_context(self, user_query: str, contexts: List[Dict[str, Any]]) -> str:
        if not contexts:
            return "没有在知识库中找到足够相关的内容。"
        context_block = "\n\n".join(
            f"[{i}] 标题: {c.get('title') or '无标题'}\n{c.get('text')}" for i, c in enumerate(contexts, start=1)
        )
        prompt = (
            "你是知识库问答助手。请仅基于给定片段回答用户问题，结论要简洁。"
            "如果片段不足以回答，要明确指出。回答中尽量附上引用编号，如 [1][2]。\n\n"
            f"用户问题：{user_query}\n\n"
            f"知识片段：\n\n{context_block}"
        )
        return await self.llm.chat(
            [{"role": "user", "content": prompt}],
            model=self.llm_model, temperature=0.2, max_tokens=1000, purpose="rag.answer",
        )

    async def retrieve_context(
        self, query: str, top_k: int = 5, style_label: str = "", use_rerank: bool = True
    ) -> Dict[str, Any]:
        degraded: List[str] = []
        recall_query = query.strip()
        extra = STYLE_KEYWORDS.get(style_label or "", "")
        if extra:
            recall_query = f"{recall_query} {extra}"

        rerank = self.retriever.cfg.rerank if use_rerank else "none"
        result = await self._retrieve(recall_query, top_k, rerank, degraded)
        hits = result["hits"]
        if rerank == "llm":
            hits = await self.llm_rerank(query, hits, top_k, degraded)

        return {
            "query": query,
            "hits": [_public_hit(h) for h in hits],
            "count": len(hits),
            "places": result["places"],
            "degraded": degraded,
        }

    async def search(self, query: str, top_k: int = 5, rerank_k: int = 10) -> Dict[str, Any]:
        degraded: List[str] = []
        intent = await self.intent_understand(query, rerank_k, degraded)
        rerank = self.retriever.cfg.rerank
        result = await self._retrieve(intent.get("query_rewrite") or query, max(top_k, 1), rerank, degraded)
        hits = result["hits"]
        if rerank == "llm":
            hits = await self.llm_rerank(query, hits, max(top_k, 1), degraded)

        try:
            answer = await self.answer_with_context(query, hits)
        except AppError as e:
            # 回答生成失败时仍返回召回结果，前端可展示片段
            degraded.append("answer")
            answer = f"回答生成失败：{e.message}"

        return {
            "query": query,
            "intent": intent,
            "places": result["places"],
            "hits": [{"rank": i, **_public_hit(h)} for i, h in enumerate(hits, start=1)],
            "answer": answer,
            "degraded": degraded,
        }


def _public_hit(h: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "note_id": h.get("row_id", ""),
        "title": h.get("title", ""),
        "text": h.get("text", ""),
        "similarity": _similarity(h),
        "rerank_score": round(h["rerank_score"], 4) if h.get("rerank_score") is not None else None,
    }
