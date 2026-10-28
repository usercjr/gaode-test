"""混合检索器：地名过滤 → 向量 / BM25 召回 → RRF 融合 → 交叉编码器重排。

全部为同步 CPU 操作，异步层通过线程池调用。每一步都可通过参数开关，供评测脚本做消融对比。
"""
import copy
import threading
import time
from typing import Any, Dict, List, Optional, Set

from config import RetrievalConfig
from core.cache import TTLCache
from core.logging import get_logger
from rag.bm25 import BM25Index, add_user_words, rrf_fuse
from rag.metadata import PlaceMatcher, load_metadata, note_places
from rag.rerank import CrossEncoderReranker
from rag.store import VectorStore

log = get_logger("rag.retriever")


class Retriever:
    def __init__(self, store: VectorStore, cfg: RetrievalConfig):
        self.store = store
        self.cfg = cfg
        self.reranker = CrossEncoderReranker(cfg.reranker_model, cfg.reranker_max_length)
        self.metadata: Dict[str, dict] = {}
        self.matcher = PlaceMatcher([])
        self._rows_by_place: Dict[str, Set[str]] = {}
        self._bm25: Optional[BM25Index] = None
        self._lock = threading.Lock()
        self.cache = TTLCache(cfg.cache_size, cfg.cache_ttl_s)

    # ---------- 构建 ----------
    def load(self) -> None:
        """加载元数据、构建 BM25 索引；入库后需要重新调用。"""
        with self._lock:
            self.cache.clear()  # 语料或元数据变了，旧结果作废
            self.metadata = load_metadata(self.cfg.metadata_path)
            self.matcher = PlaceMatcher.from_metadata(self.metadata)
            rows_by_place: Dict[str, Set[str]] = {}
            vocab: Set[str] = set()
            for note_id, meta in self.metadata.items():
                places = note_places(meta)
                vocab |= places | set(meta.get("pois", []))
                for p in places:
                    rows_by_place.setdefault(p, set()).add(note_id)
            self._rows_by_place = rows_by_place
            add_user_words(vocab)

            chunks = self.store.all_chunks()
            row_ids = [(md or {}).get("row_id", "") for md in chunks["metadatas"]]
            self._bm25 = BM25Index(chunks["ids"], chunks["documents"], row_ids)
            log.info(
                "retriever loaded",
                extra={"chunks": len(self._bm25), "notes_with_metadata": len(self.metadata), "places": len(self.matcher.places)},
            )

    @property
    def ready(self) -> bool:
        return self._bm25 is not None

    def allowed_rows_for(self, places: List[str]) -> Optional[Set[str]]:
        rows: Set[str] = set()
        for p in places:
            rows |= self._rows_by_place.get(p, set())
        return rows or None

    # ---------- 检索 ----------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        mode: Optional[str] = None,
        rerank: Optional[str] = None,
        city_filter: Optional[bool] = None,
        candidate_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """返回 {hits, places, filtered, timings_ms}。rerank="llm" 时这里不重排，交给异步层处理。"""
        if not self.ready:
            self.load()
        mode = mode or self.cfg.mode
        rerank = rerank or self.cfg.rerank
        city_filter = self.cfg.city_filter if city_filter is None else city_filter
        candidate_k = candidate_k or self.cfg.candidate_k
        key = (query, top_k, mode, rerank, city_filter, candidate_k, self.cfg.rrf_k, self.cfg.rerank_k)
        cached = self.cache.get(key)
        if cached is not None:
            # 返回副本：调用方会往 hit 里补字段，不能改到缓存里的对象
            return {**copy.deepcopy(cached), "timings_ms": {"cache_hit": 1}}
        result = self._retrieve_uncached(query, top_k, mode, rerank, city_filter, candidate_k)
        self.cache.set(key, copy.deepcopy(result))
        return result

    def _retrieve_uncached(self, query: str, top_k: int, mode: str, rerank: str, city_filter: bool,
                           candidate_k: int) -> Dict[str, Any]:
        timings: Dict[str, int] = {}

        places = self.matcher.find(query) if city_filter else []
        allowed = self.allowed_rows_for(places) if places else None

        candidates = self._recall(query, mode, candidate_k, allowed, timings)
        # 地名过滤后候选不足时，用不过滤的结果补齐，避免语料覆盖不到的城市直接无结果
        if allowed and len(candidates) < top_k:
            seen = {c["chunk_id"] for c in candidates}
            extra = self._recall(query, mode, candidate_k, None, {})
            candidates += [c for c in extra if c["chunk_id"] not in seen]

        if rerank == "cross_encoder" and candidates:
            # 只重排融合结果的前 rerank_k 条，其余保持原顺序排在后面
            t0 = time.perf_counter()
            head, tail = candidates[: self.cfg.rerank_k], candidates[self.cfg.rerank_k :]
            scores = self.reranker.score(query, [c["text"] for c in head])
            for c, s in zip(head, scores):
                c["rerank_score"] = s
            head.sort(key=lambda c: c["rerank_score"], reverse=True)
            candidates = head + tail
            timings["rerank"] = round((time.perf_counter() - t0) * 1000)

        hits = candidates if rerank == "llm" else candidates[:top_k]
        return {"hits": hits, "places": places, "filtered": bool(allowed), "timings_ms": timings}

    def _recall(
        self, query: str, mode: str, k: int, allowed: Optional[Set[str]], timings: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        vector_hits: List[Dict[str, Any]] = []
        bm25_hits: List[tuple] = []

        if mode in ("vector", "hybrid"):
            t0 = time.perf_counter()
            vector_hits = self.store.recall(query, k, allowed)
            timings["vector"] = round((time.perf_counter() - t0) * 1000)
        if mode in ("bm25", "hybrid"):
            t0 = time.perf_counter()
            bm25_hits = self._bm25.search(query, k, allowed)
            timings["bm25"] = round((time.perf_counter() - t0) * 1000)

        by_id: Dict[str, Dict[str, Any]] = {h["chunk_id"]: h for h in vector_hits}
        bm25_scores = dict(bm25_hits)

        if mode == "vector":
            ordered = [h["chunk_id"] for h in vector_hits]
            fused = {}
        elif mode == "bm25":
            ordered = [cid for cid, _ in bm25_hits]
            fused = {}
        else:
            fused = dict(rrf_fuse([[h["chunk_id"] for h in vector_hits], [cid for cid, _ in bm25_hits]], self.cfg.rrf_k))
            ordered = sorted(fused, key=fused.get, reverse=True)[:k]

        # BM25 独有的命中需要回向量库取正文
        missing = [cid for cid in ordered if cid not in by_id]
        for cid, item in self.store.get_chunks(missing).items():
            md = item["metadata"]
            by_id[cid] = {
                "chunk_id": cid,
                "row_id": md.get("row_id", ""),
                "title": md.get("title") or "",
                "text": item["text"],
                "distance": None,
                "metadata": md,
            }

        results = []
        for idx, cid in enumerate(ordered):
            if cid not in by_id:
                continue
            hit = dict(by_id[cid])
            hit["idx"] = idx
            hit["bm25_score"] = bm25_scores.get(cid)
            hit["rrf_score"] = fused.get(cid)
            results.append(hit)
        return results
