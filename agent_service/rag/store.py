"""向量库：embedding + ChromaDB。全部为同步 CPU/IO 操作，异步层通过线程池调用。"""
import hashlib
import threading
from typing import Any, Dict, List, Optional, Set

import chromadb
from sentence_transformers import SentenceTransformer

from config import RAGConfig
from core.logging import get_logger
from rag.chunking import clean_text, split_text

log = get_logger("rag.store")


class VectorStore:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self._embedder: Optional[SentenceTransformer] = None
        self._client = None
        self._collection = None
        self._lock = threading.Lock()

    # ---------- 生命周期 ----------
    def ensure_ready(self) -> None:
        if self._collection is not None:
            return
        with self._lock:
            if self._embedder is None:
                log.info("loading embedding model", extra={"model": self.cfg.embedding_model})
                self._embedder = SentenceTransformer(self.cfg.embedding_model)
            if self._client is None:
                self.cfg.db_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(self.cfg.db_dir))
            if self._collection is None:
                self._collection = self._client.get_or_create_collection(
                    name=self.cfg.collection, metadata={"hnsw:space": "cosine"}
                )

    @property
    def ready(self) -> bool:
        return self._collection is not None

    def count(self) -> int:
        self.ensure_ready()
        return int(self._collection.count())

    def reset(self) -> None:
        self.ensure_ready()
        try:
            self._client.delete_collection(self.cfg.collection)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.cfg.collection, metadata={"hnsw:space": "cosine"}
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        self.ensure_ready()
        return self._embedder.encode(texts, normalize_embeddings=True).tolist()

    # ---------- 入库 ----------
    def ingest_records(
        self,
        records: List[Dict[str, Any]],
        source: str = "csv",
        reset: bool = False,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        self.ensure_ready()
        if reset:
            self.reset()

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        skipped = 0

        for i, row in enumerate(records):
            row_id = row.get("row_id", i)
            title = clean_text(str(row.get("title", "")))
            text = clean_text(str(row.get("text", "")))
            parts = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap) if text else []
            if not parts:
                skipped += 1
                continue
            for idx, chunk in enumerate(parts):
                payload_text = f"标题：{title or '无标题'}\n正文：{chunk}" if title else chunk
                # 稳定 ID：同一来源同一内容重复导入时 upsert 覆盖，不产生重复
                doc_id = hashlib.md5(f"{source}|{row_id}|{idx}|{payload_text}".encode("utf-8")).hexdigest()
                ids.append(doc_id)
                docs.append(payload_text)
                metas.append(
                    {
                        "source": source,
                        "row_id": str(row_id),
                        "csv_id": str(row.get("csv_id", "")),
                        "title": title,
                        "chunk_index": idx,
                    }
                )

        for st in range(0, len(docs), batch_size):
            ed = min(len(docs), st + batch_size)
            self._collection.upsert(
                ids=ids[st:ed],
                documents=docs[st:ed],
                embeddings=self.embed(docs[st:ed]),
                metadatas=metas[st:ed],
            )

        result = {
            "upserted_chunks": len(docs),
            "skipped_rows": skipped,
            "collection_count": int(self._collection.count()),
        }
        log.info("ingest done", extra={"source": source, **result})
        return result

    # ---------- 读取 ----------
    def all_chunks(self) -> Dict[str, List[Any]]:
        """全部 chunk（用于构建 BM25 索引）：{ids, documents, metadatas}。"""
        self.ensure_ready()
        res = self._collection.get(include=["documents", "metadatas"])
        return {"ids": res["ids"], "documents": res["documents"], "metadatas": res["metadatas"]}

    def get_chunks(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not chunk_ids:
            return {}
        self.ensure_ready()
        res = self._collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        return {
            cid: {"text": doc or "", "metadata": md or {}}
            for cid, doc, md in zip(res["ids"], res["documents"], res["metadatas"])
        }

    # ---------- 召回 ----------
    def recall(
        self, query: str, n_results: int = 12, allowed_rows: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """向量召回。allowed_rows 不为空时只在这些笔记的 chunk 中检索。"""
        self.ensure_ready()
        where = {"row_id": {"$in": sorted(allowed_rows)}} if allowed_rows else None
        res = self._collection.query(
            query_embeddings=self.embed([query]),
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        hits: List[Dict[str, Any]] = []
        for i, doc in enumerate(docs):
            md = (metas[i] if i < len(metas) else None) or {}
            dist = distances[i] if i < len(distances) else None
            hits.append(
                {
                    "idx": i,
                    "chunk_id": ids[i],
                    "row_id": md.get("row_id", ""),
                    "title": md.get("title") or "",
                    "text": doc or "",
                    "distance": float(dist) if dist is not None else None,
                    "metadata": md,
                }
            )
        return hits
