"""交叉编码器重排（bge-reranker）：对 (查询, 片段) 成对打分，比向量相似度更准但更慢，只用于候选集。"""
import threading
from typing import List, Optional

from sentence_transformers import CrossEncoder

from core.logging import get_logger

log = get_logger("rag.rerank")


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self._model: Optional[CrossEncoder] = None
        self._lock = threading.Lock()

    def ensure_ready(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                log.info("loading reranker", extra={"model": self.model_name})
                self._model = CrossEncoder(self.model_name, max_length=self.max_length)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def score(self, query: str, texts: List[str], batch_size: int = 16) -> List[float]:
        if not texts:
            return []
        self.ensure_ready()
        scores = self._model.predict([(query, t) for t in texts], batch_size=batch_size, show_progress_bar=False)
        return [float(s) for s in scores]
