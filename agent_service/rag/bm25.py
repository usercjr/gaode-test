"""BM25 关键词检索：jieba 分词 + rank_bm25。索引在内存中，由向量库中的全部 chunk 构建。"""
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import jieba
from rank_bm25 import BM25Okapi

jieba.setLogLevel(60)  # 关闭 jieba 的加载提示

# 高频虚词与小红书常见口水词，对区分相关性没有帮助
_STOPWORDS = set(
    "的 了 和 是 在 我 有 也 就 都 去 要 这 那 你 他 她 它 们 吧 啊 呀 吗 呢 哦 嘛 很 还 又 被 把 给 让 "
    "一个 一下 这个 那个 什么 怎么 可以 没有 不是 就是 真的 非常 特别 一定 推荐 攻略 旅游 旅行 笔记 "
    "宝子 姐妹 家人 们 啦 哈 嗯 吖".split()
)
_TOKEN_RE = re.compile(r"[一-鿿A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    tokens = []
    for piece in _TOKEN_RE.findall((text or "").lower()):
        for tok in jieba.lcut(piece):
            tok = tok.strip()
            if tok and tok not in _STOPWORDS and not (len(tok) == 1 and not tok.isalnum()):
                tokens.append(tok)
    return tokens


def add_user_words(words: Iterable[str]) -> None:
    """把城市、景点名加入分词词典，避免“宽窄巷子”被切成“宽窄/巷子”。"""
    for w in words:
        w = (w or "").strip()
        if len(w) >= 2:
            jieba.add_word(w, freq=20000)


class BM25Index:
    def __init__(self, chunk_ids: Sequence[str], texts: Sequence[str], row_ids: Sequence[str]):
        self.chunk_ids = list(chunk_ids)
        self.row_ids = list(row_ids)
        corpus = [tokenize(t) for t in texts]
        # 空文档会让 BM25 的平均长度计算出问题，用占位词兜底
        self._bm25 = BM25Okapi([toks or ["_"] for toks in corpus]) if corpus else None

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def search(self, query: str, k: int, allowed_rows: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        if not self._bm25:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: List[Tuple[str, float]] = []
        for i in order:
            if scores[i] <= 0:
                break
            if allowed_rows is not None and self.row_ids[i] not in allowed_rows:
                continue
            results.append((self.chunk_ids[i], float(scores[i])))
            if len(results) >= k:
                break
        return results


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion：score(d) = Σ 1 / (k + rank)。只依赖名次，不需要统一各路分数尺度。"""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
