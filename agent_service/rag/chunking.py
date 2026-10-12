"""文本清洗与递归分块（按段落/句号/逗号等分隔符在块尾附近回退切分）。"""
import re
from typing import List

_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def split_text(text: str, chunk_size: int = 400, chunk_overlap: int = 80) -> List[str]:
    src = (text or "").strip()
    if not src:
        return []
    if chunk_size <= 0:
        return [src]
    chunk_overlap = max(0, chunk_overlap)
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    chunks: List[str] = []
    start = 0
    n = len(src)
    while start < n:
        end = min(n, start + chunk_size)
        segment = src[start:end]
        if end < n:
            # 在块的后 45% 范围内寻找最靠后的分隔符，尽量不切断句子
            for sep in _SEPARATORS:
                idx = segment.rfind(sep)
                if idx >= int(chunk_size * 0.55):
                    end = start + idx + len(sep)
                    segment = src[start:end]
                    break

        segment = segment.strip()
        if segment:
            chunks.append(segment)
        if end >= n:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks
