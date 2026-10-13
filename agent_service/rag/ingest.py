"""CSV 读取，供 CLI（ingest_csv.py）、元数据抽取与 /ingest/csv 接口共用。"""
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def note_id_for(title: str, text: str) -> str:
    """笔记 ID 由内容哈希生成：CSV 的 id 列在多文件合并后会重复，不能作为主键。"""
    return "n_" + hashlib.md5(f"{title}\n{text}".encode("utf-8")).hexdigest()[:12]


def load_csv_records(csv_path: Path, title_col: str, text_col: str, encoding: str = "utf-8-sig") -> List[Dict[str, Any]]:
    """读取 CSV 为 [{row_id, csv_id, title, text}]，按内容去重。列缺失抛 ValueError。"""
    df = pd.read_csv(csv_path, encoding=encoding)
    for col in (title_col, text_col):
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")

    records: List[Dict[str, Any]] = []
    seen = set()
    for idx, row in df.iterrows():
        title = "" if pd.isna(row[title_col]) else str(row[title_col]).strip()
        text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
        if not text:
            continue
        # 合并多个 CSV 时常混入重复表头行，这类行不是内容
        if title == title_col and text == text_col:
            continue
        note_id = note_id_for(title, text)
        if note_id in seen:
            continue
        seen.add(note_id)
        csv_id = row["id"] if "id" in df.columns and not pd.isna(row["id"]) else idx
        records.append({"row_id": note_id, "csv_id": str(csv_id), "title": title, "text": text})
    return records
