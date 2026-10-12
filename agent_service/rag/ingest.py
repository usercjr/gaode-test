"""CSV 读取，供 CLI（ingest_csv.py）与 /ingest/csv 接口共用。"""
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_csv_records(csv_path: Path, title_col: str, text_col: str, encoding: str = "utf-8-sig") -> List[Dict[str, Any]]:
    """读取 CSV 为 [{row_id, title, text}]。列缺失抛 ValueError。"""
    df = pd.read_csv(csv_path, encoding=encoding)
    for col in (title_col, text_col):
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")

    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        title = "" if pd.isna(row[title_col]) else str(row[title_col]).strip()
        text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
        if not text:
            continue
        # 合并多个 CSV 时常混入重复表头行，这类行不是内容
        if title == title_col and text == text_col:
            continue
        row_id = row["id"] if "id" in df.columns and not pd.isna(row["id"]) else idx
        records.append({"row_id": row_id, "title": title, "text": text})
    return records
