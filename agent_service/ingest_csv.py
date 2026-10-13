"""命令行入库：python ingest_csv.py --csv ../data/travel_notes.csv --reset"""
import argparse
from pathlib import Path

from config import get_settings
from core.logging import setup_logging
from rag.ingest import load_csv_records
from rag.store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Ingest CSV into ChromaDB for RAG")
    parser.add_argument("--csv", required=True, help="CSV file path")
    parser.add_argument("--title-col", default="title", help="title column name")
    parser.add_argument("--text-col", default="text", help="text/content column name")
    parser.add_argument("--source", default="", help="source tag (default: file name)")
    parser.add_argument("--reset", action="store_true", help="reset collection before ingest")
    parser.add_argument("--chunk-size", type=int, default=400, help="chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="chunk overlap")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.logging.level)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    try:
        records = load_csv_records(csv_path, args.title_col, args.text_col, args.encoding)
    except ValueError as e:
        raise SystemExit(str(e))
    if not records:
        raise SystemExit("No valid rows to ingest.")

    result = VectorStore(settings.rag).ingest_records(
        records=records,
        source=args.source or csv_path.name,
        reset=args.reset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print("Ingest done:")
    for k, v in {"rows": len(records), **result}.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
