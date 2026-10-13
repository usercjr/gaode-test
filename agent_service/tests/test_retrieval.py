from pathlib import Path

from rag.bm25 import BM25Index, rrf_fuse, tokenize
from rag.ingest import load_csv_records, note_id_for
from rag.metadata import PlaceMatcher, normalize_place, note_places


def test_rrf_rewards_agreement_between_rankers():
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60))
    # b 在两路中都靠前，应排第一；只出现在一路的 a、d 排在后面
    assert max(fused, key=fused.get) == "b"
    assert fused["c"] > fused["a"] and fused["c"] > fused["d"]


def test_tokenize_drops_stopwords_and_punctuation():
    toks = tokenize("成都的火锅真的很好吃！！推荐")
    assert "成都" in toks and "火锅" in toks
    assert "的" not in toks and "真的" not in toks and "推荐" not in toks
    assert all(t.strip() for t in toks)


def test_bm25_search_and_row_filter():
    idx = BM25Index(
        ["c1", "c2", "c3"],
        ["成都火锅 推荐 老店", "重庆 小面 火锅", "北京 故宫 预约"],
        ["n1", "n2", "n3"],
    )
    ids = [cid for cid, _ in idx.search("火锅", 3)]
    assert set(ids) == {"c1", "c2"}
    assert [cid for cid, _ in idx.search("火锅", 3, allowed_rows={"n2"})] == ["c2"]
    assert idx.search("的", 3) == []  # 全是停用词


def test_place_matcher_longest_match_and_normalization():
    assert normalize_place("成都市") == "成都"
    assert normalize_place("新疆") == "新疆"
    meta = {"cities": ["乌鲁木齐市", "成都"], "provinces": ["新疆维吾尔自治区"]}
    assert "乌鲁木齐" in note_places(meta)
    m = PlaceMatcher(["西安", "安顺", "乌鲁木齐", "成都"])
    assert m.find("乌鲁木齐和成都哪个好玩") == ["乌鲁木齐", "成都"]
    assert m.find("周末去哪玩") == []


def test_load_csv_records_dedupes_and_skips_repeated_header(tmp_path: Path):
    csv_path = tmp_path / "notes.csv"
    csv_path.write_text(
        "id,title,text\n"
        "row_0,标题A,正文A\n"
        "id,title,text\n"          # 合并文件时混入的表头
        "row_0,标题B,正文B\n"       # id 重复但内容不同：应保留
        "row_9,标题A,正文A\n",      # 内容重复：应去掉
        encoding="utf-8",
    )
    records = load_csv_records(csv_path, "title", "text")
    assert [r["title"] for r in records] == ["标题A", "标题B"]
    assert len({r["row_id"] for r in records}) == 2
    assert records[0]["row_id"] == note_id_for("标题A", "正文A")
    assert records[1]["csv_id"] == "row_0"
