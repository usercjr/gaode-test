"""计算人工抽检与 LLM 判分的一致率：python eval/review_agreement.py"""
import csv
from collections import Counter

from common import DATA_DIR, read_jsonl

REVIEW_PATH = DATA_DIR / "rag_review.csv"
KEY_PATH = DATA_DIR / "rag_review_key.jsonl"


def weighted_kappa(pairs, k=3):
    """二次加权 Cohen's kappa：考虑“差 1 分”比“差 2 分”轻。"""
    n = len(pairs)
    if not n:
        return 0.0
    obs = Counter(pairs)
    a = Counter(h for h, _ in pairs)
    b = Counter(m for _, m in pairs)
    w = lambda i, j: (i - j) ** 2 / (k - 1) ** 2
    po = sum(w(i, j) * obs[(i, j)] for i in range(k) for j in range(k)) / n
    pe = sum(w(i, j) * a[i] * b[j] for i in range(k) for j in range(k)) / n ** 2
    return 1 - po / pe if pe else 1.0


def main():
    key = {str(item["no"]): item["llm_grade"] for item in read_jsonl(KEY_PATH)}
    pairs = []
    with REVIEW_PATH.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            human = (row.get("your_grade(0/1/2)") or "").strip()
            if human in {"0", "1", "2"}:
                pairs.append((int(human), key[row["no"]]))
    if not pairs:
        print("还没有填写任何人工分数")
        return
    n = len(pairs)
    exact = sum(h == m for h, m in pairs) / n
    within1 = sum(abs(h - m) <= 1 for h, m in pairs) / n
    binary = sum((h >= 1) == (m >= 1) for h, m in pairs) / n
    strict = sum((h == 2) == (m == 2) for h, m in pairs) / n
    print(f"已填写 {n} 对")
    print(f"完全一致率：{exact:.1%}")
    print(f"相差不超过 1 分：{within1:.1%}")
    print(f"“是否相关(≥1)”一致率：{binary:.1%}")
    print(f"“是否高度相关(=2)”一致率：{strict:.1%}（Recall/MRR 依赖此判断）")
    print(f"二次加权 kappa：{weighted_kappa(pairs):.3f}（>0.6 通常视为较好一致）")
    print("\n混淆矩阵（行=人工，列=LLM）")
    c = Counter(pairs)
    print("      LLM0 LLM1 LLM2")
    for h in range(3):
        print(f"人工{h}  " + "  ".join(f"{c[(h, m)]:4d}" for m in range(3)))


if __name__ == "__main__":
    main()
