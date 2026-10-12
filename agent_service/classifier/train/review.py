"""测试集标签的人工盲审：导出抽检表（不显示原标签）→ 人工填写 → 计算一致率。

python -m classifier.train.review export
python -m classifier.train.review score
"""
import csv
import json
import random
import sys
from collections import Counter

from classifier.labels import ROUTE_DESC, ROUTE_LABELS, STYLE_DESC, STYLE_LABELS
from classifier.train.train import DATA_DIR, load

N = 50
SEED = 7


def export() -> None:
    rng = random.Random(SEED)
    for task in ("style", "route"):
        path = DATA_DIR / f"review_{task}.csv"
        if path.exists() and any(r[2:] and any(c.strip() for c in r[2:]) for r in list(csv.reader(path.open(encoding="utf-8-sig")))[1:]):
            print(f"{path.name} 已有人工填写，跳过")
            continue
        test = load(task, "test")
        # 分层抽样：按标签（风格按首个标签/无标签）轮流取，保证各类都有
        key = (lambda it: (it["labels"] or ["none"])[0]) if task == "style" else (lambda it: it["label"])
        groups = {}
        for i, it in enumerate(test):
            groups.setdefault(key(it), []).append(i)
        for g in groups.values():
            rng.shuffle(g)
        picked = []
        while len(picked) < min(N, len(test)):
            for g in groups.values():
                if g and len(picked) < N:
                    picked.append(g.pop())
        rng.shuffle(picked)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            if task == "style":
                w.writerow(["no", "text"] + [f"{l}（{STYLE_DESC[l].split('：')[0]}）" for l in STYLE_LABELS]
                           + ["综合（无明显风格）"])
            else:
                w.writerow(["no", "text", "label（填 " + " / ".join(ROUTE_LABELS) + "）"])
            for no, i in enumerate(picked, start=1):
                w.writerow([no, test[i]["text"]] + [""] * (len(STYLE_LABELS) + 1 if task == "style" else 1))
        (DATA_DIR / f"review_{task}_key.json").write_text(json.dumps(
            {str(no): i for no, i in enumerate(picked, start=1)}), encoding="utf-8")
        print(f"{path.name}: {len(picked)} 行")
    print("\n风格：适用的列填 1；没有明显风格就在“综合”列填 1。每行至少填一个。")
    print("路由：" + "；".join(f"{k}={v}" for k, v in ROUTE_DESC.items()))


def score() -> None:
    summary = {}
    for task in ("style", "route"):
        path = DATA_DIR / f"review_{task}.csv"
        test = load(task, "test")
        key = json.loads((DATA_DIR / f"review_{task}_key.json").read_text(encoding="utf-8"))
        rows = list(csv.reader(path.open(encoding="utf-8-sig")))[1:]
        if task == "style":
            filled = [r for r in rows if any(c.strip() for c in r[2:])]  # 每行至少填一列（含“综合”）才算已标
            if not filled:
                print("[style] 尚未填写")
                continue
            exact = label_agree = label_total = 0
            disagreements = Counter()
            for r in filled:
                human = {l for l, c in zip(STYLE_LABELS, r[2:]) if c.strip() in ("1", "是", "y", "Y", "√")}
                gen = set(test[key[r[0]]]["labels"])
                exact += human == gen
                for l in STYLE_LABELS:
                    label_total += 1
                    label_agree += (l in human) == (l in gen)
                    if (l in human) != (l in gen):
                        disagreements[f"{l}:{'人工有' if l in human else '人工无'}"] += 1
            summary["style"] = (f"{len(filled)} 条，整行完全一致 {exact / len(filled):.1%}，逐标签一致 "
                                f"{label_agree / label_total:.1%}；分歧最多 {disagreements.most_common(3)}")
            print(f"[style] {len(filled)} 行：整行完全一致 {exact / len(filled):.1%}，逐标签一致 {label_agree / label_total:.1%}")
            print("  分歧最多：", disagreements.most_common(6))
        else:
            filled = [r for r in rows if r[2].strip()]
            if not filled:
                print("[route] 尚未填写")
                continue
            agree = sum(r[2].strip() == test[key[r[0]]]["label"] for r in filled)
            confusions = Counter(f"{test[key[r[0]]]['label']}→{r[2].strip()}" for r in filled
                                 if r[2].strip() != test[key[r[0]]]["label"])
            summary["route"] = f"{len(filled)} 条，一致 {agree / len(filled):.1%}；分歧 {confusions.most_common(3)}"
            print(f"[route] {len(filled)} 行：一致 {agree / len(filled):.1%}；分歧（生成标签→人工）：{confusions.most_common(6)}")
    if len(summary) == 2:
        (DATA_DIR / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "export"]()
