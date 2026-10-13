"""汇总 classifier/results/*.json → eval/classifier_report.md

python -m classifier.train.report
"""
import json
from datetime import date
from pathlib import Path

import numpy as np

from classifier.labels import ROUTE_LABELS, STYLE_LABELS
from classifier.train.metrics import multilabel_metrics, singlelabel_metrics, to_multi_hot
from classifier.train.train import MODELS_DIR, RESULTS_DIR, ROOT, load, split_val

REPORT = ROOT.parent / "eval" / "classifier_report.md"
ORDER = {"llm_zero_shot": 0, "baseline": 1, "finetune": 2}


def name(r):
    return {"llm_zero_shot": f"LLM 零样本（{r['base']}）",
            "baseline": f"句向量 + 逻辑回归（{r['base'].split('/')[-1]}）",
            "finetune": f"微调（{r['_short']}{'，' + str(r['best_epoch']) + ' 轮最优' if 'best_epoch' in r else ''}）"}[r["method"]]


def _cascade_curve(task, probs, llm_pred, items, local):
    """返回 [(阈值, 本地占比, 主指标, 次指标)]。"""
    rows = []
    if task == "route":
        local_pred = [ROUTE_LABELS[i] for i in probs.argmax(1)]
        conf = probs.max(1)
        for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01):
            use = conf >= t
            merged = [lp if u else lg for lp, lg, u in zip(local_pred, llm_pred, use)]
            m = singlelabel_metrics([it["label"] for it in items], merged, ROUTE_LABELS)
            rows.append((t, float(use.mean()), m["macro_f1"], m["accuracy"]))
    else:
        th = np.array(local["thresholds"])
        local_bin = (probs >= th).astype(int)
        llm_bin = to_multi_hot(llm_pred, STYLE_LABELS)
        margin = np.abs(probs - th).min(1)  # 离阈值最近的那个标签决定置信度
        y = to_multi_hot([it["labels"] for it in items], STYLE_LABELS)
        for t in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 1.01):
            use = margin >= t
            m = multilabel_metrics(y, np.where(use[:, None], local_bin, llm_bin), STYLE_LABELS)
            rows.append((t, float(use.mean()), m["micro_f1"], m["exact_match"]))
    return rows


def cascade_rows(task: str, local: dict, llm: dict):
    """级联：本地模型置信时直接用，否则交给 LLM。阈值在验证集上选（主指标最高，并列取本地占比更高者），
    测试集只用来报告该阈值下的结果。返回 (测试集曲线, 选中的阈值)。"""
    model_dir = MODELS_DIR / f"{task}_{local['_short']}"
    val_llm_path = RESULTS_DIR.parent / "data" / f"{task}_llm_val_predictions.json"
    if not (model_dir / "test_probs.npy").exists() or "predictions" not in llm:
        return [], None
    test = load(task, "test")
    curve = _cascade_curve(task, np.load(model_dir / "test_probs.npy"), llm["predictions"], test, local)
    chosen = None
    if (model_dir / "val_probs.npy").exists() and val_llm_path.exists():
        _, val = split_val(load(task, "train"))
        val_llm = json.loads(val_llm_path.read_text(encoding="utf-8"))["predictions"]
        val_curve = [r for r in _cascade_curve(task, np.load(model_dir / "val_probs.npy"), val_llm, val, local) if r[0] <= 1]
        chosen = max(val_curve, key=lambda r: (round(r[2], 3), r[1]))[0]
    return curve, chosen


def main():
    results = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        r["_short"] = p.stem.split("_", 2)[-1]  # 文件名 {task}_{method}_{short}
        results.append(r)
    style = sorted([r for r in results if r["task"] == "style"], key=lambda r: ORDER[r["method"]])
    route = sorted([r for r in results if r["task"] == "route"], key=lambda r: ORDER[r["method"]])
    n = lambda rs, k: next((r[k] for r in rs if k in r), 0)
    lines = [
        "# 意图分类评测报告", "",
        f"- 运行日期：{date.today().isoformat()}",
        f"- 数据：LLM 按指定意图生成，训练/测试分开生成（测试集用不同提示词，更口语、带错别字与否定，且只用训练集未出现的 15 个城市），"
        f"测试集中与训练样本 bge 相似度 ≥ 0.90 的已剔除。风格：训练 {n(style, 'n_train') + n(style, 'n_val')} 条（含 10% 验证），测试 {n(style, 'n_test')} 条；"
        f"路由：训练 {n(route, 'n_train') + n(route, 'n_val')} 条，测试 {n(route, 'n_test')} 条。",
        "- 输入格式与线上一致：`目的地：X；偏好：A、B；需求：用户原话`（缺失部分省略），由 `classifier/labels.py:format_input` 统一生成。",
        "- 测试标签来自生成时指定的意图，人工盲审结果见文末。", "",
        "## 风格（多标签，6 类；全部不命中视为“综合”）", "",
        "| 方法 | micro-F1 | macro-F1 | 整行完全正确 | “综合”F1 | 单条延迟 P50 | 训练耗时 | 模型大小 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in style:
        t = r["test"]
        lines.append(f"| {name(r)} | {t['micro_f1']:.3f} | {t['macro_f1']:.3f} | {t['exact_match']:.3f} | {t['none_f1']:.3f} | "
                     f"{r['latency_ms_p50']:.0f} ms | {r.get('train_seconds', '—')}{' s' if 'train_seconds' in r else ''} | "
                     f"{str(r['model_mb']) + ' MB' if 'model_mb' in r else '—'} |")
    lines += ["", "各标签 F1：", "", "| 方法 | " + " | ".join(STYLE_LABELS) + " |", "|---" * (len(STYLE_LABELS) + 1) + "|"]
    for r in style:
        lines.append(f"| {name(r)} | " + " | ".join(f"{r['test']['per_label_f1'][l]:.2f}" for l in STYLE_LABELS) + " |")

    lines += ["", "## 路由（单标签，5 类）", "",
              "| 方法 | 准确率 | macro-F1 | 单条延迟 P50 | 训练耗时 |", "|---|---|---|---|---|"]
    for r in route:
        t = r["test"]
        lines.append(f"| {name(r)} | {t['accuracy']:.3f} | {t['macro_f1']:.3f} | {r['latency_ms_p50']:.0f} ms | "
                     f"{str(r['train_seconds']) + ' s' if 'train_seconds' in r else '—'} |")
    best_route = max(route, key=lambda r: r["test"]["macro_f1"]) if route else None
    if best_route:
        lines += ["", f"混淆矩阵（{name(best_route)}，行=真实，列=预测，顺序 {' / '.join(ROUTE_LABELS)}）：", "", "```"]
        lines += ["  ".join(f"{v:3d}" for v in row) for row in best_route["test"]["confusion"]]
        lines.append("```")

    for task, rs, metric_names in (("style", style, ("micro-F1", "整行完全正确")), ("route", route, ("macro-F1", "准确率"))):
        llm = next((r for r in rs if r["method"] == "llm_zero_shot"), None)
        tuned = [r for r in rs if r["method"] == "finetune"]
        key = "micro_f1" if task == "style" else "macro_f1"
        best = max(tuned, key=lambda r: r["val"][key]) if tuned else None  # 按验证集选模型，不看测试集
        rows, chosen = cascade_rows(task, best, llm) if (best and llm) else ([], None)
        if not rows:
            continue
        local_ms, llm_ms = best["latency_ms_p50"], llm["latency_ms_p50"]
        crit = "离阈值的最小距离 ≥" if task == "style" else "最高概率 ≥"
        lines += ["", f"### 级联：{name(best)} 置信时直接用，否则交给 LLM（{'风格' if task == 'style' else '路由'}）", "",
                  f"按验证集选出的模型：{name(best)}；阈值在验证集上选定为 **{chosen}**（标 ★ 的行），其余行仅展示取舍曲线。", "",
                  f"| 本地置信条件（{crit}） | 本地处理占比 | {metric_names[0]} | {metric_names[1]} | 预估平均延迟 |", "|---|---|---|---|---|"]
        for t, cov, m1, m2 in rows:
            label = "全部本地" if t == 0.0 else "全部 LLM" if t > 1 else f"{t:.2f}"
            label += " ★" if chosen is not None and abs(t - chosen) < 1e-9 else ""
            avg = local_ms + (1 - cov) * llm_ms if t <= 1 else llm_ms
            lines.append(f"| {label} | {cov:.0%} | {m1:.3f} | {m2:.3f} | {avg / 1000:.2f} s |")

    review = RESULTS_DIR.parent / "data" / "review_summary.json"
    lines += ["", "## 测试集标签人工盲审", ""]
    if review.exists():
        s = json.loads(review.read_text(encoding="utf-8"))
        lines.append(f"- 风格：{s['style']}\n- 路由：{s['route']}")
    else:
        lines.append("尚未完成（抽检表 `agent_service/classifier/data/review_style.csv`、`review_route.csv`）。在此之前，测试集指标含有生成标签的噪声。")
    lines += ["", "## 上线前抽查发现的问题", "",
              "- 路由模型只在单句聊天消息上训练，对“目的地：…；偏好：…；需求：…”这种表单拼接输入会给出高置信度的错误结果"
              "（如判成 chitchat 0.99）。处理：/classify 只对聊天原话（text 字段）做路由，表单输入只做风格识别。",
              "- **路由模型在人工编写的消息上明显变差**：P5 用 40 条人工编写的 held-out 消息测得“全部本地”准确率只有 75%，"
              "8 条修改行程请求漏判 6 条（多被判成 chitchat），远低于本报告 LLM 生成测试集上的 95.4%。"
              "线上因此改为置信度 < 0.97 时交给 LLM（阈值在另一组人工 dev 集上选定），held-out 准确率 92.5%，"
              "见 `eval/route_threshold_report.md`。",
              "- “综合”类仍有明显漏判：如“经典景点都去看看就行”被判为 nature + culture + rush。测试集“综合”F1 0.83 掩盖了这类个例；"
              "需要更多“综合”训练样本，或对多标签同时命中 3 个以上时降级为“综合”。"]
    lines += ["", "## 说明", "",
              "- 延迟为单机 CPU（12 线程）单条请求的中位数；LLM 为顺序调用 20 条测得的单次耗时，含网络往返。",
              "- 模型选择与级联阈值都只用验证集决定；测试集只在最后报告一次。",
              "- 训练与测试数据都由 LLM 生成，测试集虽然换了提示词和城市，仍可能与真实用户分布有差距；上线后应收集真实请求抽样标注。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {REPORT}")


if __name__ == "__main__":
    main()
