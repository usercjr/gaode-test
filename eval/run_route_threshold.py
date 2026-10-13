"""路由置信度阈值：本地模型最高概率 < 阈值时交给 LLM 判断。

协议：阈值只在 dev 集（router_messages.jsonl，40 条人工消息）上选；
held-out 集（router_messages_holdout.jsonl，选阈值前写好并已提交）只报告一次。

python eval/run_route_threshold.py
"""
import asyncio
from datetime import date
from typing import Dict, List

from common import DATA_DIR, ROOT, get_settings, read_jsonl

from classifier.llm_zero_shot import classify_route
from classifier.model import Predictor
from core.llm import LLMClient
from core.logging import setup_logging

THRESHOLDS = [0.0, 0.8, 0.9, 0.95, 0.97, 0.99]
REPORT = ROOT / "eval" / "route_threshold_report.md"


def evaluate(items: List[dict], local: List[dict], llm: List[str], t: float) -> Dict[str, float]:
    preds = [l["label"] if l["score"] >= t else g for l, g in zip(local, llm)]
    plan_items = [i for i, it in enumerate(items) if it["expected"] == "plan"]
    return {
        "acc": sum(p == it["expected"] for p, it in zip(preds, items)) / len(items),
        "coverage": sum(l["score"] >= t for l in local) / len(items),
        "plan_missed": sum(preds[i] != "plan" for i in plan_items),  # 想改行程却没改，代价最高
        "errors": [f"{it['message']}（{it['expected']}→{p}）" for p, it in zip(preds, items) if p != it["expected"]],
    }


async def main() -> None:
    settings = get_settings()
    setup_logging("WARNING")
    predictor = Predictor(settings.classifier.models_dir / settings.classifier.route_model)
    llm = LLMClient(settings)
    sets = {"dev": read_jsonl(DATA_DIR / "router_messages.jsonl"),
            "holdout": read_jsonl(DATA_DIR / "router_messages_holdout.jsonl")}
    local = {k: [predictor.predict(it["message"]) for it in v] for k, v in sets.items()}
    llm_pred = {}
    for k, v in sets.items():
        res = await asyncio.gather(*[classify_route(llm, it["message"]) for it in v])
        llm_pred[k] = [r["route"] for r in res]
    await llm.aclose()

    dev = {t: evaluate(sets["dev"], local["dev"], llm_pred["dev"], t) for t in THRESHOLDS}
    chosen = max(THRESHOLDS, key=lambda t: (round(dev[t]["acc"], 4), -dev[t]["plan_missed"], dev[t]["coverage"]))
    hold = {t: evaluate(sets["holdout"], local["holdout"], llm_pred["holdout"], t) for t in (0.0, chosen)}
    hold_llm = evaluate(sets["holdout"], local["holdout"], llm_pred["holdout"], 1.01)

    lines = ["# 路由置信度阈值", "", f"- 运行日期：{date.today().isoformat()}",
             "- 规则：本地路由模型最高概率 < 阈值时交给 LLM 零样本判断",
             "- 阈值只在 dev 集（40 条人工消息）上选择；held-out 集（40 条，选阈值前编写并提交，见 git 历史）只报告一次",
             "- 选择标准：准确率最高；并列时“想改行程却没改”的条数更少者优先，再看本地处理占比", "",
             "## dev 集上的扫描", "", "| 阈值 | 本地处理占比 | 准确率 | 修改类漏判 |", "|---|---|---|---|"]
    for t in THRESHOLDS:
        d = dev[t]
        lines.append(f"| {t:.2f}{' ★' if t == chosen else ''} | {d['coverage']:.0%} | {d['acc']:.1%} | {d['plan_missed']}/8 |")
    lines += ["", f"选定阈值：**{chosen}**", "", "## held-out 集（只看这一次）", "",
              "| 方案 | 本地处理占比 | 准确率 | 修改类漏判 |", "|---|---|---|---|",
              f"| 全部本地（阈值 0） | 100% | {hold[0.0]['acc']:.1%} | {hold[0.0]['plan_missed']}/8 |",
              f"| 级联（阈值 {chosen}） | {hold[chosen]['coverage']:.0%} | {hold[chosen]['acc']:.1%} | {hold[chosen]['plan_missed']}/8 |",
              f"| 全部 LLM | 0% | {hold_llm['acc']:.1%} | {hold_llm['plan_missed']}/8 |",
              "", "held-out 上级联方案的错误：", ""]
    lines += [f"- {e}" for e in hold[chosen]["errors"]] or ["- 无"]
    lines += ["", "## 说明", "", "- 两个集合都由开发者编写和标注，规模小（各 40 条），只能说明趋势。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
