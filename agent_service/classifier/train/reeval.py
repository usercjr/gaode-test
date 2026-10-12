"""测试集变化后重新评测已训练好的微调模型（只做推理，不重新训练），并保存验证集/测试集概率供级联分析。

python -m classifier.train.reeval --task style
"""
import argparse
import json
import statistics

import numpy as np

from classifier.model import Predictor
from classifier.train.train import MODELS_DIR, RESULTS_DIR, evaluate, load, split_val


def main(task: str) -> None:
    test = load(task, "test")
    _, val = split_val(load(task, "train"))
    for path in sorted(RESULTS_DIR.glob(f"{task}_finetune_*.json")):
        res = json.loads(path.read_text(encoding="utf-8"))
        short = path.stem.split("_", 2)[-1]
        model_dir = MODELS_DIR / f"{task}_{short}"
        predictor = Predictor(model_dir)
        p_te = np.array(predictor.predict_proba([it["text"] for it in test]))
        p_va = np.array(predictor.predict_proba([it["text"] for it in val]))
        np.save(model_dir / "test_probs.npy", p_te)
        np.save(model_dir / "val_probs.npy", p_va)
        for it in test[:5]:
            predictor.predict(it["text"])  # 预热
        lat = [predictor.predict(it["text"])["latency_ms"] for it in test[:50]]
        res["test"] = evaluate(task, p_te, test, res.get("thresholds"))
        res["n_test"] = len(test)
        res["latency_ms_p50"] = round(statistics.median(lat), 1)
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        key = "micro_f1" if task == "style" else "macro_f1"
        print(f"[{task}/{short}] test {key}={res['test'][key]:.4f} latency_p50={res['latency_ms_p50']}ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["style", "route"], required=True)
    main(ap.parse_args().task)
