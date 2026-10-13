import asyncio
from pathlib import Path

import numpy as np

from classifier.labels import LEGACY_DEFAULT, STYLE_LABELS, format_input
from classifier.service import ClassifierService
from classifier.train.metrics import multilabel_metrics, singlelabel_metrics, to_multi_hot, tune_thresholds


def test_format_input_matches_training_format():
    assert format_input("成都", ["美食探店", " "], "想吃火锅") == "目的地：成都；偏好：美食探店；需求：想吃火锅"
    assert format_input("", [], "随便逛逛") == "需求：随便逛逛"
    assert format_input("", [], "") == ""


def test_multilabel_metrics_and_none_class():
    y_true = to_multi_hot([["food"], ["food", "photo"], []], STYLE_LABELS)
    y_pred = to_multi_hot([["food"], ["food"], []], STYLE_LABELS)
    m = multilabel_metrics(y_true, y_pred, STYLE_LABELS)
    assert m["exact_match"] == 2 / 3
    assert m["none_f1"] == 1.0  # 第三条“综合”预测正确
    assert m["per_label_f1"]["food"] == 1.0 and m["per_label_f1"]["photo"] == 0.0


def test_singlelabel_metrics():
    m = singlelabel_metrics(["plan", "plan", "chitchat"], ["plan", "notes_qa", "chitchat"],
                            ["plan", "notes_qa", "graph_qa", "realtime", "chitchat"])
    assert abs(m["accuracy"] - 2 / 3) < 1e-9
    assert m["confusion"][0][1] == 1  # 一条 plan 被判成 notes_qa


def test_tune_thresholds_picks_separating_value():
    y = np.array([[1], [1], [0], [0]])
    probs = np.array([[0.7], [0.65], [0.4], [0.3]])
    t = tune_thresholds(y, probs)[0]
    assert 0.4 < t <= 0.65


class FakeLLM:
    async def chat(self, messages, **kw):
        return '{"label": "Food_Adventure", "score": 0.9}'


def test_service_falls_back_to_llm_when_model_missing(tmp_path: Path):
    svc = ClassifierService(FakeLLM(), tmp_path, "missing_style", "missing_route")
    svc.load()
    assert svc.status()["style_model"] is False
    res = asyncio.run(svc.classify("目的地：成都；偏好：美食探店"))
    assert res["label"] == "Food_Adventure" and res["source"] == "llm" and res["labels"] == []


class FakePredictor:
    def __init__(self, labels, scores):
        self._r = {"labels": labels, "scores": scores, "latency_ms": 3.0}

    def predict(self, text):
        return self._r


def test_service_maps_multilabel_to_legacy_prompt_label():
    svc = ClassifierService(FakeLLM(), Path("."), "a", "b")
    scores = {l: 0.1 for l in STYLE_LABELS} | {"food": 0.7, "photo": 0.9}
    svc.style = FakePredictor(["food", "photo"], scores)
    res = asyncio.run(svc.classify("x"))
    assert res["label"] == "Photo_Experience"  # 取多标签中得分最高的映射到前端旧标签
    assert res["labels"] == ["food", "photo"] and res["source"] == "local"

    svc.style = FakePredictor([], {l: 0.2 for l in STYLE_LABELS})
    assert asyncio.run(svc.classify("x"))["label"] == LEGACY_DEFAULT


def test_route_only_runs_for_chat_messages():
    svc = ClassifierService(FakeLLM(), Path("."), "a", "b")
    svc.style = FakePredictor(["food"], {l: 0.1 for l in STYLE_LABELS} | {"food": 0.9})
    svc.route = type("R", (), {"predict": lambda self, t: {"label": "graph_qa", "score": 0.97}})()
    assert "route" not in asyncio.run(svc.classify("目的地：成都；偏好：美食探店"))
    assert asyncio.run(svc.classify("宽窄巷子附近有什么好吃的", with_route=True))["route"] == "graph_qa"
