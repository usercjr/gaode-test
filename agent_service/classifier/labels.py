"""意图分类的标签体系与统一输入格式。训练数据、在线推理、评测都必须经过 format_input，保证分布一致。"""
from typing import Dict, Iterable, List

# 风格：多标签（一个需求可以同时是“美食 + 拍照”）；全部低于阈值视为“综合”
STYLE_LABELS: List[str] = ["food", "nature", "culture", "citywalk", "photo", "rush"]
STYLE_DESC: Dict[str, str] = {
    "food": "美食探店：以吃为主线，餐厅、小吃、夜市、早茶",
    "nature": "自然放松：山水风景、徒步、慢节奏、疗愈、避开人群",
    "culture": "人文历史：博物馆、古迹、非遗、历史街区、深度讲解",
    "citywalk": "城市漫步：街区闲逛、咖啡馆、小店、市井生活",
    "photo": "拍照出片 / vlog：机位、光线、穿搭、拍视频",
    "rush": "特种兵打卡：时间紧、景点多、高效率",
}

# 路由：单标签，供动态路由使用
ROUTE_LABELS: List[str] = ["plan", "notes_qa", "graph_qa", "realtime", "chitchat"]
ROUTE_DESC: Dict[str, str] = {
    "plan": "生成或修改行程（排几天、怎么安排、把第二天改轻松些）",
    "notes_qa": "经验类问答（要注意什么、值不值得去、怎么避坑、预算够不够）",
    "graph_qa": "地点关系类问答（某地附近有什么、A 和 B 顺不顺路、去了 A 还能去哪）",
    "realtime": "实时信息（天气、路况、两地之间交通多久、营业时间）",
    "chitchat": "闲聊或与旅行无关（你好、谢谢、你是谁）",
}

# 新风格标签 → 前端已有的风格提示词（ai-chat.html 的 LABEL_PROMPTS）
LEGACY_LABEL = {
    "food": "Food_Adventure",
    "nature": "Nature_Relaxation",
    "culture": "Deep_Cultural_Trip",
    "citywalk": "City_Leisure_Tour",
    "photo": "Photo_Experience",
    "rush": "Rush_Trip",
}
LEGACY_DEFAULT = "Customized_AI_Tour"


def format_input(destination: str = "", preferences: Iterable[str] = (), text: str = "") -> str:
    """统一输入格式：目的地 / 偏好标签 / 用户原话，缺失的部分省略。"""
    parts = []
    if destination.strip():
        parts.append(f"目的地：{destination.strip()}")
    prefs = [p.strip() for p in preferences if p and p.strip()]
    if prefs:
        parts.append(f"偏好：{'、'.join(prefs)}")
    if text.strip():
        parts.append(f"需求：{text.strip()}")
    return "；".join(parts)
