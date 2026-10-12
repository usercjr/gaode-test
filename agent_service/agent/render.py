"""把结构化行程渲染成聊天展示用的文本（沿用前端地图页能解析的“第N天 / - [城市] 地点（备注）”格式）。"""
from agent.schemas import Itinerary

_MODE_CN = {"walking": "步行", "driving": "驾车", "transit": "公交/地铁"}


def render_text(it: Itinerary) -> str:
    lines = []
    if it.summary:
        lines += [it.summary, ""]
    for day in it.days:
        lines.append(f"第{day.day}天 {day.theme}".rstrip())
        for stop in day.stops:
            notes = []
            if stop.arrive:
                notes.append(f"{stop.arrive} 到")
            notes.append(f"停留约{stop.stay_min}分钟")
            if stop.cost_per_person:
                notes.append(f"人均约{stop.cost_per_person:g}元")
            notes += stop.tips
            leg = stop.transport_to_next
            if leg and leg.duration_min is not None:
                notes.append(f"→下一站{_MODE_CN.get(leg.mode, leg.mode)}约{leg.duration_min}分钟")
            lines.append(f"- [{it.city}] {stop.name}（{'；'.join(notes)}）")
        lines.append("")
    if it.warnings:
        lines.append("⚠️ 注意：" + "；".join(it.warnings))
    return "\n".join(lines).strip()
