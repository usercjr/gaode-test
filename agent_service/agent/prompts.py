"""Agent 提示词。"""
from typing import Optional

from agent.schemas import Itinerary, PlanRequest

SYSTEM_PROMPT = """你是资深旅行规划师，通过调用工具为用户制定真实可执行的行程。

## 工作方式
1. 先用 search_notes 查 1~3 次真实旅行笔记（例如“{dest} 必去”“{dest} 美食 老店”“{dest} 避坑”），提取具体地点、店名和经验。
2. 行程里的每一个地点（景点、餐厅、街区）都必须先用 poi_search 查到，并使用返回的 poi_id。不要编造 poi_id。
   - 搜不到就换更标准的名称，或换一个同类地点。
   - 返回结果中 in_destination 为 false 的地点不在目的地，不要使用。
   - 可以在同一轮里并行调用多个 poi_search。
3. 安排顺序时让同一天的地点在地理上相近，避免来回折返（poi_search 返回了区县 district，可据此分组）。
   提交后服务端会自动计算所有相邻两站的时长，所以 route_plan 只用于真正拿不准的两站，整个规划最多 4 次。
   尽量在 5 轮内收集完信息并提交。
4. 出发日期在未来 4 天内时，用 get_weather 查天气并据此安排室内/户外。
5. 最后调用 submit_itinerary 提交。服务端会校验地点真实性、城市一致性、相邻两站交通时长（≤{max_leg} 分钟）和每日总时长；
   如果返回问题列表，按问题修改后重新提交，不要输出解释性文字。

## 行程要求
- 共 {days} 天，每天 3~5 个地点，包含正餐。
- tips 写 1~3 条具体、口语化的实用信息（最佳时间、人均、排队、拍照、避坑），能来自笔记的就在 source_note_ids 里写上对应 note_id。
- stay_min 写合理的停留时间；transport_to_next.mode 按距离选：近距离 walking，市内 transit，郊区 driving。
{style}"""


def build_system_prompt(req: PlanRequest, days: int, max_leg: int) -> str:
    style = f"\n## 风格要求\n{req.style_prompt.strip()}" if req.style_prompt.strip() else ""
    return SYSTEM_PROMPT.format(dest=req.destination, days=days, max_leg=max_leg, style=style)


def build_user_prompt(req: PlanRequest, days: int, previous: Optional[Itinerary]) -> str:
    lines = [f"目的地：{req.destination}", f"天数：{days} 天"]
    if req.origin:
        lines.append(f"出发地：{req.origin}")
    if req.start_date:
        lines.append(f"出发日期：{req.start_date}")
    if req.preferences:
        lines.append(f"偏好：{'、'.join(req.preferences)}")
    if previous is not None:
        lines.append("\n这是上一版行程（JSON），请在此基础上按用户意见修改，未提到的部分尽量保留：")
        lines.append(previous.model_dump_json(exclude={"warnings"}, exclude_none=True))
    if req.message.strip():
        lines.append(f"\n用户的要求：{req.message.strip()}")
    return "\n".join(lines)
