"""结构化行程：LLM 通过 submit_itinerary 工具提交，Pydantic 校验结构，validator 校验业务规则。"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

TransportMode = Literal["walking", "driving", "transit"]


class Location(BaseModel):
    lng: float
    lat: float


class Leg(BaseModel):
    mode: TransportMode = "transit"
    duration_min: Optional[int] = None  # 由校验器用高德实算值覆盖
    distance_m: Optional[int] = None
    verified: bool = False  # True 表示来自高德路线规划，而非模型估计


class Stop(BaseModel):
    name: str = Field(..., min_length=1)
    poi_id: str = Field(..., min_length=1, description="必须来自 poi_search 返回结果")
    category: Literal["attraction", "food", "shopping", "hotel", "other"] = "attraction"
    arrive: str = Field("", description="建议到达时间，如 09:30")
    stay_min: int = Field(60, ge=10, le=600)
    cost_per_person: Optional[float] = Field(None, ge=0)
    tips: List[str] = Field(default_factory=list, max_length=5)
    source_note_ids: List[str] = Field(default_factory=list, description="参考的笔记 note_id")
    transport_to_next: Optional[Leg] = None
    # 以下字段由服务端根据 poi_id 回填，模型无需填写
    location: Optional[Location] = None
    address: str = ""
    district: str = ""


class Day(BaseModel):
    day: int = Field(..., ge=1)
    theme: str = ""
    stops: List[Stop] = Field(..., min_length=1, max_length=8)


class Itinerary(BaseModel):
    title: str
    city: str
    days: List[Day] = Field(..., min_length=1, max_length=10)
    summary: str = ""
    warnings: List[str] = Field(default_factory=list)


class PlanRequest(BaseModel):
    destination: str = Field(..., min_length=1, max_length=50)
    origin: str = Field("", max_length=50)
    days: Optional[int] = Field(None, ge=1, le=10)
    start_date: str = Field("", max_length=20)
    preferences: List[str] = Field(default_factory=list, max_length=20)
    style_prompt: str = Field("", max_length=2000, description="前端风格标签对应的写作要求")
    message: str = Field("", max_length=2000, description="用户的补充需求或多轮修改意见")
    previous_itinerary: Optional[Itinerary] = None
