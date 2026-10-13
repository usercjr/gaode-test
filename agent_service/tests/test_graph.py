from graph.align import clean, names_match, pick


def _poi(name, adcode="510104", type_="风景名胜;风景名胜;旅游景点"):
    return {"poi_id": f"B{abs(hash((name, adcode))) % 10**8}", "name": name, "adcode": adcode, "type": type_}


def test_names_match_rules():
    assert clean("钟水饺（总府路店）") == "钟水饺"
    assert names_match("钟水饺", "钟水饺(总府路店)")
    assert names_match("成都大熊猫繁育研究基地东门", "成都大熊猫繁育研究基地")
    assert not names_match("沙茶面", "厦门大学")


def test_pick_prefers_venue_in_city():
    cands = [_poi("宽窄巷子(地铁站)", type_="交通设施服务;地铁站;地铁站"),
             _poi("宽窄巷子", adcode="500103"),   # 重庆的同名点
             _poi("宽窄巷子")]
    assert pick("宽窄巷子", cands, "5101") is cands[2]
    assert pick("宽窄巷子", [_poi("人民公园")], "5101") is None


def test_tags_accept_and_distance():
    from graph.build import accept, haversine_m, tags_of

    assert tags_of("餐饮服务;中餐厅;火锅店") == ["火锅", "美食"]
    assert tags_of("风景名胜;风景名胜;世界遗产|科教文化服务;博物馆;博物馆") == ["博物馆", "文化", "景点"]
    school = {"query": "黑芝麻胡同", "poi": {"poi_id": "B1", "name": "黑芝麻胡同小学", "type": "科教文化服务;学校;小学"}}
    assert accept(school) == (False, False)  # 名称不完全一致 + 学校类 → 误配
    chain = {"query": "紫光园", "poi": {"poi_id": "B2", "name": "紫光园(隆福寺店)", "type": "餐饮服务;中餐厅;中餐厅"}}
    assert accept(chain) == (True, True)    # 连锁品牌：分店是猜的
    exact = {"query": "鼓楼", "poi": {"poi_id": "B3", "name": "鼓楼", "type": "风景名胜;风景名胜;风景名胜"}}
    assert accept(exact) == (True, False)
    # 抽检后新增的两条规则
    fragment = {"query": "都匀东黄记地摊火锅", "poi": {"poi_id": "B4", "name": "地摊火锅", "type": "餐饮服务;中餐厅"}}
    assert accept(fragment)[0] is False  # 高德名是查询的短片段
    borrowed = {"query": "洱海", "poi": {"poi_id": "B5", "name": "大理白族洱海渔庄店", "type": "餐饮服务;中餐厅"}}
    assert accept(borrowed)[0] is False  # 借名的商户
    scenic = {"query": "兵马俑博物馆", "poi": {"poi_id": "B6", "name": "秦始皇兵马俑博物馆", "type": "科教文化服务;博物馆"}}
    assert accept(scenic)[0] is True     # 景点全称比简称长是正常的
    assert accept({"query": "x", "poi": {"poi_id": [], "name": "x", "type": ""}})[0] is False  # 无 id
    d = haversine_m({"lng": 104.0533, "lat": 30.6696}, {"lng": 104.0627, "lat": 30.6612})  # 宽窄巷子 → 人民公园
    assert 1100 < d < 1400


class _FakeGraph:
    enabled = True

    def __init__(self, rows=None, error=None):
        self.rows, self.error, self.calls = rows or [], error, []

    async def run(self, query, params=None, write=False):
        self.calls.append(params)
        if self.error:
            raise self.error
        return self.rows


def _exec(graph):
    import asyncio

    from agent.tools import PlanContext, ToolExecutor

    ctx = PlanContext(destination="成都市")
    ex = ToolExecutor(amap=None, retriever=None, graph=graph)
    return lambda args: asyncio.run(ex.run(ctx, "graph_query", __import__("json").dumps(args, ensure_ascii=False)))


def test_graph_query_tool_and_degradation():
    from core.errors import AppError, ErrorCode

    run = _exec(_FakeGraph(rows=[{"name": "钟水饺", "mentions": 3}]))
    out = run({"template": "nearby", "place": "春熙路", "tag": "美食"})
    assert out["ok"] and out["rows"][0]["mentions"] == 3
    assert run({"template": "drop_all"})["ok"] is False  # 只能用固定模板

    down = _exec(_FakeGraph(error=AppError(ErrorCode.GRAPH_UNAVAILABLE)))
    res = down({"template": "nearby", "place": "春熙路"})
    assert res["ok"] is False and "nearby_search" in res["hint"]  # 图谱挂了提示改用高德
    assert "nearby_search" in _exec(None)({"template": "by_tag", "tag": "美食"})["hint"]


def test_graph_query_normalizes_city():
    fake = _FakeGraph(rows=[{"name": "x"}])
    _exec(fake)({"template": "by_tag", "tag": "美食"})
    assert fake.calls[0]["city"] == "成都"  # “成都市” → “成都”，与图谱里的城市名一致
