# -*- coding: utf-8 -*-
# 模型调用（来自 prompt7.txt，整理为可直接运行的 CLI 脚本，并改为使用环境变量读取密钥）

import os
import json
import time
import paddle
import numpy as np

from openai import OpenAI  # 星河社区 OpenAI 兼容 SDK
from paddlenlp.transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========== 1) 文心大模型（星河API，OpenAI SDK兼容）==========
API_KEY = (os.getenv("BAIDU_LLM_API_KEY") or os.getenv("BAIDU_API_KEY") or "").strip()
if not API_KEY:
    raise RuntimeError("缺少 BAIDU_LLM_API_KEY（或 BAIDU_API_KEY）环境变量")

BASE_URL = os.getenv("BAIDU_BASE_URL", "https://aistudio.baidu.com/llm/lmapi/v3").rstrip("/")
MODEL_NAME = os.getenv("BAIDU_MODEL", "ernie-3.5-8k").strip()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ========== 2) 加载你的文本分类模型 ==========
BEST_MODEL_DIR = os.getenv("BEST_MODEL_DIR", "best_model")  # 确保目录存在：config.json / model_state.pdparams / vocab.txt ...

tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
clf_model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
clf_model.eval()

# 读取标签映射优先级：best_model/label_map.json -> best_model/labels.txt -> 默认内置
def load_label_list():
    lm_json = os.path.join(BEST_MODEL_DIR, "label_map.json")
    lm_txt  = os.path.join(BEST_MODEL_DIR, "labels.txt")
    if os.path.exists(lm_json):
        with open(lm_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if "id2label" in obj:
                # {"0":"Rush_Trip", ...}
                id2label = {int(k): v for k, v in obj["id2label"].items()}
                return [id2label[i] for i in sorted(id2label.keys())]
            elif "labels" in obj and isinstance(obj["labels"], list):
                return obj["labels"]
    if os.path.exists(lm_txt):
        with open(lm_txt, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    # 默认（按你示例中的 8 类）
    return [
        "Rush_Trip",
        "Deep_Cultural_Trip",
        "Food_Adventure",
        "Nature_Relaxation",
        "City_Leisure_Tour",
        "Customized_AI_Tour",
        "Vlog_Creator_Trip",
        "Photo_Experience",
    ]

LABELS = load_label_list()
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}

def classify_text(text: str) -> str:
    enc = tokenizer(
        text,
        max_seq_len=256,
        pad_to_max_seq_len=True,
        return_tensors="pd"
    )
    with paddle.no_grad():
        logits = clf_model(enc["input_ids"], enc["token_type_ids"])
    pred_id = int(paddle.argmax(logits, axis=1).item())
    return ID2LABEL.get(pred_id, LABELS[0])

# ========== 3) 针对不同偏好标签的系统 Prompt ==========
LABEL_PROMPTS = {
    "Rush_Trip": (
        "🎯【特种兵式高效快游风格】\n"
        "用户希望在最短时间打卡最多景点，追求效率、压缩休息时间。\n"
        "规划规则：\n"
        "- 行程高密度，一天安排 4~6 个景点\n"
        "- 强调“时间-地点-理由-通勤方式/时长”结构\n"
        "- 合理排序景点避免来回折返\n"
        "- 提供替代方案（下雨/排队太久）\n"
        "- 可以适当牺牲舒适度以提高效率\n"
    ),
    "Deep_Cultural_Trip": (
        "📚【深度文化体验风格】\n"
        "用户关注历史、人文、非遗、博物馆、古镇，想要理解城市文化。\n"
        "规划规则：\n"
        "- 每个景点附加文化背景或故事\n"
        "- 提供讲解方式（请讲解员/语音导览）\n"
        "- 合理安排参观顺序，避免信息过载\n"
        "- 标注预约方式、闭馆日、最佳参观时间\n"
    ),
    "Food_Adventure": (
        "🍜【美食探店风格】\n"
        "用户以“吃”为核心，餐厅是行程重点。\n"
        "规划规则：\n"
        "- 每天 2~4 家餐厅（正餐 + 小吃）\n"
        "- 每家餐厅列出：招牌菜、排队时段、是否需预约、价格区间\n"
        "- 景点作为餐厅之间的过渡\n"
        "- 标注深夜食堂/早餐/夜市等特色\n"
    ),
    "Nature_Relaxation": (
        "🌿【自然放松疗愈风格】\n"
        "用户偏好自然风景、公园、山/湖/森林，追求松弛感与治愈。\n"
        "规划规则：\n"
        "- 一天 1~3 个地点，节奏慢\n"
        "- 安排轻徒步/观景台/草地野餐/休息时间\n"
        "- 避免长时间舟车劳顿\n"
        "- 推荐最佳观景时间（晨景/夕阳/夜景）\n"
    ),
    "City_Leisure_Tour": (
        "🏙【城市休闲 Citywalk 风格】\n"
        "用户喜欢逛街、街道氛围、咖啡馆、小店、夜景、拍照。\n"
        "规划规则：\n"
        "- 串联商圈、街区、地标\n"
        "- 强调“步行可达性”和“氛围感”\n"
        "- 推荐咖啡店、小众商店、艺术空间\n"
        "- 可加入夜生活/酒吧/灯光秀\n"
    ),
    "Customized_AI_Tour": (
        "🤖【智能综合行程规划风格】\n"
        "用户未明确偏好或想要智能生成最优解。\n"
        "规划规则：\n"
        "- 综合考虑景点分布、住宿位置、交通方式、开放时间\n"
        "- 每段安排通勤建议（地铁/公交/步行/打车）和预估时长\n"
        "- 给出费用估算和替代选项\n"
        "- 行程结构化输出，逻辑清晰\n"
    ),
    "Vlog_Creator_Trip": (
        "🎥【Vlog 创作者风格】\n"
        "用户要拍旅行视频，内容需要有画面感和故事性，但不要出现专业拍摄术语。\n"
        "规划规则：\n"
        "- 语言口语化、轻松，像旅游博主在讲 vlog 故事\n"
        "- 提示“这里很好拍”“适合拍特写”“出片角度”\n"
        "- 安排有节奏感的场景切换（早-热闹/午-特色/晚-夜景）\n"
        "- 给出少量拍摄建议（光线好/背景好/人少时段）\n"
    ),
    "Photo_Experience": (
        "📸【拍照打卡/出片风格】\n"
        "用户追求拍照好看，想获得“照片成果感”。\n"
        "规划规则：\n"
        "- 每个景点列出机位坐标/最佳角度/构图建议\n"
        "- 标注时间光线（晨光/夕阳/蓝调/夜景）\n"
        "- 提供避人技巧和拍照姿势建议\n"
        "- 安排化妆/换装/道具租赁等辅助信息\n"
    ),
}

BASE_OUTPUT_FORMAT = (
    "请使用以下结构输出行程（自然语言，不要用 JSON）：\n"
    "\n"
    "🌞 第X天 上午：\n"
    "- 景点/活动1（理由 + 体验亮点 + 停留时间 + 通勤方式/时长）\n"
    "- 景点/活动2（可选）\n"
    "\n"
    "🌤 第X天 下午：\n"
    "- 景点/活动1 …\n"
    "- 可选美食/咖啡/休息地点\n"
    "\n"
    "🌙 第X天 晚上：\n"
    "- 夜景/夜市/酒吧/放松活动\n"
    "- 如果需要，加入最佳拍照时间或避人建议\n"
    "\n"
    "✅ 如果景点需要预约、闭馆、排队、限制入场，请提前提醒\n"
    "✅ 如果下雨/高温/人太多，请提供替代方案\n"
    "✅ 行程结束后追加：『注意事项 / 小贴士 / 费用估算 / 交通建议』\n"
)

def build_system_prompt(label: str) -> str:
    head = LABEL_PROMPTS.get(label, LABEL_PROMPTS["Customized_AI_Tour"])
    return head + " " + BASE_OUTPUT_FORMAT

# ========== 4) 多轮对话（命令行）==========
def chat_loop():
    print(">>> AI 旅游助手（多轮对话）已启动。输入 exit/quit 结束。\n")
    messages = []  # OpenAI SDK 聊天历史：[{role, content}, ...]

    # 初始化为通用旅行助理（首次用户说话后，会追加偏好标签的系统提示）
    system_base = {
        "role": "system",
        "content": (
            "你是一名专业级 AI 旅行规划师，具备以下能力："
            "实际可执行的路线规划能力（考虑距离、时间、开闭馆、交通方式、排队时长），"
            "精通不同类型景点、美食、拍照点、文化体验、自然景观，"
            "能根据用户偏好、预算、体力、时间进行个性化定制，"
            "能自动优化动线，减少无效往返，"
            "输出形式参考优质旅游博主/小红书/专业攻略平台风格，语言自然、生动、有人味。"
        ),
    }
    messages.append(system_base)

    while True:
        user_input = input("你：").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("助手：再见，祝旅途愉快！")
            break

        # 1) 分类（基于你训练好的模型）
        label = classify_text(user_input)

        # 2) 根据标签构建/更新系统提示（放在用户消息前，保持最新偏好）
        system_label_msg = {"role": "assistant", "content": f"(偏好标签：{label})我会按照这个偏好继续规划。"}
        messages.append(system_label_msg)

        # 3) 追加用户消息
        messages.append({"role": "user", "content": user_input})

        # 4) 调用文心（OpenAI SDK 兼容）
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                extra_headers={
                    "x-bce-date": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                }
            )
            assistant_text = resp.choices[0].message.content
        except Exception as e:
            assistant_text = f"[LLM ERROR] {e}"

        # 5) 存入历史并输出
        messages.append({"role": "assistant", "content": assistant_text})
        print("助手：", assistant_text, "\n")

        # 可选：裁剪历史，避免过长（根据需求设定）
        if len(messages) > 30:
            # 始终保留最初的 system_base，其它对话滑窗
            messages = [system_base] + messages[-20:]


if __name__ == "__main__":
    chat_loop()


