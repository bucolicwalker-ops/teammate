"""teammate · L0 —— MyAgent 结构化输出版。
让模型不再吐自由文本，而是吐「程序能直接用」的结构化 JSON，并做校验 + 失败重试。
跑法：.venv/bin/python agent.py
"""
import os
import json
from anthropic import Anthropic
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
MODEL = os.getenv("MODEL", "glm-5.2")


# ① 契约：我们要的结构长这样（Pydantic 会替我们校验类型）
class AgentReply(BaseModel):
    intent: str           # 用户意图，如 问候 / 提问 / 请求 / 闲聊
    reply: str            # 给用户的回复
    need_followup: bool   # 是否需要追问澄清


SYSTEM = (
    "你是 teammate 的 AI 队友 MyAgent。"
    "只返回一个 JSON 对象，字段为："
    "intent(字符串)、reply(字符串)、need_followup(布尔)。"
    "不要输出 JSON 以外的任何字符，也不要用 ``` 包裹。"
)


def _extract_json(text: str) -> str:
    """坑①：模型常把 JSON 包在 ```json ... ``` 里、或前后加废话。
    这里把代码围栏去掉、再抠出最外层花括号之间的内容。"""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end != -1 else text


def ask(user_msg: str, max_retries: int = 2) -> AgentReply:
    last_err = ""
    for attempt in range(max_retries + 1):
        # 重试时把上次的错误回灌给模型，让它改正（这招很有用）
        prompt = user_msg if attempt == 0 else (
            f"{user_msg}\n\n（你上次的输出无法解析，错误：{last_err}。"
            f"请严格只返回合法 JSON。）"
        )
        resp = client.messages.create(
            model=MODEL, max_tokens=512, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        try:
            data = json.loads(_extract_json(raw))     # 解析
            return AgentReply.model_validate(data)     # ③ Pydantic 校验类型
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = str(e).splitlines()[0][:120]
            print(f"  ⚠️ 第 {attempt + 1} 次解析失败：{last_err}")
    raise RuntimeError(f"重试 {max_retries} 次仍失败，最后错误：{last_err}")


if __name__ == "__main__":
    for msg in ["你好呀！", "帮我把『今天天气很好』翻译成英文", "随便聊聊吧"]:
        r = ask(msg)
        print(f"\n用户: {msg}")
        print(f"  intent       = {r.intent!r}")
        print(f"  need_followup = {r.need_followup}")
        print(f"  reply        = {r.reply}")
