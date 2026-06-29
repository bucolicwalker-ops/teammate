"""teammate · L0 smoke test —— 让你的第一个 teammate(MyAgent) 说话。

走 DashScope 的 Anthropic 兼容端点 + 智谱 GLM（你的 key 走的就是这条路）。
跑法：.venv/bin/python tests/smoke_hello.py（从 teammate/ 根目录跑）。
"""
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)

resp = client.messages.create(
    model=os.getenv("MODEL", "glm-5.1"),
    max_tokens=512,
    system="你是 teammate 项目里的第一个 AI 队友 MyAgent，热情、简洁、可靠。",
    messages=[
        {"role": "user", "content": "你好，介绍一下你自己，并说说你能帮我做什么。"},
    ],
)
print("".join(b.text for b in resp.content if getattr(b, "type", None) == "text"))
