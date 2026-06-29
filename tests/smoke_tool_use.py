"""测试 GLM Anthropic 兼容端点是否支持 tool use (Function Calling)。
跑法：.venv/bin/python tests/smoke_tool_use.py（从 teammate/ 根目录跑）
"""
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
MODEL = os.getenv("MODEL", "glm-5.2")

# 定义一个简单工具
tools = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名，如 北京、上海",
                }
            },
            "required": ["city"],
        },
    }
]


def get_weather(city: str) -> str:
    """mock：返回假数据（W2 先 focus 机制，不接真 API）"""
    data = {"北京": "晴 25°C", "上海": "多云 28°C"}
    return data.get(city, "未知城市")


print(f"模型: {MODEL}")
print(f"端点: {os.getenv('ANTHROPIC_BASE_URL')}")
print("=" * 50)

# 第一步：发请求，看模型要不要调工具
resp = client.messages.create(
    model=MODEL,
    max_tokens=512,
    system="你是天气助手。需要天气信息时请调用 get_weather 工具。",
    tools=tools,
    messages=[{"role": "user", "content": "北京天气怎么样？"}],
)

print(f"\nstop_reason: {resp.stop_reason}")
print(f"content blocks:")
for i, block in enumerate(resp.content):
    print(f"  [{i}] type={block.type}")
    if block.type == "text":
        print(f"      text={block.text!r}")
    elif block.type == "tool_use":
        print(f"      name={block.name}")
        print(f"      input={block.input}")
        print(f"      id={block.id}")

# 如果模型要调工具，执行并回填
if resp.stop_reason == "tool_use":
    print("\n✅ 模型请求了工具调用！")
    tool_block = next(b for b in resp.content if b.type == "tool_use")
    print(f"   调用: {tool_block.name}({tool_block.input})")
    result = get_weather(**tool_block.input)
    print(f"   执行结果: {result}")

    # 回填结果，再问模型
    resp2 = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system="你是天气助手。需要天气信息时请调用 get_weather 工具。",
        tools=tools,
        messages=[
            {"role": "user", "content": "北京天气怎么样？"},
            {"role": "assistant", "content": resp.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": result,
                    }
                ],
            },
        ],
    )
    print(f"\n第二轮 stop_reason: {resp2.stop_reason}")
    for block in resp2.content:
        if block.type == "text":
            print(f"✅ 最终回答: {block.text}")
else:
    print("\n⚠️ 模型没有请求工具调用（stop_reason != tool_use）")
    for block in resp.content:
        if block.type == "text":
            print(f"   直接回答: {block.text}")
