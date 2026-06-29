"""teammate · L0 W2 —— MyAgent 多工具版（Function Calling）。
W1 做了结构化输出；W2 加工具调用循环，让 MyAgent 能"动手"。
跑法：.venv/bin/python -m src.agent   （从 teammate/ 根目录跑）
"""
import os
import ast
import operator
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
MODEL = os.getenv("MODEL", "glm-5.2")

SYSTEM = (
    "你是 teammate 的 AI 队友 MyAgent。"
    "你可以调用工具来获取信息或执行计算。"
    "需要时先调工具，拿到结果后再给出回复。"
)


# ============================================================
# ① 工具实现 + 注册表
# ============================================================

def get_weather(city: str) -> str:
    """获取城市天气（mock，W2 先不接真 API）"""
    data = {"北京": "晴 25°C", "上海": "多云 28°C", "广州": "雷阵雨 30°C"}
    return data.get(city, f"暂无 {city} 的天气数据")


def search(query: str) -> str:
    """搜索技术概念（mock）"""
    kb = {
        "rag": "RAG = Retrieval-Augmented Generation，检索增强生成。用检索到的真实资料辅助 LLM 回答，减少幻觉。",
        "langgraph": "LangGraph 是 LangChain 出的图式 Agent 编排框架，支持循环、条件路由、状态持久化。",
        "function calling": "Function Calling 让 LLM 输出结构化的工具调用意图，由工程侧执行后回填结果。",
    }
    q = query.lower()
    for k, v in kb.items():
        if k in q:
            return v
    return f"未找到关于「{query}」的信息"


def calc(expression: str) -> str:
    """安全计算数学表达式（只允许数字和加减乘除，防止 eval 注入）"""
    _OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("只支持数字")
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"不支持的语法: {type(node).__name__}")

    tree = ast.parse(expression, mode="eval")
    return str(_eval(tree.body))


# 工具注册表：name → {fn, description, schema}
# 这就是给 LLM 看的"工具菜单"——它根据 description 判断要不要调
TOOL_REGISTRY = {
    "get_weather": {
        "fn": get_weather,
        "description": "获取指定城市的天气信息",
        "schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如 北京、上海"},
            },
            "required": ["city"],
        },
    },
    "search": {
        "fn": search,
        "description": "搜索技术概念或知识",
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        },
    },
    "calc": {
        "fn": calc,
        "description": "计算数学表达式，支持加减乘除和括号",
        "schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式，如 (15+27)*3"},
            },
            "required": ["expression"],
        },
    },
}

# 转成 Anthropic SDK 的 tools 参数格式
TOOLS = [
    {"name": name, "description": cfg["description"], "input_schema": cfg["schema"]}
    for name, cfg in TOOL_REGISTRY.items()
]


def execute_tool(name: str, args: dict) -> str:
    """执行工具，返回结果。出错时返回错误信息（不崩，把错误回灌给模型让它应对）。"""
    if name not in TOOL_REGISTRY:
        return f"错误：未知工具 '{name}'，可用工具：{list(TOOL_REGISTRY.keys())}"
    try:
        result = TOOL_REGISTRY[name]["fn"](**args)
        return str(result)
    except Exception as e:
        return f"工具执行出错：{type(e).__name__}: {e}"


# ============================================================
# ② Agent 调用循环（Function Calling 的核心）
# ============================================================

def ask(user_msg: str, max_steps: int = 5) -> str:
    """多工具 Agent 主循环。

    每轮：
      1. 把对话发给 LLM（带工具列表）
      2. LLM 要么调工具（tool_use），要么直接回答（text）
      3. 如果调工具 → 执行 → 结果回填 → 回到 1
      4. 如果直接回答 → 返回
      5. max_steps 兜底：超过 N 轮还没完就中止（防死循环）
    """
    messages = [{"role": "user", "content": user_msg}]

    for step in range(max_steps):
        print(f"\n  ── 第 {step + 1} 轮 ──")
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        print(f"  stop_reason: {resp.stop_reason}")

        # 检查 LLM 输出里有没有 tool_use
        tool_calls = [b for b in resp.content if b.type == "tool_use"]
        texts = [b for b in resp.content if b.type == "text"]

        if not tool_calls:
            # 没有工具调用 = 模型给出了最终回答
            final = "".join(b.text for b in texts)
            print(f"  ✅ 完成（共 {step + 1} 轮）")
            return final

        # 有工具调用 → 执行所有 tool_use → 收集结果
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for tc in tool_calls:
            print(f"  调工具: {tc.name}({tc.input})")
            result = execute_tool(tc.name, tc.input)
            print(f"  结果: {result}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result,
            })
        # 把工具结果作为 user 消息回填
        messages.append({"role": "user", "content": tool_results})

    return f"⚠️ 超过 {max_steps} 轮仍未完成，已中止（可能死循环）。"


# ============================================================
# ③ 入口
# ============================================================

if __name__ == "__main__":
    test_cases = [
        "北京天气怎么样？",
        "帮我算一下 (15 + 27) * 3 等于多少",
        "北京和上海哪个温度更高？高多少度？",
        "什么是 RAG？",
    ]
    for msg in test_cases:
        print(f"\n{'='*60}")
        print(f"用户: {msg}")
        print(f"{'='*60}")
        reply = ask(msg)
        print(f"\nMyAgent: {reply}")
