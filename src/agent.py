"""teammate · L2 W4 —— MyAgent 带 RAG 知识库。
W1 结构化输出；W2 工具循环；W3 短期+长期记忆；W4 RAG 知识库检索。
跑法：.venv/bin/python -m src.agent   （从 teammate/ 根目录跑）
"""
import os
import ast
import operator
from anthropic import Anthropic
from dotenv import load_dotenv
from src.memory import VectorMemory
from src.rag import KnowledgeBase

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
# ① 工具实现 + 注册表（W2 不变）
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
# ② MyAgent class（W3 新增：短期记忆）
# ============================================================

class MyAgent:
    """带短期 + 长期记忆的 Agent。

    短期：self.history（内存 list，session 内跨轮存活）。
    长期：self.memory（VectorMemory，跨 session 持久化 + 语义召回）。

    W2 的 ask() 是 stateless function；W3 重构为 class。
    """

    def __init__(self, max_history: int = 20, use_long_term: bool = True,
                 memory_path: str = "data/memory.json",
                 embed_endpoint: str | None = None,
                 use_knowledge: bool = False,
                 knowledge_path: str = "data/knowledge.json"):
        self.history: list[dict] = []
        self.max_history = max_history
        self.memory = VectorMemory(memory_path, embed_endpoint) if use_long_term else None
        self.knowledge = KnowledgeBase(knowledge_path, embed_endpoint) if use_knowledge else None

    def load_knowledge(self, file_path: str, chunk_size: int = 500, overlap: int = 50):
        """加载文档到知识库（RAG 语料）。"""
        if self.knowledge:
            return self.knowledge.load_document(file_path, chunk_size, overlap)
        return 0

    def ask(self, user_msg: str, max_steps: int = 5) -> str:
        """多工具 Agent 主循环（带短期 + 长期记忆）。

        短期：self.history 跨调用存活。
        长期：ask 前 retrieve 相关记忆注入 system；ask 后存对话到向量库。
        """
        recalled = []
        if self.memory:
            recalled = self.memory.retrieve(user_msg, top_k=3)
        kb_chunks = []
        if self.knowledge:
            kb_chunks = self.knowledge.retrieve(user_msg, top_k=3)
        system = SYSTEM
        if recalled:
            system += self.memory.format_context(recalled)
        if kb_chunks:
            system += self.knowledge.format_context(kb_chunks)

        self.history.append({"role": "user", "content": user_msg})
        self._truncate()

        for step in range(max_steps):
            print(f"\n  ── 第 {step + 1} 轮 ──")
            try:
                resp = client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=system,
                    tools=TOOLS,
                    messages=self.history,
                )
            except Exception as e:
                err = f"⚠️ LLM 调用失败（{type(e).__name__}），已中止: {e}"
                self.history.append({"role": "assistant", "content": err})
                if self.memory:
                    self.memory.add(f"用户: {user_msg}\nMyAgent: {err}")
                return err
            print(f"  stop_reason: {resp.stop_reason}")

            tool_calls = [b for b in resp.content if b.type == "tool_use"]
            texts = [b for b in resp.content if b.type == "text"]

            if not tool_calls:
                final = "".join(b.text for b in texts)
                self.history.append({"role": "assistant", "content": final})
                if self.memory:
                    self.memory.add(f"用户: {user_msg}\nMyAgent: {final}")
                print(f"  ✅ 完成（共 {step + 1} 轮）")
                return final

            self.history.append({"role": "assistant", "content": resp.content})
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
            self.history.append({"role": "user", "content": tool_results})

        fallback = f"⚠️ 超过 {max_steps} 轮仍未完成，已中止（可能死循环）。"
        self.history.append({"role": "assistant", "content": fallback})
        if self.memory:
            self.memory.add(f"用户: {user_msg}\nMyAgent: {fallback}")
        return fallback

    def _truncate(self):
        """截断策略：保留最近 max_history 条，但确保不破坏 user/assistant 配对。

        盲切片会从 turn 中间断开——首条可能是 assistant 或 tool_result，
        API 要求 messages 必须以 proper user 消息开头且交替配对。
        修复：切片后从头部丢弃不完整的消息，直到首条是 user + string content。
        注意：清理后实际条数可能少于 max_history（连带丢了完整 turn 的碎片）。

        进阶：按轮截断（turn-aware）比按消息截断更干净——
        不会连带丢配对；摘要压缩（用 LLM summarize 老对话）；
        混合（老对话 summarize + 近 N 轮保留原文）。
        """
        if len(self.history) <= self.max_history:
            return
        self.history = self.history[-self.max_history:]
        while self.history:
            first = self.history[0]
            if first["role"] == "user" and isinstance(first["content"], str):
                break
            self.history.pop(0)
        print(f"  📝 历史截断：保留最近 {len(self.history)} 条消息")


# ============================================================
# ③ 入口
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "session2":
        # D-6 验证：长期记忆——新 session（新 MyAgent 实例），能不能 recall 上个 session 的记忆
        agent = MyAgent(max_history=20)
        print("=" * 60)
        print("D-6 长期记忆验证：跨 session recall")
        print("=" * 60)
        for msg in ["我叫什么名字？", "我住在哪里？"]:
            print(f"\n{'─' * 60}")
            print(f"用户: {msg}")
            print(f"{'─' * 60}")
            reply = agent.ask(msg)
            print(f"\nMyAgent: {reply}")

    elif len(sys.argv) > 1 and sys.argv[1] == "rag":
        # W4 验证：RAG 知识库——加载文档→检索→注入→生成
        agent = MyAgent(max_history=20, use_knowledge=True)
        agent.load_knowledge("../bagu/qbank.md")
        print("=" * 60)
        print("W4 RAG 验证：知识库问答")
        print("=" * 60)
        for msg in [
            "agent记忆系统怎么实现？",
            "上下文压缩怎么做？",
        ]:
            print(f"\n{'─' * 60}")
            print(f"用户: {msg}")
            print(f"{'─' * 60}")
            reply = agent.ask(msg)
            print(f"\nMyAgent: {reply}")

    else:
        # D-5 验证：短期记忆——多轮对话能接住上下文
        agent = MyAgent(max_history=20)
        conversations = [
            "北京天气怎么样？",
            "那上海呢？",
            "两个城市哪个更热？高多少？",
        ]
        print("=" * 60)
        print("D-5 短期记忆验证：多轮对话")
        print("=" * 60)
        for msg in conversations:
            print(f"\n{'─' * 60}")
            print(f"用户: {msg}")
            print(f"{'─' * 60}")
            reply = agent.ask(msg)
            print(f"\nMyAgent: {reply}")

        # D-6 种子：存一条跨 session 应该记住的事实
        print(f"\n{'=' * 60}")
        print("D-6 种子：存一条记忆供下个 session recall")
        print(f"{'=' * 60}")
        reply = agent.ask("记住：我叫小明，我住在北京。")
        print(f"\nMyAgent: {reply}")
        print(f"\n→ 现在运行 `.venv/bin/python -m src.agent session2` 验证跨 session recall")
