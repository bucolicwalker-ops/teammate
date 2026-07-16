"""teammate · L3 W8 —— MyAgent 带可观测 + 服务化 + MCP。
W1 结构化输出；W2 工具循环；W3 记忆；W4 RAG；W5 评估；W6 规划+多Agent；W7 真工具+MCP；W8 可观测+服务化。
跑法：.venv/bin/python -m src.agent [rag|session2|plan|tools|mcp]  （从 teammate/ 根目录跑）
"""
import os
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from anthropic import Anthropic
from dotenv import load_dotenv
from src.memory import VectorMemory
from src.rag import KnowledgeBase
from src.trace import Tracer
from src.tools import TOOL_REGISTRY, TOOLS, execute_tool, TOOL_TIMEOUT

load_dotenv()
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
MODEL = os.getenv("MODEL", "glm-5.2")

DEFAULT_SYSTEM = (
    "你是 teammate 的 AI 队友 MyAgent。"
    "你可以调用工具来获取信息或执行计算。"
    "需要时先调工具，拿到结果后再给出回复。"
)



# 工具层已分离到 src/tools.py

# ============================================================
# ③ MCP 客户端（W7：工具标准化接入）
# ============================================================

class MCPClient:
    """MCP 客户端：通过子进程 + JSON-RPC 调用 MCP server 的工具。

    和直接函数调用（execute_tool）的区别：
    - 直接调用：同进程，fn(**args)
    - MCP 调用：子进程，JSON-RPC over stdio
    两者走同一个 execute_tool（含失败处理），只是传输方式不同。

    MCP 协议（JSON-RPC 2.0 over stdio）：
    1. initialize 握手
    2. tools/list → 返回工具 schema
    3. tools/call → 执行工具，返回结果
    """

    def __init__(self):
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "src.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # P2-3: 不读 stderr → DEVNULL 防 pipe 满死锁
            text=True,
            cwd=cwd,
        )
        # initialize 握手
        self._send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        print("  🔌 MCP server 已连接")

    def call_tool(self, name: str, args: dict) -> str:
        """通过 MCP 协议调用工具。"""
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": name, "arguments": args}}
        resp = self._send(req)
        if "error" in resp:
            return resp["error"]["message"]
        return resp["result"]["content"][0]["text"]

    def _send(self, req: dict) -> dict:
        """发送 JSON-RPC 请求，读响应。notification（无 id）不读响应。

        P2-2: readline 加 timeout——server 崩了/不回响应不会 hang 死 Agent。
        """
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        if "id" not in req:
            return {}
        # readline with timeout（跨线程安全，不用 SIGALRM）
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.proc.stdout.readline)
            try:
                line = future.result(timeout=TOOL_TIMEOUT + 5)
            except FuturesTimeoutError:
                raise TimeoutError("MCP server 响应超时")
        if not line:
            raise TimeoutError("MCP server 无响应（可能已崩溃）")
        return json.loads(line)

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()


# ============================================================
# ④ MyAgent class
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
                 knowledge_path: str = "data/knowledge.json",
                 system: str | None = None,
                 use_mcp: bool = False):
        self.history: list[dict] = []
        self.max_history = max_history
        self.system = system or DEFAULT_SYSTEM
        self.memory = VectorMemory(memory_path, embed_endpoint) if use_long_term else None
        self.knowledge = KnowledgeBase(knowledge_path, embed_endpoint) if use_knowledge else None
        self.mcp_client = MCPClient() if use_mcp else None
        self.tracer = Tracer()
        self._lock = threading.Lock()  # 串行化 ask()——服务化并发安全

    def load_knowledge(self, file_path: str, chunk_size: int = 500, overlap: int = 50):
        """加载文档到知识库（RAG 语料）。"""
        if self.knowledge:
            return self.knowledge.load_document(file_path, chunk_size, overlap)
        return 0

    def ask(self, user_msg: str, max_steps: int = 5) -> str:
        """线程安全入口：串行化 ask——服务化并发时防 history 混 + trace 串。"""
        with self._lock:
            return self._ask_unlocked(user_msg, max_steps)

    def _ask_unlocked(self, user_msg: str, max_steps: int = 5) -> str:
        """多工具 Agent 主循环（带短期 + 长期记忆 + trace）。

        短期：self.history 跨调用存活。
        长期：ask 前 retrieve 相关记忆注入 system；ask 后存对话到向量库。
        """
        recalled = []
        if self.memory:
            recalled = self.memory.retrieve(user_msg, top_k=3)
        kb_chunks = []
        if self.knowledge:
            kb_chunks = self.knowledge.retrieve(user_msg, top_k=3)
        system = self.system
        if recalled:
            system += self.memory.format_context(recalled)
        if kb_chunks:
            system += self.knowledge.format_context(kb_chunks)

        self.history.append({"role": "user", "content": user_msg})
        self._truncate()

        trace_id = self.tracer.start_trace(user_msg)

        for step in range(max_steps):
            print(f"\n  ── 第 {step + 1} 轮 ──")
            t0 = time.time()
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
                self.tracer.finish_trace(trace_id, err)
                return err
            llm_latency = time.time() - t0
            in_tok = getattr(getattr(resp, 'usage', None), 'input_tokens', 0) or 0
            out_tok = getattr(getattr(resp, 'usage', None), 'output_tokens', 0) or 0
            self.tracer.log_llm_call(trace_id, step + 1, in_tok, out_tok, llm_latency)
            print(f"  stop_reason: {resp.stop_reason}")

            tool_calls = [b for b in resp.content if b.type == "tool_use"]
            texts = [b for b in resp.content if b.type == "text"]

            if not tool_calls:
                final = "".join(b.text for b in texts)
                self.history.append({"role": "assistant", "content": final})
                if self.memory:
                    self.memory.add(f"用户: {user_msg}\nMyAgent: {final}")
                self.tracer.finish_trace(trace_id, final)
                print(f"  ✅ 完成（共 {step + 1} 轮）")
                return final

            self.history.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for tc in tool_calls:
                print(f"  调工具: {tc.name}({tc.input})")
                t_tool = time.time()
                if self.mcp_client:
                    result = self.mcp_client.call_tool(tc.name, tc.input)
                else:
                    result = execute_tool(tc.name, tc.input)
                tool_latency = time.time() - t_tool
                self.tracer.log_tool_call(trace_id, tc.name, tc.input, result, tool_latency)
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
        self.tracer.finish_trace(trace_id, fallback)
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

    def plan_and_execute(self, user_msg: str, max_steps: int = 8) -> str:
        """Plan-Execute：先规划全部步骤，再逐步执行，最后汇总。

        和 ReAct（ask）的区别：ReAct 走一步看一步，Plan-Execute 先全局规划再执行。
        适合步骤多、依赖关系复杂的任务（如"帮我做三日游攻略"）。
        每个步骤用 ReAct（ask）执行——Plan-Execute 是 ReAct 的上层包装，不是替换。
        """
        print(f"  📋 Plan 阶段：生成执行计划...")
        plan_resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system="你是规划者。拆解任务为可执行的步骤。返回 JSON。只返回 JSON。",
            messages=[{"role": "user", "content": (
                f"分析以下任务，拆解成具体可执行的步骤（每步一个子任务）。\n"
                f"任务: {user_msg}\n\n"
                f'返回 JSON: {{"steps": ["步骤1", "步骤2", ...]}}'
            )}],
        )
        plan_text = "".join(b.text for b in plan_resp.content if b.type == "text").strip()
        if "```" in plan_text:
            plan_text = plan_text.split("```")[1]
            if plan_text.startswith("json"):
                plan_text = plan_text[4:]
        try:
            steps = json.loads(plan_text)["steps"]
        except json.JSONDecodeError:
            return f"规划失败，直接回答：\n{self.ask(user_msg)}"

        print(f"  📋 计划：{len(steps)} 步")
        for i, s in enumerate(steps):
            print(f"     {i+1}. {s}")

        # Execute：每步用 ReAct 执行
        results = []
        self.history.clear()
        for i, step in enumerate(steps):
            print(f"\n  🔨 Execute 步骤 {i+1}/{len(steps)}: {step}")
            result = self.ask(step)
            results.append(f"步骤{i+1}（{step}）: {result}")

        # Summarize：汇总所有步骤结果
        print(f"\n  📝 Summarize 阶段：汇总结果...")
        summary_resp = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=self.system,
            messages=[{"role": "user", "content": (
                f"基于以下执行结果，回答用户的原始问题。\n"
                f"原始问题: {user_msg}\n\n"
                f"执行结果:\n" + "\n\n".join(results)
            )}],
        )
        final = "".join(b.text for b in summary_resp.content if b.type == "text")
        return final


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

    elif len(sys.argv) > 1 and sys.argv[1] == "mcp":
        # W7 验证：MCP 独立进程——工具通过 MCP 协议调用（不是同进程函数）
        agent = MyAgent(max_history=20, use_mcp=True)
        print("=" * 60)
        print("W7 MCP 验证：工具通过独立进程调用")
        print("=" * 60)
        for msg in ["读一下 requirements.txt 文件", "帮我算 15 * 3"]:
            print(f"\n{'─' * 60}")
            print(f"用户: {msg}")
            print(f"{'─' * 60}")
            reply = agent.ask(msg)
            print(f"\nMyAgent: {reply[:100]}...")
        agent.mcp_client.close()

    elif len(sys.argv) > 1 and sys.argv[1] == "tools":
        # W7 验证：真工具 + 失败处理
        agent = MyAgent(max_history=20)
        print("=" * 60)
        print("W7 真工具 + 失败处理验证")
        print("=" * 60)
        for msg in [
            "读一下 requirements.txt 文件",
            "读一下 nonexistent.txt 文件",
        ]:
            print(f"\n{'─' * 60}")
            print(f"用户: {msg}")
            print(f"{'─' * 60}")
            reply = agent.ask(msg)
            print(f"\nMyAgent: {reply}")

    elif len(sys.argv) > 1 and sys.argv[1] == "plan":
        # W6 验证：Plan-Execute——先规划再执行
        agent = MyAgent(max_history=20, use_knowledge=True)
        agent.load_knowledge("../bagu/qbank.md")
        print("=" * 60)
        print("W6 Plan-Execute 验证")
        print("=" * 60)
        task = "帮我查一下北京和上海的天气，算出温差，再解释哪个城市更热"
        print(f"\n任务: {task}")
        reply = agent.plan_and_execute(task)
        print(f"\n{'=' * 60}")
        print(f"最终结果:")
        print(f"{'=' * 60}")
        print(reply)

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
