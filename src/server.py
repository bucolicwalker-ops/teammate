"""teammate · W9 —— Agent 服务（用户隔离 + 安全 + 限流 + Docker 就绪）。

端点：
  POST /ask?user_id=xxx   发消息（per-user agent，记忆隔离 + 安全检查 + 限流）
  GET  /health            健康检查 + 全局 metric
  GET  /trace             列出所有 trace
  GET  /trace/{id}        查指定 trace

跑法：
  本地：.venv/bin/python -m src.server
  Docker：docker-compose up
"""
import os
import threading
from collections import defaultdict
from time import time
from fastapi import FastAPI, Query
from pydantic import BaseModel
from dotenv import load_dotenv
from src.agent import MyAgent
from src.rag import KnowledgeBase
from src.security import check_input, check_output

load_dotenv()
EMBED_ENDPOINT = os.getenv("EMBED_ENDPOINT")  # Docker 模式连宿主机 embed_server
KB_PATH = os.getenv("KB_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "bagu", "qbank.md"))

app = FastAPI(title="teammate agent service")

# ============================================================
# 共享知识库（只读，所有用户共享，只加载一次）
# ============================================================

_kb = KnowledgeBase("data/knowledge.json", EMBED_ENDPOINT)
if not _kb.chunks:
    _kb.load_document(KB_PATH)
    print(f"  📄 知识库已加载（共享）")


# ============================================================
# 用户隔离：per-user agent 实例（独立 memory + history + LRU 淘汰）
# ============================================================

agents: dict[str, MyAgent] = {}
_lock = threading.Lock()  # 保护 agents dict + 限流计数（get_agent 和 check_rate_limit 都用）
MAX_AGENTS = 50  # 最大并发用户——超过淘汰最旧的（防内存泄漏 + 子进程泄漏）


def get_agent(user_id: str) -> MyAgent:
    """获取或创建用户专属 Agent。线程安全 + LRU 淘汰。"""
    with _lock:
        if user_id in agents:
            return agents[user_id]

        # LRU 淘汰：超过 MAX_AGENTS 时销毁最旧的（含 mcp_client.close()）
        if len(agents) >= MAX_AGENTS:
            oldest = next(iter(agents))
            old = agents.pop(oldest)
            if old.mcp_client:
                old.mcp_client.close()
            print(f"  🧹 淘汰用户 {oldest}（超过 {MAX_AGENTS} 上限）")

        a = MyAgent(
            max_history=20,
            use_long_term=True,
            memory_path=f"data/memory_{user_id}.json",
            use_knowledge=False,
            embed_endpoint=EMBED_ENDPOINT,
            use_mcp=True,
        )
        a.knowledge = _kb
        agents[user_id] = a
        print(f"  👤 用户 {user_id} 的 Agent 已创建（当前 {len(agents)}/{MAX_AGENTS}）")
        return agents[user_id]


# ============================================================
# 限流：每用户每分钟最多 RATE_LIMIT 个请求
# ============================================================

RATE_LIMIT = 10
_last_request = defaultdict(float)
_request_count = defaultdict(int)


def check_rate_limit(user_id: str) -> tuple[bool, str]:
    with _lock:
        now = time()
        if now - _last_request[user_id] > 60:
            _request_count[user_id] = 0
            _last_request[user_id] = now
        _request_count[user_id] += 1
        if _request_count[user_id] > RATE_LIMIT:
            return False, f"请求频率超限（{RATE_LIMIT}/分钟）"
        return True, "ok"


# ============================================================
# API 端点
# ============================================================

class AskRequest(BaseModel):
    msg: str


@app.post("/ask")
def ask(req: AskRequest, user_id: str = Query("default")):
    """发消息给 Agent。per-user 隔离 + 安全检查 + 限流。"""
    # 1. 限流
    ok, reason = check_rate_limit(user_id)
    if not ok:
        return {"error": "rate_limited", "detail": reason}

    # 2. 输入安全检查
    ok, reason = check_input(req.msg)
    if not ok:
        return {"error": "blocked", "detail": reason}

    # 3. 获取用户专属 Agent
    agent = get_agent(user_id)

    # 4. 执行
    reply = agent.ask(req.msg)
    trace = agent.tracer.traces[-1] if agent.tracer.traces else None

    # 5. 输出安全检查
    ok, reason = check_output(reply)
    if not ok:
        reply = f"[回复已脱敏：{reason}]"

    return {
        "reply": reply,
        "user_id": user_id,
        "trace_id": trace["trace_id"] if trace else None,
        "total_tokens": trace["total_tokens"] if trace else 0,
        "total_latency_s": trace["total_latency_s"] if trace else 0,
        "history_len": len(agent.history),
    }


@app.get("/health")
def health():
    """健康检查 + 全局 metric。"""
    total_tokens = sum(
        t["total_tokens"]
        for a in agents.values()
        for t in a.tracer.traces
    )
    return {
        "status": "ok",
        "user_count": len(agents),
        "total_tokens": total_tokens,
        "kb_chunks": len(_kb.chunks),
    }


@app.get("/trace")
def list_traces(user_id: str = Query(None)):
    """列出 trace summary（可选按 user_id 过滤）。"""
    target_agents = [agents[user_id]] if user_id and user_id in agents else agents.values()
    traces = []
    for a in target_agents:
        for t in a.tracer.traces:
            traces.append({
                "trace_id": t["trace_id"],
                "total_tokens": t["total_tokens"],
                "total_latency_s": t["total_latency_s"],
                "llm_steps": t["llm_steps"],
                "tool_steps": t["tool_steps"],
            })
    return {"traces": traces}


@app.get("/trace/{trace_id}")
def get_trace(trace_id: str, user_id: str = Query(None)):
    """查指定 trace 完整 spans。"""
    target_agents = [agents[user_id]] if user_id and user_id in agents else agents.values()
    for a in target_agents:
        trace = a.tracer.get_trace(trace_id)
        if trace:
            return trace
    return {"error": "trace not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # 0.0.0.0 让 Docker 能访问
