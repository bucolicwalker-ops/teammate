"""teammate · W8 —— Agent 服务化（FastAPI）。

起服务后持续运行，不重启进程，记忆/KB 只加载一次。
端点：
  POST /ask          发消息给 Agent，返回回复 + trace
  GET  /health       健康检查 + 基础 metric
  GET  /trace        列出所有 trace summary
  GET  /trace/{id}   查指定 trace 完整 spans

跑法：.venv/bin/python -m src.server  （端口 8000）
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from src.agent import MyAgent

load_dotenv()

app = FastAPI(title="teammate agent service")

# Agent 实例只创建一次（持久运行，记忆/KB 只加载一次）
agent = MyAgent(max_history=20, use_long_term=True, use_knowledge=True)
agent.load_knowledge(os.path.join(os.path.dirname(__file__), "..", "..", "bagu", "qbank.md"))


class AskRequest(BaseModel):
    msg: str


@app.post("/ask")
def ask(req: AskRequest):
    """发消息给 Agent，返回回复 + trace summary。"""
    reply = agent.ask(req.msg)
    trace = agent.tracer.traces[-1] if agent.tracer.traces else None
    return {
        "reply": reply,
        "trace_id": trace["trace_id"] if trace else None,
        "total_tokens": trace["total_tokens"] if trace else 0,
        "total_latency_s": trace["total_latency_s"] if trace else 0,
        "history_len": len(agent.history),
    }


@app.get("/health")
def health():
    """健康检查 + 基础 metric（harness 管理接口）。"""
    total_tokens = sum(t["total_tokens"] for t in agent.tracer.traces)
    return {
        "status": "ok",
        "history_len": len(agent.history),
        "trace_count": len(agent.tracer.traces),
        "total_tokens": total_tokens,
    }


@app.get("/trace")
def list_traces():
    """列出所有 trace summary（可观测端点）。"""
    return {
        "traces": [
            {
                "trace_id": t["trace_id"],
                "total_tokens": t["total_tokens"],
                "total_latency_s": t["total_latency_s"],
                "llm_steps": t["llm_steps"],
                "tool_steps": t["tool_steps"],
            }
            for t in agent.tracer.traces
        ]
    }


@app.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    """查指定 trace 的完整 spans（可观测端点）。"""
    trace = agent.tracer.get_trace(trace_id)
    if trace:
        return trace
    return {"error": "trace not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
