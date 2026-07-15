"""teammate · W8 —— 可观测：trace + log + metric。

Tracer 记录每次 ask() 的完整链路：
  user_input → llm_call(token/延迟) → tool_call(工具/延迟) → ... → final_reply → summary

trace 存内存 list，可从 /trace 端点查询。
生产版对比：生产级可观测平台（LangSmith/Langfuse）做持久化（DB + Redis），
我们这是简化版（内存 list），之后对比理解 trade-off。
"""
import json
import time
import uuid


class Tracer:
    """简易 trace + logger：每次 ask() 生成 trace_id，记录每步 span。

    span 结构：{trace_id, event, timestamp, data}
    event 类型：user_input / llm_call / tool_call / final_reply / summary
    """

    def __init__(self):
        self.traces: list[dict] = []  # 所有完成的 trace
        self._current_spans: list[dict] = []  # 当前 trace 的 spans

    def start_trace(self, user_msg: str) -> str:
        """开始一次 trace，返回 trace_id。"""
        trace_id = uuid.uuid4().hex[:8]
        self._current_spans = []
        self._log(trace_id, "user_input", {"msg": user_msg[:100]})
        return trace_id

    def log_llm_call(self, trace_id: str, step: int,
                     input_tokens: int, output_tokens: int, latency: float):
        """记录一次 LLM 调用（token + 延迟 = 成本 + 性能 metric）。"""
        self._log(trace_id, "llm_call", {
            "step": step,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_s": round(latency, 2),
        })

    def log_tool_call(self, trace_id: str, tool_name: str,
                      args: dict, result: str, latency: float):
        """记录一次工具调用。"""
        self._log(trace_id, "tool_call", {
            "tool": tool_name,
            "args": str(args)[:80],
            "result": str(result)[:80],
            "latency_s": round(latency, 3),
        })

    def finish_trace(self, trace_id: str, reply: str) -> dict:
        """结束 trace，计算 summary metric，存入 traces。"""
        self._log(trace_id, "final_reply", {"reply": reply[:100]})

        llm_spans = [s for s in self._current_spans if s["event"] == "llm_call"]
        tool_spans = [s for s in self._current_spans if s["event"] == "tool_call"]
        total_input = sum(s["data"]["input_tokens"] for s in llm_spans)
        total_output = sum(s["data"]["output_tokens"] for s in llm_spans)
        total_latency = sum(s["data"]["latency_s"] for s in llm_spans + tool_spans)

        summary = {
            "trace_id": trace_id,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_latency_s": round(total_latency, 2),
            "llm_steps": len(llm_spans),
            "tool_steps": len(tool_spans),
            "spans": self._current_spans,
        }
        self.traces.append(summary)
        self._log(trace_id, "summary", {
            "total_tokens": total_input + total_output,
            "total_latency_s": round(total_latency, 2),
            "llm_steps": len(llm_spans),
            "tool_steps": len(tool_spans),
        })
        return summary

    def get_all_traces(self) -> list[dict]:
        """返回所有完成的 trace summary。"""
        return self.traces

    def get_trace(self, trace_id: str) -> dict | None:
        """查指定 trace 的完整 spans。"""
        for t in self.traces:
            if t["trace_id"] == trace_id:
                return t
        return None

    def _log(self, trace_id: str, event: str, data: dict):
        """记录一个 span + 输出结构化日志。"""
        span = {
            "trace_id": trace_id,
            "event": event,
            "timestamp": time.time(),
            "data": data,
        }
        self._current_spans.append(span)
        # 结构化日志（JSON 一行，可 grep/过滤——生产版用 logging 模块）
        print(f"  📊 {json.dumps(span, ensure_ascii=False)[:120]}", flush=True)
