"""teammate MCP server — 通过 MCP 协议暴露工具（JSON-RPC over stdio）。

被 Agent 作为子进程启动，通过 stdin/stdout 通信。
协议：JSON-RPC 2.0，方法 tools/list + tools/call。

跑法：被 src.agent.MCPClient 自动 spawn，不需要手动跑。
"""
import json
import sys

# 从 agent.py 导入工具注册表和执行器（含失败处理）
from src.tools import TOOL_REGISTRY, execute_tool


def _get_tools():
    """返回 MCP 格式的工具列表。"""
    return [
        {
            "name": name,
            "description": cfg["description"],
            "inputSchema": cfg["schema"],
        }
        for name, cfg in TOOL_REGISTRY.items()
    ]


def _handle_request(req: dict) -> dict:
    """处理单个 JSON-RPC 请求，返回 JSON-RPC 响应。"""
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}},
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _get_tools()}}

    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {})
        result = execute_tool(name, args)
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"content": [{"type": "text", "text": result}]},
        }

    return {"jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"}}


def main():
    """主循环：从 stdin 读 JSON-RPC 请求，处理后写响应到 stdout。

    JSON-RPC 2.0 规则：notification（无 id）不回响应。
    如果对 notification 回响应，client 不读会流串行 → 后续读错位。
    """
    req = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            # notification（无 id）不回响应，静默处理
            if "id" not in req:
                _handle_request(req)
                continue
            resp = _handle_request(req)
        except Exception as e:
            req_id = req.get("id") if req else None
            resp = {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32603, "message": str(e)}}
        print(json.dumps(resp, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
