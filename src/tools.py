"""teammate · 工具层 —— 工具实现 + 注册表 + 执行器。

分离原因：mcp_server.py 只需要工具层，不需要加载 agent.py 的
Anthropic client / VectorMemory / KnowledgeBase（避免子进程浪费）。

架构：
  tools.py       ← 工具实现 + TOOL_REGISTRY + execute_tool（"调什么"）
  mcp_server.py  ← MCP 协议层（from src.tools import ...，"怎么暴露"）
  agent.py       ← MyAgent + MCPClient（from src.tools import ...，"谁来用"）
"""
import ast
import json
import operator
import os
import re
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


# ============================================================
# 工具实现
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


def read_file(path: str) -> str:
    """读取文件内容（W7 真工具，不 mock）。

    会撞到真实失败：FileNotFoundError / PermissionError / UnicodeDecodeError。
    f.read(2000) 只读前 2000 字——防大文件 OOM。
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read(2000)


def web_search(query: str) -> str:
    """搜索网络，返回带URL的结果（DuckDuckGo，免费无需API key）。

    返回 JSON 字符串：[{"url":..., "title":..., "snippet":...}]
    最多 5 条，每条带 URL 供 fetch_url 深入阅读。
    """
    from ddgs import DDGS
    with DDGS() as ddg:
        results = list(ddg.text(query, max_results=5))
    items = [{"url": r.get("href") or r.get("link", ""),
              "title": r.get("title", ""),
              "snippet": r.get("body") or r.get("snippet", "")}
             for r in results]
    return json.dumps(items, ensure_ascii=False)


def _is_safe_url(url: str) -> bool:
    """检查 URL 是否安全（防 SSRF）——拒绝 localhost/内网/非 http(s)。"""
    from urllib.parse import urlparse
    import ipaddress
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return False
    hostname = parsed.hostname or ""
    if not hostname:
        return False
    if hostname in ('localhost', '0.0.0.0', '::1', '::'):
        return False
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return False
    except ValueError:
        pass  # 域名（非 IP），允许
    return True


def fetch_url(url: str) -> str:
    """抓取网页内容，HTML → Markdown（前3000字）。

    用 markdownify 做 HTML→Markdown 转换，保持标题/链接结构。
    超时/404 等错误抛回 execute_tool 的 retry/降级机制处理。
    SSRF 防护：拒绝 localhost/内网/非 http(s) URL。
    """
    if not _is_safe_url(url):
        return f"错误：URL 不安全（拒绝 localhost/内网/非 http(s)）：{url}"
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    from markdownify import markdownify as html_to_md
    text = html_to_md(resp.text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text[:3000]


# ============================================================
# 注册表 + 执行器
# ============================================================

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
    "read_file": {
        "fn": read_file,
        "description": "读取文件内容（限2000字）",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径，如 src/agent.py"},
            },
            "required": ["path"],
        },
    },
    "web_search": {
        "fn": web_search,
        "description": "搜索网络，返回带URL的结果（DuckDuckGo）。返回JSON数组，每条含url/title/snippet。用fetch_url深入阅读感兴趣的页面。",
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        },
        "timeout": 30,
    },
    "fetch_url": {
        "fn": fetch_url,
        "description": "抓取网页内容，转为Markdown文本（限3000字）。传入web_search返回的URL。",
        "schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要抓取的网页URL"},
            },
            "required": ["url"],
        },
        "timeout": 20,
    },
}

# Subagent late binding — handler set from agent.py（需要 LLM client）
_task_handler = None

def set_task_handler(handler):
    global _task_handler
    _task_handler = handler

def run_task(description: str) -> str:
    if _task_handler:
        return _task_handler(description)
    return "Error: subagent not configured"

TOOL_REGISTRY["task"] = {
    "fn": run_task,
    "description": "启动子Agent处理复杂子任务。只回传最终结论，中间过程不污染主对话。",
    "schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "子任务描述"},
        },
        "required": ["description"],
    },
}

TOOLS = [
    {"name": name, "description": cfg["description"], "input_schema": cfg["schema"]}
    for name, cfg in TOOL_REGISTRY.items()
]

# 子 Agent 工具集：去掉 task（防递归）
SUB_TOOLS = [t for t in TOOLS if t["name"] != "task"]

TOOL_TIMEOUT = 10  # 秒
MAX_RETRIES = 2
RETRYABLE_ERRORS = (TimeoutError, ConnectionError)  # 临时性错误才重试

WORKDIR = Path(__file__).parent.parent.resolve()  # teammate 项目根目录


# ============================================================
# Permission System (from learn-claude-code s03)
# 三道闸门：deny list → rules → user approval
# ============================================================

# Gate 1: 硬拒绝列表 — 路径/模式直接拦，不问用户
DENY_PATTERNS = [".env", ".ssh", ".git/config", "/etc/passwd", "/etc/shadow", "/etc/sudoers"]


def check_deny_list(tool_name: str, args: dict) -> str | None:
    """Gate 1: 检查是否命中拒绝列表。"""
    for pattern in DENY_PATTERNS:
        for val in args.values():
            if isinstance(val, str) and pattern in val:
                return f"Blocked: '{pattern}' is on the deny list"
    return None


# Gate 2: 规则匹配 — 声明式，可扩展
PERMISSION_RULES = [
    {
        "tools": ["read_file"],
        "check": lambda args: not _is_safe_path(args.get("path", "")),
        "message": "Reading outside workspace",
    },
]


def _is_safe_path(p: str) -> bool:
    """检查路径是否在工作区内。"""
    try:
        path = Path(p).resolve()
        return path.is_relative_to(WORKDIR)
    except Exception:
        return False


def check_rules(tool_name: str, args: dict) -> str | None:
    """Gate 2: 检查规则匹配。命中才走 Gate 3。"""
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None


# Gate 3: 用户审批
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    """Gate 3: 暂停等用户确认。非交互模式自动拒绝。"""
    import sys
    if not sys.stdin.isatty():
        return "deny"  # 非交互模式（server/CI）自动拒绝
    print(f"\n\033[33m⚠  {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"


def check_permission(name: str, args: dict) -> tuple[bool, str]:
    """三道闸门权限管线。返回 (是否允许, 原因)。"""
    # Gate 1: deny list
    reason = check_deny_list(name, args)
    if reason:
        return False, reason
    # Gate 2: rules → Gate 3: user approval
    reason = check_rules(name, args)
    if reason:
        if ask_user(name, args, reason) == "deny":
            return False, reason
    return True, ""


# ============================================================
# Hook System (from learn-claude-code s04)
# 扩展逻辑挂到 hook 上，execute_tool / agent loop 不改
# ============================================================

HOOKS = {"PreToolUse": [], "PostToolUse": [], "Stop": [], "UserPromptSubmit": []}


def register_hook(event: str, callback):
    """注册 hook：event 发生时调用 callback。"""
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    """触发事件的所有 hook。第一个返回非 None 的拦截。"""
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


# --- hook 实现 ---

def permission_hook(name: str, args: dict) -> str | None:
    """PreToolUse: s03 权限检查（三道闸门搬过来）。"""
    allowed, reason = check_permission(name, args)
    if not allowed:
        return f"Permission denied: {reason}"
    return None


def log_hook(name: str, args: dict) -> str | None:
    """PreToolUse: 记录每次工具调用。"""
    preview = str(list(args.values())[:2])[:60]
    print(f"  [HOOK] {name}({preview})")
    return None


# 注册 hook
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)


def execute_tool(name: str, args: dict) -> str:
    """执行工具，带 timeout + retry + 分级降级。

    失败处理 taxonomy：
    1. 临时性错误（TimeoutError/ConnectionError）→ 重试
    2. 永久性错误（FileNotFoundError/PermissionError）→ 不重试，错误回灌
    3. 未知错误 → 错误回灌 + 类型信息
    """
    if name not in TOOL_REGISTRY:
        return f"错误：未知工具 '{name}'，可用工具：{list(TOOL_REGISTRY.keys())}"

    # s04: PreToolUse hooks（替代 s03 的 check_permission 直接调用）
    blocked = trigger_hooks("PreToolUse", name, args)
    if blocked:
        print(f"  ⛔ {blocked}")
        return f"⛔ {blocked}"

    for attempt in range(MAX_RETRIES + 1):
        try:
            timeout = TOOL_REGISTRY[name].get("timeout", TOOL_TIMEOUT)
            result = _call_with_timeout(TOOL_REGISTRY[name]["fn"], args, timeout)
            output = str(result)
            # s04: PostToolUse hooks
            trigger_hooks("PostToolUse", name, args, output)
            return output
        except RETRYABLE_ERRORS as e:
            if attempt < MAX_RETRIES:
                print(f"  ⚠️ 工具 {name} 第{attempt+1}次失败（{type(e).__name__}），重试...")
                continue
            return f"工具 {name} 重试{MAX_RETRIES}次后仍失败：{type(e).__name__}: {e}"
        except Exception as e:
            return f"工具 {name} 执行出错（不可重试）：{type(e).__name__}: {e}"

    return f"工具 {name} 超时且重试失败"


def _call_with_timeout(fn, args: dict, timeout: int):
    """用 ThreadPoolExecutor 实现工具调用的超时控制。

    跨线程安全（SIGALRM 只在主线程有效，uvicorn 线程池里不能用）。
    超时后线程不会被杀（Python 限制），继续后台跑——学习项目够用。
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, **args)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            raise TimeoutError(f"工具执行超时（{timeout}秒）")
