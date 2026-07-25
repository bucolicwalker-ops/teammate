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
import re
import requests
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
    },
}

TOOLS = [
    {"name": name, "description": cfg["description"], "input_schema": cfg["schema"]}
    for name, cfg in TOOL_REGISTRY.items()
]

TOOL_TIMEOUT = 10  # 秒
MAX_RETRIES = 2
RETRYABLE_ERRORS = (TimeoutError, ConnectionError)  # 临时性错误才重试


def execute_tool(name: str, args: dict) -> str:
    """执行工具，带 timeout + retry + 分级降级。

    失败处理 taxonomy：
    1. 临时性错误（TimeoutError/ConnectionError）→ 重试
    2. 永久性错误（FileNotFoundError/PermissionError）→ 不重试，错误回灌
    3. 未知错误 → 错误回灌 + 类型信息
    """
    if name not in TOOL_REGISTRY:
        return f"错误：未知工具 '{name}'，可用工具：{list(TOOL_REGISTRY.keys())}"

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = _call_with_timeout(TOOL_REGISTRY[name]["fn"], args, TOOL_TIMEOUT)
            return str(result)
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
