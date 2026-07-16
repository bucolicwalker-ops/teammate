"""teammate · W9 —— 安全防护：prompt injection 检测 + 输出脱敏。

两层防护：
1. 输入检查：检测常见 injection 模式（"ignore previous instructions" 等）
2. 输出检查：检测回复是否泄露 API key / 文件路径 / system prompt

注意：这是规则匹配（简单但有漏报），生产版需要 LLM-as-judge + 规则组合。
"""
import re

# 常见 prompt injection 模式（不完整，生产需要持续更新）
INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|above|all)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all|any|previous)\s+(instructions?|prompts?)",
    r"you\s+are\s+now\s+(a|an)\s+\w+",
    r"act\s+as\s+(if|a|an)\s",
    r"new\s+(instructions?|rules?)\s*:",
    r"system\s+prompt\s*:",
    r"reveal\s+(your|the)\s+(system|prompt|instructions?)",
    r"what\s+are\s+your\s+(instructions?|rules?|system\s+prompt)",
]

# 敏感信息模式
SENSITIVE_PATTERNS = [
    r"sk-[a-zA-Z0-9]{20,}",      # OpenAI key
    r"ghp_[a-zA-Z0-9]{36}",      # GitHub token
    r"ANTHROPIC_API_KEY\s*=",
    r"api[_-]?key\s*[:=]\s*\S+",
    r"/Users/[^/]+/",             # macOS user path
]

MAX_INPUT_LEN = 5000  # 用户输入长度上限（防超长 prompt 烧 token）


def check_input(msg: str) -> tuple[bool, str]:
    """检查用户输入——prompt injection 检测 + 长度限制。

    返回 (safe, reason)。safe=False 时拒绝请求。
    """
    if len(msg) > MAX_INPUT_LEN:
        return False, f"输入过长（限 {MAX_INPUT_LEN} 字，当前 {len(msg)}）"

    lower = msg.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            return False, f"检测到可疑指令模式（prompt injection 疑似）"

    return True, "ok"


def check_output(reply: str) -> tuple[bool, str]:
    """检查 Agent 回复——敏感信息泄露检测。

    返回 (safe, reason)。safe=False 时替换回复为脱敏提示。
    """
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, reply, re.IGNORECASE):
            return False, f"输出可能包含敏感信息，已脱敏"
    return True, "ok"
