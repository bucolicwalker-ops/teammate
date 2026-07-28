"""Agent 基类接口 — 所有 Agent（MyAgent/CodexAgent/ClaudeAgent）实现此接口。

设计原则（参照 cat-cafe）：不同 Agent 能力不同、模型不同、工具不同，
但对外接口统一——调用方不关心底层是 in-process Python 还是 CLI 子进程。

隔离三层：
- 进程隔离：MyAgent in-process；Codex/CC 子进程
- Context 隔离：各 Agent 独立 history + system prompt
- 工具隔离：各 Agent 受限工具集
"""
from abc import ABC, abstractmethod


class AgentBase(ABC):
    """Agent 基类接口。"""

    @abstractmethod
    def ask(self, user_msg: str, max_steps: int = 5) -> str:
        """基础对话：输入消息，返回回复。"""
        pass

    @abstractmethod
    def research(self, topic: str, max_steps: int = 12) -> str:
        """研究模式：输入主题，返回带引用的研究报告。"""
        pass

    @abstractmethod
    def close(self):
        """清理资源（MCP 连接、数据库等）。"""
        pass
