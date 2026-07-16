# teammate

从零搭建 AI Agent 的学习项目 —— 一步步把 MyAgent 从"只会说话"变成"能动手、有记忆、能检索、会评估、会规划、能协作、可观测、可部署"。

## 当前进度

- **L0 W1** ✅ 结构化输出（Pydantic）
- **L0 W2** ✅ Function Calling（多工具调用循环 + 失败重试）
- **L1 W3** ✅ 记忆系统（短期 history + 长期向量记忆 + MLX 本地 embedding）
- **L2 W4** ✅ RAG 知识库（文档加载→语义切块→batch embed→检索→grounding）
- **L2 W5** ✅ 评估（RAG Triad：faithfulness / answer_relevance / context_relevance）
- **L3 W6** ✅ 规划 + 多 Agent（Plan-Execute + Supervisor 动态路由）
- **L4 W7** ✅ MCP + 真工具 + 失败处理（read_file 真工具 + timeout/retry/降级 + MCP server）
- **L5 W8** ✅ 可观测 + 服务化（trace/log/metric + token 追踪 + FastAPI 服务）
- **L5/L6 W9** ✅ 高可用 + 安全（用户隔离 + 限流 + prompt injection 防护 + Docker）

## 项目结构

```
teammate/
├── src/
│   ├── agent.py        # MyAgent：工具循环 + 记忆 + RAG + trace + Plan-Execute
│   ├── tools.py        # 工具实现 + TOOL_REGISTRY + execute_tool（含失败处理）
│   ├── memory.py       # VectorMemory：跨 session 向量记忆
│   ├── rag.py          # KnowledgeBase：RAG pipeline（加载→切块→检索→注入）
│   ├── embedder.py     # Embedder：embedding 后端抽象（HTTP + 本地 MLX + batch）
│   ├── embed_server.py # FastAPI embedding 服务（OpenAI 兼容 /v1/embeddings）
│   ├── mcp_server.py   # MCP server（JSON-RPC over stdio，独立进程）
│   ├── eval.py         # RAG Triad 评估器（LLM-as-judge + eval 集 + ship gate）
│   ├── trace.py        # Tracer：trace_id + span + metric（token/延迟/步数）
│   ├── security.py     # 安全防护（prompt injection 检测 + 输出脱敏）
│   ├── team.py         # 多 Agent 协作（Supervisor 动态路由 + Sequential baseline）
│   ├── server.py       # FastAPI 服务（用户隔离 + 限流 + 安全 + /ask /health /trace）
│   └── chat.py         # 终端交互入口
├── Dockerfile          # 容器化部署
├── docker-compose.yml  # 服务编排
├── .env.example
├── .gitignore
└── requirements.txt
```

## 快速开始

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # 填入 API key

# 基础：多轮对话 + 工具调用
.venv/bin/python -m src.agent

# 交互聊天
.venv/bin/python -m src.chat

# RAG 问答（加载知识库 → 检索 → 生成）
.venv/bin/python -m src.agent rag

# 跨 session 记忆验证
.venv/bin/python -m src.agent session2

# Plan-Execute 规划
.venv/bin/python -m src.agent plan

# 真工具 + 失败处理
.venv/bin/python -m src.agent tools

# MCP 独立进程调用
.venv/bin/python -m src.agent mcp

# 多 Agent Supervisor 动态路由
.venv/bin/python -m src.team

# RAG 评估（跑 10 题 Triad → 出成绩单）
.venv/bin/python -m src.eval

# embedding 服务（模型加载一次共享）
.venv/bin/python -m src.embed_server

# Agent 服务（用户隔离 + 安全 + 限流 + trace）
.venv/bin/python -m src.server
# curl -X POST 'localhost:8000/ask?user_id=alice' -H "Content-Type: application/json" -d '{"msg":"你好"}'

# Docker 部署
docker-compose up
```

## 架构演进

```
W1: ask() — 单次 LLM + 结构化输出
W2: ask() — 工具循环（tool_use → execute → result → 循环）
W3: MyAgent — self.history（短期）+ VectorMemory（长期，MLX embedding）
W4: + KnowledgeBase — RAG pipeline（语义切块 + batch embed + grounding）
W5: + eval.py — RAG Triad 评估 + ship gate baseline
W6: + plan_and_execute() + team.py — Plan-Execute + Supervisor 动态路由
W7: + tools.py + mcp_server.py — 真工具 + 失败处理 + MCP 标准化
W8: + trace.py + server.py — 可观测（trace/token/metric）+ FastAPI 服务化
W9: + security.py + Docker — 用户隔离 + 安全 + 限流 + 容器化部署
```
