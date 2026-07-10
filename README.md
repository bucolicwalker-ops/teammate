# teammate

从零搭建 AI Agent 的学习项目 —— 一步步把 MyAgent 从"只会说话"变成"能动手、有记忆、能检索、会评估"。

## 当前进度

- **L0 W1** ✅ 结构化输出（Pydantic）
- **L0 W2** ✅ Function Calling（多工具调用循环 + 失败重试）
- **L1 W3** ✅ 记忆系统（短期 history + 长期向量记忆 + MLX 本地 embedding）
- **L2 W4** ✅ RAG 知识库（文档加载→语义切块→batch embed→检索→grounding）
- **L2 W5** ✅ 评估（RAG Triad：faithfulness / answer_relevance / context_relevance）

## 项目结构

```
teammate/
├── src/
│   ├── agent.py        # MyAgent：工具循环 + 短期/长期记忆 + RAG 集成
│   ├── memory.py       # VectorMemory：跨 session 向量记忆
│   ├── rag.py          # KnowledgeBase：RAG pipeline（加载→切块→检索→注入）
│   ├── embedder.py     # Embedder：embedding 后端抽象（HTTP + 本地 MLX + batch）
│   ├── embed_server.py # FastAPI embedding 服务（OpenAI 兼容 /v1/embeddings）
│   ├── eval.py         # RAG Triad 评估器（LLM-as-judge + eval 集 + ship gate）
│   └── chat.py         # 终端交互入口
├── tests/
├── docs/
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

# RAG 评估（跑 10 题 Triad → 出成绩单）
.venv/bin/python -m src.eval

# embedding 服务（可选，模型加载一次共享）
.venv/bin/python -m src.embed_server
```

## 架构演进

```
W1: ask() — 单次 LLM + 结构化输出
W2: ask() — 工具循环（tool_use → execute → result → 循环）
W3: MyAgent — self.history（短期）+ VectorMemory（长期，MLX embedding）
W4: + KnowledgeBase — RAG pipeline（语义切块 + batch embed + grounding）
W5: + eval.py — RAG Triad 评估 + ship gate baseline
```
