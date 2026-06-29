# teammate

从零搭建 AI Agent 的学习项目 —— 一步步把 MyAgent 从"只会说话"变成"能动手、有记忆、会推理"。

## 当前进度

- **L0 W1** ✅ 结构化输出（Pydantic）
- **L0 W2** ✅ Function Calling（多工具调用循环）
- **L0 W3** 🔜 记忆（对话历史）

## 项目结构

```
teammate/
├── src/                # 源码
│   ├── agent.py        # MyAgent 核心（工具注册表 + 调用循环）
│   └── chat.py         # 终端交互入口
├── tests/              # 测试
│   ├── test_*.py       # 单元测试（pytest 自动收集）
│   └── smoke_*.py     # 冒烟测试（手动跑，验证能力通不通）
├── docs/               # 文档
│   └── 坑日志.md       # 踩坑记录
├── .env.example        # 环境变量模板
├── .gitignore
└── requirements.txt    # 依赖清单
```

命名约定：`src/<模块>.py` ←→ `tests/test_<模块>.py` ←→ `docs/<模块>.md`

## 快速开始

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # 填入你的 API key
.venv/bin/python -m src.agent    # 跑测试用例
.venv/bin/python -m src.chat     # 进交互聊天
```
