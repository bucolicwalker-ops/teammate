"""teammate · 和 MyAgent 对话——在终端里试一试。
跑法：.venv/bin/python -m src.chat   （从 teammate/ 根目录跑，输入 exit 退出）

W3 更新：MyAgent 现在有短期记忆了——连续对话能接住上下文。
试试："北京天气" → "那上海呢" → "哪个更热" → 看它能不能接住。
"""
from src.agent import MyAgent

agent = MyAgent(max_history=20)

print("=== 和 MyAgent 聊天（输入 exit 退出）===\n")
while True:
    try:
        msg = input("你: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if msg in ("exit", "quit", ""):
        break
    try:
        reply = agent.ask(msg)
        print(f"\nMyAgent: {reply}\n")
    except Exception as e:
        print(f"   ⚠️ 出错了: {e}\n")
print("\n拜拜~ 👋")
