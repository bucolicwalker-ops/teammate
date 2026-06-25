"""teammate · 和 MyAgent 对话——在终端里试一试。
跑法：.venv/bin/python chat.py   （输入 exit 退出）

注意：现在 MyAgent 还没有记忆，每句话都是独立处理的——
它不记得你上一句说了啥。这正是 W3「记忆」要解决的，你试的时候能明显感觉到。
"""
from agent import ask

print("=== 和 MyAgent 聊天（输入 exit 退出）===\n")
while True:
    try:
        msg = input("你: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if msg in ("exit", "quit", ""):
        break
    try:
        r = ask(msg)
        print(f"MyAgent: {r.reply}")
        print(f"   └ [intent={r.intent} · need_followup={r.need_followup}]\n")
    except Exception as e:
        print(f"   ⚠️ 出错了: {e}\n")
print("\n拜拜~ 👋")
