"""Stage 4 small demo: research → write → review → revise.

对照 learn-claude-code s15/s16 验证 Stage 4 的 5 个概念。
- 角色: 4 个 MyAgent 实例，各自 role-specific system prompt
- 协调: sync sequential（team.py 现状模式）
- 职责边界: system prompt + max_steps
- 循环: max_steps=2 硬上限 + revise 闭环
- 单 agent 退化: 简单任务也会走完整流水线（故意不退化，演示用）

跑法: .venv/bin/python demo_stage4.py
"""
import os
import sys
from dotenv import load_dotenv
from src.agent import MyAgent

load_dotenv()

ROLES = {
    "researcher": "你是检索专家。只负责查资料、调用搜索工具。不要写最终答案。如果任务不需要检索，直接说'无需检索，任务可直接回答'。",
    "writer": "你是写作专家。只负责组织语言、结构化输出。基于已有资料写答案。",
    "reviewer": "你是审查专家。只负责检查答案质量和准确性。给出修改建议或确认通过。回答末尾必须包含'通过'或'不通过'。",
    "reviser": "你是修订专家。只负责根据审查意见修订答案。输出最终版答案。",
}


def run_demo(task: str, max_steps: int = 2):
    print("=" * 60)
    print(f"任务: {task}")
    print(f"模式: Sequential (research→write→review→revise)")
    print(f"每 agent max_steps: {max_steps}")
    print("=" * 60)

    researcher = MyAgent(max_history=10, use_long_term=False, system=ROLES["researcher"])
    writer = MyAgent(max_history=10, use_long_term=False, system=ROLES["writer"])
    reviewer = MyAgent(max_history=10, use_long_term=False, system=ROLES["reviewer"])
    reviser = MyAgent(max_history=10, use_long_term=False, system=ROLES["reviser"])

    print("\n── Phase 1: researcher ──")
    research = researcher.ask(task, max_steps=max_steps)
    print(f"\n[research 输出 / 前 200 字]\n{research[:200]}")

    print("\n── Phase 2: writer ──")
    draft = writer.ask(
        f"基于以下资料回答用户问题。\n用户问题: {task}\n资料: {research}",
        max_steps=max_steps,
    )
    print(f"\n[draft 输出 / 前 200 字]\n{draft[:200]}")

    print("\n── Phase 3: reviewer ──")
    review = reviewer.ask(
        f"检查以下答案的质量和准确性。\n用户问题: {task}\n答案: {draft}",
        max_steps=max_steps,
    )
    print(f"\n[review 输出 / 前 200 字]\n{review[:200]}")

    print("\n── Phase 4: reviser ──")
    if "通过" in review:
        print("✅ reviewer 通过，无需修订")
        final = draft
    else:
        print("⚠️ reviewer 不通过，触发 revise")
        final = reviser.ask(
            f"根据审查意见修订答案。\n用户问题: {task}\n原答案: {draft}\n审查意见: {review}",
            max_steps=max_steps,
        )
        print(f"\n[final 输出 / 前 200 字]\n{final[:200]}")

    return final, research, draft, review


if __name__ == "__main__":
    task = "用一句话解释什么是递归"
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    result, *_ = run_demo(task, max_steps=2)
    print(f"\n{'=' * 60}")
    print(f"最终结果:")
    print(f"{'=' * 60}")
    print(result[:1000])
