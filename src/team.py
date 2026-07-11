"""teammate · L3 W6 —— 多 Agent 协作。
两种模式对比：
- TeamSequential：固定流水线（researcher → writer → reviewer），baseline 对比
- TeamSupervisor：动态路由，LLM 决定下一步交给谁（真正的 Agent 模式）

跑法：.venv/bin/python -m src.team
"""
import os
from anthropic import Anthropic
from dotenv import load_dotenv
from src.agent import MyAgent, DEFAULT_SYSTEM

load_dotenv()
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
MODEL = os.getenv("MODEL", "glm-5.2")

AGENT_ROLES = {
    "researcher": "你是检索专家。只负责查资料、搜索知识库、调用搜索工具。不要写最终答案。",
    "writer": "你是写作专家。只负责组织语言、结构化输出。基于已有资料写答案。",
    "reviewer": "你是审查专家。只负责检查答案质量、准确性、有没有遗漏。给出修改建议或确认通过。",
}


class TeamSequential:
    """固定流水线：researcher → writer → reviewer。

    路由方式：硬编码顺序。简单、可预测，但不是 Agent——是 workflow。
    作为 baseline 和 TeamSupervisor 对比。
    """

    def __init__(self, max_history=20, use_long_term=False):
        self.researcher = MyAgent(max_history=max_history, use_long_term=use_long_term,
                                   system=AGENT_ROLES["researcher"])
        self.writer = MyAgent(max_history=max_history, use_long_term=use_long_term,
                               system=AGENT_ROLES["writer"])
        self.reviewer = MyAgent(max_history=max_history, use_long_term=use_long_term,
                                 system=AGENT_ROLES["reviewer"])

    def handle(self, user_msg: str) -> str:
        print(f"\n  🔧 Sequential 模式（固定流水线）")
        print(f"  ── researcher ──")
        research = self.researcher.ask(user_msg)

        print(f"\n  ── writer ──")
        draft = self.writer.ask(f"基于以下资料回答用户问题。\n用户问题: {user_msg}\n资料: {research}")

        print(f"\n  ── reviewer ──")
        review = self.reviewer.ask(f"检查以下答案的质量和准确性。\n用户问题: {user_msg}\n答案: {draft}")

        if "通过" in review:
            return draft
        return f"{draft}\n\n[审查意见: {review}]"


class TeamSupervisor:
    """动态路由 Supervisor：LLM 决定下一步交给哪个 Agent。

    路由方式：_route() 是一次 LLM 调用——路由决策本身是 AI 做的，不是 if-else。
    这是"Agent"模式——代码不决定流程，LLM 决定。
    """

    def __init__(self, max_history=20, use_long_term=False, max_rounds=6):
        self.agents = {
            name: MyAgent(max_history=max_history, use_long_term=use_long_term,
                          system=role)
            for name, role in AGENT_ROLES.items()
        }
        self.max_rounds = max_rounds

    def handle(self, user_msg: str) -> str:
        print(f"\n  🧠 Supervisor 模式（动态路由）")
        context = ""
        next_agent = self._route(user_msg, context)

        for round_num in range(self.max_rounds):
            if next_agent == "done":
                print(f"  ✅ Supervisor 判定完成")
                break

            print(f"\n  ── 第{round_num+1}轮: {next_agent} 接球 ──")
            agent = self.agents.get(next_agent)
            if not agent:
                print(f"  ⚠️ 未知角色: {next_agent}，结束")
                break

            output = agent.ask(
                f"用户问题: {user_msg}\n已有结果: {context}\n\n你负责的部分："
            )
            context += f"\n[{next_agent}]: {output}"

            next_agent = self._route(user_msg, context)

        if next_agent != "done":
            context += f"\n[⚠️ 达到最大轮数 {self.max_rounds}，强制结束]"

        return context

    def _route(self, user_msg: str, context: str) -> str:
        """LLM 决定下一步交给谁。

        这是 Supervisor 的核心——路由决策是 AI 做的，不是硬编码 if-else。
        球权传递模式：supervisor 决定下一个 agent → agent 执行 → supervisor 再决定。
        """
        available = ", ".join(AGENT_ROLES.keys())
        resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=(
                f"你是团队协调者。根据任务和已有结果，决定下一步交给谁。\n"
                f"可选角色: {available}, done（已完成）\n"
                f"只返回一个角色名，不要其他内容。"
            ),
            messages=[{"role": "user", "content": (
                f"任务: {user_msg}\n"
                f"已有结果: {context}\n"
                f"下一步交给谁？"
            )}],
        )
        decision = "".join(b.text for b in resp.content if b.type == "text").strip().lower()
        decision = decision.split()[0] if decision else "done"
        for role in AGENT_ROLES:
            if decision == role:
                return role
        return "done"


if __name__ == "__main__":
    import sys

    task = "什么是 RAG？请检索资料后给出结构化回答，并检查质量。"

    if len(sys.argv) > 1 and sys.argv[1] == "sequential":
        team = TeamSequential()
    else:
        team = TeamSupervisor()

    print("=" * 60)
    print(f"任务: {task}")
    print(f"模式: {type(team).__name__}")
    print("=" * 60)

    result = team.handle(task)
    print(f"\n{'=' * 60}")
    print(f"最终结果:")
    print(f"{'=' * 60}")
    print(result)
