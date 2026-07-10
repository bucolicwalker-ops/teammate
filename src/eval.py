"""teammate · W5 —— RAG Triad 评估器。
给 W4 的 RAG 出"成绩单"：faithfulness / answer_relevance / context_relevance。
跑法：.venv/bin/python -m src.eval   （需先加载知识库）

三个维度都用 LLM-as-judge（entailment 判断，不是关键词匹配）。
已知偏差：同一模型既生成又评判有自我偏好——gap，留进阶用多模型交叉。
"""
import json
import os
from anthropic import Anthropic
from dotenv import load_dotenv
from src.rag import KnowledgeBase
from src.agent import MyAgent, SYSTEM, TOOLS

load_dotenv()
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
MODEL = os.getenv("MODEL", "glm-5.2")


# ============================================================
# ① Eval 集（10 题，跨 D/E/F/G 域，5 已学 + 5 未学）
# ============================================================

EVAL_SET = [
    {
        "id": "D-Q09",
        "query": "agent项目的记忆系统怎么实现？",
        "expected_points": ["四积木embed存储召回注入", "全量存vs抽取存trade-off"],
        "domain": "D",
    },
    {
        "id": "D-Q10",
        "query": "agent怎么做上下文压缩？",
        "expected_points": ["截断摘要混合三策略", "Lost-in-the-Middle"],
        "domain": "D",
    },
    {
        "id": "D-Q16",
        "query": "tool call失败怎么处理？",
        "expected_points": ["错误回灌给模型", "超时重试策略"],
        "domain": "D",
    },
    {
        "id": "D-Q28",
        "query": "agent报错怎么办？",
        "expected_points": ["分层处理LLM工具死循环", "不崩加可观测"],
        "domain": "D",
    },
    {
        "id": "G-Q05",
        "query": "如何避免AI回答中的幻觉？",
        "expected_points": ["RAG接地加溯源", "后验校验"],
        "domain": "G",
    },
    {
        "id": "D-Q02",
        "query": "为什么要多Agent？单Agent不够吗？",
        "expected_points": ["上下文膨胀角色混杂", "通信成本状态同步"],
        "domain": "D",
    },
    {
        "id": "F-Q05",
        "query": "system prompt前缀稳定怎么做？",
        "expected_points": ["KV cache命中", "静态在前动态在后"],
        "domain": "F",
    },
    {
        "id": "F-Q06",
        "query": "流式输出怎么做？",
        "expected_points": ["SSE单向推送", "stream=True逐token"],
        "domain": "F",
    },
    {
        "id": "F-Q12",
        "query": "AI搜索和传统搜索有什么差异？",
        "expected_points": ["关键词vs语义", "链接列表vs生成答案"],
        "domain": "F",
    },
    {
        "id": "E-Q05",
        "query": "可观测性怎么做？",
        "expected_points": ["trace日志指标三件套", "可视化trace树"],
        "domain": "E",
    },
]


# ============================================================
# ② RAG Triad 评估器（LLM-as-judge）
# ============================================================

def _llm_judge(prompt: str) -> dict:
    """调 LLM 当裁判，返回 {score, reason}。解析失败默认 0.5。"""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system="你是 RAG 评估员。严格按指令返回 JSON，不要多余内容。必须基于实际分析打分，不要默认给中间分。",
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    # 提取 JSON（模型可能包裹 ```json ... ```）
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"score": 0.5, "reason": f"JSON parse failed: {text[:100]}"}


def eval_faithfulness(question: str, context: str, answer: str) -> dict:
    """faithfulness：答案是否忠于检索内容（entailment，不是关键词匹配）。

    判断答案的每个 claim 是否被检索内容支持——不是 grep 关键词，
    是语义蕴含（"我叫小明" 支持 "用户名是小明"，关键词不重叠但语义一样）。
    """
    prompt = f"""评估答案的 faithfulness（是否忠于检索内容）。

问题：{question}

检索内容：
{context}

答案：
{answer}

分析步骤：
1. 找出答案中不被检索内容支持的 claim（检索内容里没有的，包括编造/扩展/推理过度）
2. 合理常识推理不算 unsupported（如"这说明需要优化"是推理不是编造）
3. 计算 score：如果没有 unsupported claim，score=1.0；如果有，按比例降低

返回 JSON（简洁）：{{"unsupported": ["不被支持的claim1", ...], "score": 0.0-1.0, "reason": "一句话"}}"""
    return _llm_judge(prompt)


def eval_answer_relevance(question: str, answer: str) -> dict:
    """answer relevance：答案是否回答了用户问的问题。"""
    prompt = f"""判断以下答案是否回答了问题（answer relevance）。

问题：{question}

答案：
{answer}

规则：
- 答案是否直接回答了问题，不是答非所问
- 1.0 = 完全回答，0.0 = 完全不相关

返回 JSON：{{"score": 0.0-1.0, "reason": "简述"}}"""
    return _llm_judge(prompt)


def eval_context_relevance(question: str, context: str) -> dict:
    """context relevance：检索到的内容是否和问题相关。

    注意：不是用 cosine score 当分数——cosine 是召回信号（把可能相关的捞回来），
    真的相关性要 LLM 判断。
    """
    prompt = f"""判断以下检索内容是否与问题相关（context relevance）。

问题：{question}

检索内容：
{context}

规则：
- 检索到的内容是否能帮助回答这个问题
- 1.0 = 高度相关，0.0 = 完全不相关

返回 JSON：{{"score": 0.0-1.0, "reason": "简述"}}"""
    return _llm_judge(prompt)


# ============================================================
# ③ Runner：跑 eval + 出成绩单
# ============================================================

def run_eval(agent: MyAgent, kb: KnowledgeBase):
    """跑 10 题 RAG Triad 评估，出成绩单。"""
    results = []
    for i, item in enumerate(EVAL_SET):
        qid, query = item["id"], item["query"]
        print(f"\n{'─' * 60}")
        print(f"  [{i+1}/{len(EVAL_SET)}] {qid}: {query}")
        print(f"{'─' * 60}")

        # 1. 检索（单独调，拿 chunks 用于 context_relevance 评估）
        chunks = kb.retrieve(query, top_k=3)
        context = kb.format_context(chunks) if chunks else "(无检索结果)"

        # 2. 生成（agent.ask 内部也检索，这里用 agent 生成答案）
        agent.history.clear()  # 每题独立，不带上轮 history
        answer = agent.ask(query)
        print(f"  答案前80字: {answer[:80]}...")

        # 3. RAG Triad 评估
        print(f"  评估中...")
        f_result = eval_faithfulness(query, context, answer)
        a_result = eval_answer_relevance(query, answer)
        c_result = eval_context_relevance(query, context)

        f_score = f_result.get("score", 0.5)
        a_score = a_result.get("score", 0.5)
        c_score = c_result.get("score", 0.5)

        print(f"  faithfulness: {f_score:.2f} | answer_rel: {a_score:.2f} | context_rel: {c_score:.2f}")

        results.append({
            "id": qid,
            "query": query,
            "domain": item["domain"],
            "faithfulness": f_score,
            "answer_relevance": a_score,
            "context_relevance": c_score,
            "avg": (f_score + a_score + c_score) / 3,
            "unsupported": f_result.get("unsupported", []),
        })

    # 成绩单
    print(f"\n{'=' * 60}")
    print(f"  RAG Triad 成绩单（{len(results)} 题）")
    print(f"{'=' * 60}")

    n = len(results)
    avg_f = sum(r["faithfulness"] for r in results) / n
    avg_a = sum(r["answer_relevance"] for r in results) / n
    avg_c = sum(r["context_relevance"] for r in results) / n
    avg_total = (avg_f + avg_a + avg_c) / 3

    print(f"\n  faithfulness（不幻觉）:  {avg_f:.2f}  ({sum(1 for r in results if r['faithfulness']>=0.7)}/{n} 及格)")
    print(f"  answer_relevance（答对题）: {avg_a:.2f}  ({sum(1 for r in results if r['answer_relevance']>=0.7)}/{n} 及格)")
    print(f"  context_relevance（检索准）: {avg_c:.2f}  ({sum(1 for r in results if r['context_relevance']>=0.7)}/{n} 及格)")
    print(f"  ────────────────────────────────")
    print(f"  总分: {avg_total:.2f}")

    print(f"\n  逐题明细：")
    print(f"  {'ID':<8} {'F':>5} {'A':>5} {'C':>5} {'Avg':>5}  Query")
    print(f"  {'─'*50}")
    for r in results:
        print(f"  {r['id']:<8} {r['faithfulness']:>.2f} {r['answer_relevance']:>.2f} {r['context_relevance']:>.2f} {r['avg']:>.2f}  {r['query'][:20]}")

    # 保存 baseline
    report_path = "data/eval_baseline.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {"faithfulness": avg_f, "answer_relevance": avg_a,
                        "context_relevance": avg_c, "total": avg_total},
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  📊 Baseline 已存：{report_path}")
    print(f"  下次改动后重跑，对比 baseline 看分数有没有掉（ship gate）")


if __name__ == "__main__":
    agent = MyAgent(max_history=20, use_long_term=False, use_knowledge=True)
    agent.load_knowledge("../bagu/qbank.md")
    run_eval(agent, agent.knowledge)
