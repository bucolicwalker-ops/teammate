# Stage 4.1 — 角色与接口设计

> 参照 cat-cafe（F032 Agent Plugin Architecture + F105 opencode 接入）+ Claude Code Subagents 文档。
> 设计 teammate Stage 4 多 Agent 协作的角色 taxonomy、AgentBase I/O schema、停止条件。
>
> 宪宪/布偶猫 🐾 · 2026-07-28

---

## 1. 三层身份模型（参照 cat-cafe F032）

cat-cafe 把"身份"拆成三个独立维度，避免硬编码导致的系统不可扩展：

| 维度 | cat-cafe 例 | teammate 对应 | 含义 |
|---|---|---|---|
| **Family**（物种/厂商） | ragdoll / maine-coon / siamese | `myagent` / `codex` / `claude-code` | 模型家族 / 接入通道 |
| **Individual**（个体） | opus-45 / opus-46 / codex / gemini | `myagent-1` / `codex-1` / `cc-1` | 具体一个 agent 实例 |
| **Role**（职能） | architect / reviewer / designer / security | planner / executor / reviewer / critic / router | 干什么活 |

**关键洞察**（来自 F032 Problem Statement）：
> "身份 = 物种 = 个体" 是 cat-cafe 早期最大的痛——三个概念糊在一起，多分身并存后所有 reviewer 配对规则失效。

teammate 现状犯了一样的错：`team.py` 里 `researcher/writer/reviewer` 三个 MyAgent 实例，把 role 和 individual 绑死——同模型 in-process，没有 family 维度，router 没法基于能力路由。

**Stage 4 重构方向**：把这三个维度拆开，让 router 基于角色 + 能力 + 可用性做派发，而不是写死"researcher→writer→reviewer"。

---

## 2. 5 角色定义（Stage 4 checklist item 1）

| 角色 | 职责 | 输入 | 输出 | 停止条件 | cat-cafe 对应 | Claude Code 对应 |
|---|---|---|---|---|---|---|
| **Planner** | 任务分解 + 产出 step 列表 | 用户原始任务 + roster | 有序 step plan（每步标 executor 能力要求） | 所有 step 已产出 / 触顶 max_steps | operator/co-creator + architect cat | Plan built-in subagent（read-only 研究） |
| **Executor** | 执行单个 step（调工具 / 写代码） | step 描述 + 上下文 | step 结果 + artifacts | step 完成 / 失败 / 超时 | 任何 cat 在 executor role | General-purpose subagent |
| **Reviewer** | 检查 output 是否符合 spec | output + spec | verdict（通过/退回）+ feedback | 给出 verdict | reviewer role cat（缅因猫） | (custom) code-reviewer subagent |
| **Critic** | 挑战 spec / plan 本身，找遗漏和假设 | plan or output | challenge list + gaps | 提出全部质疑清单 | 平行于 Maine Coon 的 pushback 协议 | (无直接对应，是 cat-cafe 独有) |
| **Router** | 决定下一步交给谁 | context + roster | next agent_id 或 "done" | 返回 "done" | cat-cafe 的 `_route()` / 球权传递 | Claude parent 用 subagent `description` 委派 |

### 角色边界原则（来自 cat-cafe 家规）

- **一机制一件事**：planner 不执行，executor 不路由，reviewer 不重写。
- **路由不让 agent 随便聊天**（Stage 4 checklist item 2）：router 是唯一的球权持有者，其他 agent 之间不直接对话，必须经 router。
- **Critic ≠ Reviewer**：reviewer 检查"output 对不对"，critic 挑战"spec/plan 该不该这么定"。两者并存防止"战术勤劳掩饰战略懒惰"。

### 单 Agent 退化判断（Stage 4 checklist item 5）

Router 第一轮就能判断：
- 任务 ≤ 1 step → 直接派给 executor，跳过 planner
- 任务无外部依赖 + 用户已给清晰 spec → 直接派给 executor
- 任务是单点查询 → 跳过 planner + reviewer，单 executor + critic

cat-cafe 对应："能自决吗？"决策漏斗——可逆 + 不影响外部 + 不碰硬排除 → 直接做，不预先 @。

---

## 3. AgentBase I/O Schema 设计

当前 `src/agents/base.py` 太薄（只有 ask/research/close），不够支撑多 Agent 协作。设计扩展（**4.1 只设计，4.2 实现**）：

```python
# src/agents/base.py — Stage 4 设计（4.2 实现）

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

# ---- 三层身份 ----
@dataclass
class AgentIdentity:
    """参照 cat-cafe F032：三层身份独立，避免硬编码。"""
    individual_id: str        # e.g. "myagent-1", "codex-1", "cc-1"
    family: str               # e.g. "myagent", "codex", "claude-code"
    roles: list[str]          # e.g. ["planner", "executor"]
    lead: bool = False        # 是否 family 内的 lead（参照 cat-cafe roster.lead）
    available: bool = True    # 当前是否可派发
    description: str = ""     # 给 router 的委派提示（参照 CC subagent description）


# ---- 结构化任务（替代 free-form ask）----
@dataclass
class Task:
    """多 Agent 协作的最小单元——结构化输入输出。"""
    task_id: str
    step_description: str           # 做什么
    context: str = ""               # 上游结果 / 用户原始问题
    required_roles: list[str] = field(default_factory=list)  # router 派发用
    max_steps: int = 12              # 该 step 的 agent loop 上限
    spec: Optional[str] = None       # reviewer 检查依据（None = 无 spec）


@dataclass
class TaskResult:
    task_id: str
    output: str                     # 主产出（文本）
    artifacts: list[str] = field(default_factory=list)  # 文件路径等
    status: str = "done"            # done / failed / partial / blocked
    error: Optional[str] = None
    trace: Optional[dict] = None   # tracer 输出（token/延迟/工具调用）


# ---- Agent 接口 ----
class AgentBase(ABC):
    """所有 Agent（MyAgent / CodexAgent / ClaudeCodeAgent）实现此接口。

    隔离三层（参照 base.py 现有注释 + F105 L1 Adapter）：
    - 进程隔离：MyAgent in-process；Codex/CC 子进程
    - Context 隔离：各 Agent 独立 history + system prompt
    - 工具隔离：各 Agent 受限工具集
    """

    @abstractmethod
    def identity(self) -> AgentIdentity:
        """三层身份 + 角色声明。Router 用 roles + description 派发。"""
        ...

    @abstractmethod
    def execute(self, task: Task) -> TaskResult:
        """结构化任务执行（4.2 起替代 ask/research 成为多 Agent 主入口）。

        实现约束：
        - 不直接路由其他 agent（router 才有球权）
        - 不写共享状态（任务追踪走 router 转发）
        - 必须返回 TaskResult（含 status，让 router 决定下一步）
        """
        ...

    @abstractmethod
    def close(self):
        """清理资源（MCP 连接 / CLI 子进程 / 数据库）。"""
        ...

    # ---- legacy 接口（保留兼容 team.py 现状，4.3 重构掉）----
    @abstractmethod
    def ask(self, user_msg: str, max_steps: int = 5) -> str:
        """基础对话。Stage 4 后只用于调试 / 单 Agent 模式。"""
        ...

    @abstractmethod
    def research(self, topic: str, max_steps: int = 12) -> str:
        """研究模式。Stage 4 后由 planner→executor 链替代。"""
        ...
```

### 设计决策理由

| 决策 | 替代方案 | 选这个的理由 |
|---|---|---|
| 结构化 `Task` 替代 free-form `ask` | 继续用 str in/str out | 多 Agent 协作需要明确 input/output schema + status，free-form 让 router 没法判断"完成了吗" |
| `identity()` 返回三层身份 | 只用 individual_id | cat-cafe F032 教训：硬编码 identity 在多分身后失效 |
| `execute()` 不让 agent 直接路由 | 让 agent 可以互相 @ | Stage 4 checklist item 2："不让 agent 随便聊天"——router 独占球权 |
| `required_roles` 在 Task 里 | router 自己猜 | 让 planner 显式声明"这步需要 reviewer"，router 不用 LLM 推理 |
| 保留 legacy `ask/research` | 直接砍掉 | team.py 现在还在用，4.3 重构 router 后再删 |

---

## 4. Team Roster（参照 cat-cafe F032 Phase B1）

cat-cafe 把所有 cat 的身份信息放 `cat-config.json` 的 `roster` 字段——single source of truth。teammate 4.2/4.3 加一个 `team_roster.json`：

```jsonc
{
  "version": 1,
  "agents": {
    "myagent-1": {
      "family": "myagent",
      "roles": ["planner", "executor", "reviewer", "critic"],
      "lead": true,
      "available": true,
      "description": "in-process GLM-5.2，全能但慢。能做规划/执行/审查/质疑。",
      "model": "glm-5.2"
    },
    "codex-1": {
      "family": "codex",
      "roles": ["executor", "reviewer"],
      "lead": false,
      "available": true,
      "description": "Codex CLI 子进程，coding 强，review 客观（跨 family）。",
      "model": "gpt-5.2-codex"
    },
    "cc-1": {
      "family": "claude-code",
      "roles": ["executor", "reviewer"],
      "lead": false,
      "available": true,
      "description": "Claude Code CLI 子进程，工具丰富，长上下文。",
      "model": "claude-sonnet-4"
    }
  }
}
```

**Router 派发算法**（4.3 实现）：
1. 收到 task → 看 `required_roles`
2. 从 roster 过滤 `available=true && role in required_roles` 的候选
3. 优先级：lead > 非 lead；跨 family 优先（参照 cat-cafe "Review 必须跨个体" 铁律）
4. 候选为空 → router 自己降级做（单 Agent 退化）

---

## 5. 停止条件总表（Stage 4 checklist item 3）

| 层级 | 停止条件 | 触发动作 |
|---|---|---|
| **单 step 内** | agent loop `stop_reason != tool_use` 或 `step >= max_steps` | 返回 TaskResult(status=done/failed) |
| **Task 级** | executor 返回 done / failed | router 决定下一步（review / 重试 / 跳过） |
| **Pipeline 级** | router 返回 "done" 或触顶 `max_rounds` | 结束，返回最终结果 |
| **退化** | 任务 ≤1 step 或无外部依赖 | router 跳过 planner，直接派 executor |
| **死循环防护** | 同一 agent 连续 2 轮返回相同 failed | router 强制换 agent 或退化为单 Agent |
| **上下文膨胀** | context > N tokens | router 触发 compact（参照 teammate s08 snip+LLM summary） |

---

## 6. 4.2 实现预告

下一 session 做：
1. 实现 `AgentIdentity` / `Task` / `TaskResult` dataclass
2. 改 `src/agents/base.py` 加 `identity()` + `execute(task)` 抽象方法
3. 让 `MyAgent` 实现新接口（保留 `ask/research` 作为 legacy）
4. 写 `team_roster.json` + `Roster` 加载器
5. 提交 → **@砚砚 跨族 review**（家规铁律 2）

4.3 才动 router + thread store + @mention parser（最大块，参照 F043 协作工具补全）。

---

## 7. 参考证据

- `cat-coffee/docs/features/F032-agent-plugin-architecture.md` — 三层身份模型 + Team Roster + Reviewer Matcher
- `cat-coffee/docs/features/F043-mcp-unification.md` — cat-cafe 协作工具（@mention / hold_ball / cross_post / multi_mention / create_task）
- `cat-coffee/docs/features/F105-opencode-golden-chinchilla.md` — L1 CLI Adapter 模式（外部 agent 接入）
- Claude Code Subagents 文档 — `description + tools + model` 委派模式 + 5 种 built-in subagent
- teammate `src/agents/base.py` (e1161f2) — 现状接口
- teammate `src/team.py` — 现状 TeamSequential + TeamSupervisor（4.3 重构对象）
