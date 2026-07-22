# Fin-EKG — 可验证的事件/事理图谱构建与推理

从文档级事件抽取，到事件关系/图谱构建，再到事件因果图上的推理与预测；**每个节点、边、预测都可
回指原文证据**（evidence-grounded / verifiable）。

> 主线 = **通用事件/事理图谱的可验证自动化构建与应用**。学位论文 3 章脊柱（extraction → relations
> → reasoning），当前主任务 = **CGEP**（事件因果图上的后继事件预测，基座 **SeDGPL**）。
>
> **完整设计与开发驱动 → [`docs/SPEC.md`](docs/SPEC.md)｜实时状态 → [`docs/TODO.md`](docs/TODO.md)。**

## 方法论

系统由**专精 agent 提议 + 验证器把关**组织，验证器有三重身份：

> **门控（gate，推理时）· 奖励（reward，训练时）· 风险控制器（risk-controller，带有限样本保证）。**

贯穿属性（论文身份）= 可验证性：每条边带 `evidence`、每个预测带 `evidence_chain`。3 章脊柱：

| 阶段 | 任务 | 基准 | 代码域 |
|---|---|---|---|
| 事件抽取（上游） | 文档级事件检测 | SARGE（`external/sarge/` subtree） | 经契约 `core.io.event_nodes_from_sarge` 消费 |
| 事件关系与图谱 | coref/temporal/causal/subevent | MAVEN-ERE | `finekg.relations` |
| **事件图推理（主任务=CGEP）** | 后继事件预测 | CGEP-MAVEN / ESC（基座 SeDGPL） | `finekg.succession` |

## 设计原则（为升级而设计，避免返工）

1. **冻结的跨阶段契约**（`finekg.core.schema`）：`EventNode → RelationEdge / EventGraph → TemporalQuad
   / ForecastQuery → Prediction`。只加可选字段、不复用既有字段。
2. **按功能域组织、不按章节命名**：包名/函数名不含 `ch1/ch2/ch3`。
3. **插件式 registry**（`finekg.core.registry`）：换方法只加实现、不动接口。
4. **CPU/GPU 惰性分层**：`core` + 启发式 baseline 无 torch，本地跑通；神经代码 lazy import，服务器训练。

## 目录结构

```
src/finekg/
├── core/         冻结契约: schema·io·graph·registry·config·calibration·eval
├── succession/   ★CGEP(后继事件预测): data/(cgep·esc)·linearize·model·encode·sedgpl·metrics·predictor
├── relations/    事件关系抽取 + 图构建 (+ CRC 边准入 admission)
├── agents/       多智能体基底 (Agent/Blackboard/Orchestrator/Verifier)
└── rl/           RL 基底 (组合奖励·组相对优势·势塑形·课程)
scripts/  configs/  tests/  data/fixtures/  docs/
```

## 快速开始

```bash
# 本地（无 GPU）：契约 / 评测 / 启发式 baseline / 冒烟
uv sync --extra dev
uv run pytest          # 全绿
uv run ruff check src tests scripts
uv run finekg-smoke

# 服务器（CUDA 12.4）：装 GPU 栈做真正训练（运维见 docs/GPU_RUNBOOK.md / AGENTS.md）
uv sync --extra dev --extra llm --extra gnn --extra serve
```

## 数据 / 许可

公开数据集不入库；`data/fixtures/` 仅放极小样例，全量由 `scripts/` 在服务器拉取。来源与切分合规见
[`docs/DATASETS.md`](docs/DATASETS.md)。工程坑见 [`docs/ENGINEERING_NOTES.md`](docs/ENGINEERING_NOTES.md)。

许可：MIT
