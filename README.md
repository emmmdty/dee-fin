# EKG — 可信事件图谱构建与可靠应用

EKG 从文本构建可验证的事件图谱，并研究构建误差如何影响下游推理。节点、关系边、修复动作和
预测都应能回指原文证据或明确的计算轨迹。

> 当前唯一主线是 **v4（2026-07-21）**：围绕事件图谱的身份可信、结构可信、事实可信和传播可信，
> 形成四章递进研究。设计权威见 [`docs/SPEC.md`](docs/SPEC.md)，实时状态见
> [`docs/TODO.md`](docs/TODO.md)。历史三章路线和旧 TKG 实验只作归档证据，不再驱动主干开发。

## v4 四章脊柱

| 章 | 可信维度 | 研究任务 | 主数据 | 当前代码域 |
|---|---|---|---|---|
| Ch1 | 身份可信 | 证据与不确定性感知的规范事件节点 | MAVEN、MAVEN-Arg、MAVEN-ERE coref | `ekg.core` + 待建节点规范化模块 |
| Ch2 | 结构可信 | 风险受控的多关系边、全局一致性与可追溯修复 | MAVEN-ERE | `ekg.relations` |
| Ch3 | 事实可信 | 构建图上的事件事实性检测与图净化 | MAVEN-FACT | 待建 factuality 模块 + `core.calibration` |
| Ch4 | 传播可信/可用 | 构建误差传播、下游门控闭环修复与可靠后继预测 | CGEP-MAVEN、ESC | `ekg.succession` + `ekg.agents` |

全篇 headline 是 **面向下游的构建误差预算 + 下游验证的闭环修复**：只在后继预测目标改善时接受
图编辑。既有 SeDGPL、M1/M2、选择性 conformal 和受控 cross-stage 扫描是 Ch4 的可靠性模块，
不是独立主线。

SARGE（中文金融文档级抽取）已于 **2026-07-27 移出主干**——v4 四章无一依赖它，留在仓内只增加
检索与上下文噪声；结果快照与其余历史留档已移出仓库，索引见
[`docs/ARCHIVE_INDEX.md`](docs/ARCHIVE_INDEX.md)，源仓独立维护。
旧实体中心金融 TKG、RE-GCN、Path-RL 和 hybrid 同样已移出主干，保存在 tag `frozen-tkg-line`。

## 当前执行位置

- P0：MAVEN-ERE / Arg / FACT 主干数据就位；扩展数据状态见
  [`docs/DATASET_SURVEY.md`](docs/DATASET_SURVEY.md)。
- Phase A ✅ 已达标（2026-07-24）：判别式 `supervised` 关系抽取器把 causal 召回 0.4%→67.5%，
  causal F1 .250 / subevent .213 / temporal .338。
- 当前关键路径：Phase B（全局一致解码 + 可追溯修复 + CRC 风控准入），代码已 CPU 全绿，
  真实 predicted 图 dump 待 GPU 空闲。
- 后续依赖：A → B（修复与风控）→ C（规范节点）→ D（事实性）→ E（闭环）。
- 阶段验收与止损条件见 [`docs/phases/`](docs/phases/README.md)。

## 工程原则

1. 冻结跨阶段契约：`EventNode → RelationEdge / EventGraph → CgepInstance → Prediction`；
   `EventNode` 扩展统一放入 `metadata`。
2. 按功能域组织代码，包名和函数名不使用章节编号。
3. 可替换组件走 registry；GPU 依赖 lazy import，并提供 CPU 测试或缓存回放。
4. 报告真实结果：负结果、工具失败和未完成实验必须明确区分。

## 目录

```text
src/ekg/
├── core/         schema、I/O、图算法、registry、calibration、通用评测
├── relations/    候选对、判别式/LLM/启发式抽取、grounding、一致性与修复、CRC 准入、旧 GRPO 基线
├── succession/   CGEP 数据、SeDGPL、风险感知线性化、结构编码、选择性预测、cross-stage、ECG 可重建率
├── agents/       阶段无关的编排与黑板协议
└── rl/           旧 GRPO 基线复用的通用奖励/课程原语
```

## 本地验证

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests scripts
uv run ekg-smoke
```

GPU 只在 `gpu-4090` 执行；当前可运行任务、同步方法和启动约束见
[`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md)。数据不提交 Git，只有极小 fixture 和数据溯源文档入库。

许可：MIT
