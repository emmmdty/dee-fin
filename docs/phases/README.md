# Fin-EKG v4 · 阶段化执行手册（单会话自包含 phase）

> 把批准的 v4 四章计划拆成**互相隔离、单个 Claude Code / Codex 会话可完成**的 phase，避免长会话上下文互相污染。
> 权威设计见 [`../SPEC.md`](../SPEC.md)（§1 四章 + §5 防审稿创新点）与批准计划稿；实时状态见 [`../TODO.md`](../TODO.md)。

## 方法论（据官方最佳实践）

- **自包含契约（Spec-Driven Development）**：每个 phase 是一份"super-prompt"，**只读该 phase 文件 + 自动载入的
  `CLAUDE.md`/`AGENTS.md` + `SPEC.md` 即可执行，无需翻聊天记录或别的 phase**。
- **Codex 四要素**（每个 phase 文件的骨架）：**Goal**（要的结果、不是步骤）· **Context**（涉及文件/数据/前置产物）·
  **Constraints**（约定与安全）· **Done when**（可验证的结束态）。
- **Claude Code 纪律**：plan 优先；`CLAUDE.md` 开局入上下文并全程保留；**把每个会话当一次性的——进度落文件、频繁提交**；
  phase 之间 `/clear`；能 TDD 就 TDD。
- 参考：[Claude Code best practices](https://code.claude.com/docs/en/best-practices) ·
  [Codex best practices](https://developers.openai.com/codex/learn/best-practices) ·
  [Spec-driven development (GitHub)](https://github.blog/ai-and-ml/generative-ai/spec-driven-development-with-ai-get-started-with-a-new-open-source-toolkit/)。

## 每个 phase 文件的模板
`Goal / 依赖·产物 / Context(复用·新建) / 执行内容(Steps) / Constraints / 验收标准(Done when) / GPU / 达不到怎么办`

## 如何在新会话跑一个 phase

**Claude Code**：进仓库 → `/clear` → 输入：
> 读 `docs/phases/PHASE_X_*.md` 并执行。遵守 `CLAUDE.md` 硬约束与 `docs/SPEC.md` 设计。走 plan 模式先规划，能 TDD 就 TDD。完成后按该文件「验收标准」逐条自检，跑 `uv run pytest && uv run ruff check src tests scripts && uv run finekg-smoke`，把结果写进 `docs/TODO.md`。

**Codex**：仓库根有 `AGENTS.md`（与 `CLAUDE.md` 同步）→ 让 Codex 读 `docs/phases/PHASE_X_*.md`，用其 Goal/Context/Constraints/Done-when 执行，跑校验命令验收。

**收尾协议（每个 phase 结束都做）**：① 校验命令全绿（`pytest` 只增不改 / `ruff` 0 / `smoke` 绿）；② 结果落 `runs/`；
③ 更新 `docs/TODO.md`（如实：降就说降）；④ 需要提交时才提交（作者要求）。

## 阶段依赖图（关键路径 A→B→C→D→E）

```
P0 数据 ✅(FACT valid 待作者) ──┐
                                ▼
[A] Ch2 判别式抽取器 ──▶ [B] Ch2 一致性+修复+CRC ──┐
                                                    ▼
        [C] Ch1 规范节点 ──▶ [C2] Ch1 跨文档泛化    │
              │                                     ▼
              └──────────────▶ [D] Ch3 事实性+净化 ─▶ [E] Ch4 闭环+误差传播 ─▶ [F] 端到端预算
                                                         │
                                                         ├─▶ [G] 金融应用层
                                                         └─▶ [H] 稳健化(多种子+消融+新颖性扫) ─▶ [I] 写作
```

## 阶段索引（一句话 Goal + 验收 + 依赖）

| Phase | 章 | Goal（一句话） | 关键验收 | 依赖 | GPU |
|---|---|---|---|---|---|
| **A** | Ch2 | 判别式 `supervised` 关系抽取器，金标节点上 causal/subevent 可用 | causal F1 ≫0.4%（目标 ~30–37） | P0 | 重 |
| **B** | Ch2 | 全局一致解码 + 可追溯修复 + CRC 风控准入 | violation↓、分层 FNR、repair trace、ECG 可重建率↑ | A | 轻 |
| **C** | Ch1 | 证据+不确定性规范事件节点（含论元、难例判别） | 检测 F1 ~60+、coref MUC ~86、误合并率↓ | P0 | 轻 |
| **C2** | Ch1 | 跨文档泛化（ECB+/CLES） | 对比 SECURE/MEET/DIE-EC | C（+取数据） | 轻 |
| **D** | Ch3 | 构建图上事实性检测 + 事实性驱动图净化 | macro-F1 ≥47.6、预测图掉点量化、净化下游增益 | P0(+B) | 轻 |
| **E** | Ch4 | 下游门控闭环修复 + 三图误差传播（headline） | repaired>predicted、误差传播曲线、下游门控接受 | A·B·D | 重 |
| **F** | 跨章 | 端到端误差预算（union bound+可达性，标注前提） | 端到端界 + 分层 FNR、naive vs 预算对照 | B·D·E | 轻 |
| **G** | 应用 | 金融构建→预测可迁移（CCKS-FinCausal + SARGE） | 金融案例、可迁移性 | A·B | 中 |
| **H** | 全篇 | 多种子 13/17/42 + 消融补齐 + 投稿前新颖性扫 | 主表 mean±std、Ch2 改名定稿 | A–F | 重 |
| **I** | — | 论文写作（非代码 phase） | 初稿 + 终辩 | 全部 | — |

> 后段（C2/F/G/H）细节随前段真实结果再细化——文件先给 Goal/Steps 骨架，执行前按当时产物补全。
