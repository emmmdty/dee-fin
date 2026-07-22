# Phase A — Ch2 判别式监督关系抽取器（关键路径起点）

> 单会话自包含契约。执行前：`CLAUDE.md`/`AGENTS.md`（硬约束+环境+校验命令）已自动入上下文；读 `docs/SPEC.md` §1/§4 拿设计。
> 只读本文件即可执行。完成后按「验收标准」自检并更新 `docs/TODO.md`。

## Goal（完成目标）
注册一个判别式 `@register("supervised")` 关系抽取器，在**金标事件节点**上对 MAVEN-ERE 的 temporal/causal/subevent
关系做成对分类，把 **causal 召回从当前 0.4%（3/810）提到文献可比区间（F1 ~30–37）、subevent ~30**。这是 Ch3/Ch4
真实图的前提，是全篇瓶颈。

## 依赖 / 产物
- 前置：P0（数据就位，`data/raw/maven_ere/{train,valid}.jsonl` 已在库）。
- 产出：`src/finekg/relations/extractor/supervised.py`（注册项）+ 测试 + `runs/relations/supervised_*.json`。

## Context（复用 / 新建）
- **复用**：`relations/extractor/heuristic.py`（`@register("heuristic")` 与接口范式）、`relations/extractor/llm.py`
  （`@register("llm")` 生成式注册范式）、`relations/pairs.py`（候选 pair 对齐尺子，用于构候选宇宙）、
  `relations/data/maven_ere.py`（gold events + 关系加载）、`core/schema.py`（`RelationEdge` 输出类型）、`core/registry`。
- **新建**：`relations/extractor/supervised.py` = PLM 事件对编码（标记触发词 `<e>`）+ 逐关系类分类头。
- **接线**：`scripts/evaluate_relations.py` 加 `--extractor supervised`，报每类 P/R/F1。
- **参考坑**：当前 0.4% 来自生成式 SFT+GRPO（`relations/rl/`），**不是**判别式；本 phase 用判别式打底。

## 执行内容（Steps · TDD）
1. **测试先行**：① 从金标节点构**文档级候选对**（窗口内全配对，标签取自 gold 关系；负例=无关系对）；② 抽取器
   注册 + 接口 `extract(doc) -> list[RelationEdge]`，CPU 可导入、torch lazy。
2. **实现** `supervised.py`：RoBERTa 编码事件对（marker 插入），三组头（causal: CAUSE/PRECONDITION；subevent；
   temporal: BEFORE/…）。**必须处理类不平衡**（causal/subevent 极稀疏）：加权 CE / focal / 负例下采样，三选一并消融。
3. **接** `scripts/evaluate_relations.py --extractor supervised`，在 valid 报 per-relation P/R/F1、doc-macro-F1。
4. **GPU 训练**：选卡前 `nvidia-smi`，优先 card 1；`screen -dmS` + `python -u` 重定向 `logs/`。

## Constraints
- 遵守 `CLAUDE.md` 硬约束（名字不含 `ch1/2/3`；registry + lazy import；`EventNode` schema 冻结；CPU 可导入/GPU lazy；如实报数）。
- 只在**金标节点**上评测（解耦 Ch1）；候选须**文档级**（非仅相邻句），否则召回天花板低。

## 验收标准（Done when）
- [ ] `@register("supervised")` 存在 + 测试；`uv run pytest` 只增不改绿、`ruff` 0、`finekg-smoke` 绿。
- [ ] `evaluate_relations.py --extractor supervised` 输出三类 F1；**causal F1 ≥ ~25（目标 30–37）、subevent ≥ ~20**，
      **如实记录并与 0.4% 对比**（降/未达也照报）。
- [ ] 结果落 `runs/relations/supervised_*.json` + 写入 `docs/TODO.md`。

## GPU
需要（训练）。约 1 分钟才占显存；长训练用 `screen/nohup`。

## 达不到怎么办（止损）
causal F1 <10% → 排查类不平衡/候选窗口/编码方式；仍不行则**维持受控模拟**（`succession/cross_stage.py`），Ch2 贡献
收缩到一致性/修复/风控（Phase B 仍成立），Ch4 退受控扰动版。
