# 评价协议选择的论文口径说明

本文说明为什么本项目同时保留历史 native fixed-slot 指标和 `unified-strict` canonical role-value 指标，以及为什么二者不能混入同一个 leaderboard 列。

## 1. 历史指标的必要性

Doc2EDAG 和 ProcNet 的 native 指标服务于历史 baseline 复现。它们的输入、解码结构和评价结构都围绕固定、有序的事件角色槽展开：一个事件类型对应一组固定角色，模型输出或转换结果被放入这些角色槽后再计算 precision、recall 和 F1。

因此，native fixed-slot 指标是必要的。它回答的问题是：在原论文或原代码约定的评价协议下，当前复现结果与历史 baseline 是否可比。对于复现 Doc2EDAG/ProcNet 等历史系统，这类指标应保留，并适合放在 native reproduction、historical baseline 或 appendix-style 表格中。

## 2. Fixed-Slot 指标的表示边界

Fixed-slot 协议的核心边界是：一个 `(event_type, record, role_slot)` 最多自然承载一个 argument。这个设计对许多历史金融事件抽取基准是合理的，因为它把任务转化为固定槽位填充，便于训练、解码和与旧结果对齐。

但真实公告中可能出现同一事件记录内同一角色对应多个取值。例如本项目审计到 DuEE-Fin-dev500 test split 中：

- fixed-slot non-empty schema-role slots：7533
- canonical unique role-value units：7907
- multi-value extra units：374
- multi-value role occurrences：264
- documents with multi-value role：219

这些额外 units 不是语义推断产生的，而是由原始标注中同一角色下多个不同表面值产生的。典型例子包括：

- `中标 / 中标公司`：华润、国药器械、国药控股、海王
- `股份回购 / 每股交易价格`：3.57元、6.1297元
- `质押 / 质押方`：邵健伟、邵健锋

因此 fixed-slot 指标并不是“错误”的指标，而是服务于另一种历史协议。它在多值角色场景下可能低估或折叠真实 role-value 信息。

## 3. 为什么需要 `unified-strict`

本项目需要跨 ChFinAnn、DuEE-Fin 和 DocFEE 做科学比较。跨数据集比较要求输出空间尽可能统一，而不是把不同历史系统的 native 评价协议直接相互替代。

Canonical JSONL 允许把角色值表达为 `role -> list/set of values`。在这种表示下，一个角色可以包含多个值；评价时可以把基本单位定义为 canonical role-value unit，即 `(event_type, record, role, normalized value)`。`unified-strict` 在 event-type-constrained record matching 后，对这些 canonical role-value units 进行严格表面匹配。

这种协议适合回答不同的问题：在统一 canonical output space 下，不同模型或数据集的角色-取值抽取能力如何比较。它不替代 Doc2EDAG/ProcNet native 指标，也不否定历史 fixed-slot 结果；它补充的是跨数据集、跨系统时更一致的科学评价口径。

## 4. 表格与报告政策

论文或报告中应明确分列：

- native fixed-slot scores：用于历史 baseline comparison / native reproduction。
- canonical JSONL exported evaluation：说明模型输出已转换到统一 canonical schema，可作为 `unified-strict` 的输入。
- `unified-strict` scores：用于跨 ChFinAnn、DuEE-Fin、DocFEE 的统一科学比较。

ProcNet native training reports 不应与 exported canonical JSONL 的 `unified-strict` 分数混入同一个 leaderboard 列。二者可以并排展示，但列名、caption 和正文必须说明它们回答不同问题：

- native fixed-slot 列回答“是否复现历史协议下的 baseline”。
- `unified-strict` 列回答“在统一 canonical role-value 协议下的跨数据集/跨系统表现”。

## 5. 非目标

`unified-strict` 的 main metric 不做以下事情：

- 不做 semantic matching。
- 不做 embedding matching。
- 不使用 LLM judge。
- 不做 alias expansion。
- 不做金额或日期推理。
- 不把历史 native fixed-slot 指标改写成 canonical 指标。

主指标只使用确定性表面归一化：Unicode NFKC、去除首尾空白、折叠连续空白。这样做的目的不是追求语义宽松，而是保持跨数据集评价协议可复现、可审计、可解释。

## 6. 论文口径结论

本项目应同时报告 native fixed-slot 与 `unified-strict`，但必须分列。Native 指标保留历史可比性；`unified-strict` 保证 canonical role-value 层面的跨数据集科学可比性。多值角色审计表明，至少在 DuEE-Fin-dev500 中，fixed-slot 表示会遇到真实多值角色边界，因此仅报告历史 fixed-slot 指标不足以完整刻画 canonical role-value 抽取质量。
