# SFT 与提示模块风险分析

> 更新时间：2026-05-23 13:09 CST  
> 本地项目根：`/home/tjk/myProjects/masterProjects/DEE/SARGE/`  
> 服务器项目根：`/data/TJK/DEE/SARGE/`  
> 本地 Python：`/home/tjk/miniconda3/envs/feg-dev-py310/bin/python`  
> 服务器 Python：`/data/TJK/envs/sarge_vllm_full/bin/python`

## 结论

LoRA SFT 本身不是算法创新。当前可以站住的主线是：将中文金融文档级事件抽取构造成 schema-grounded instruction SFT，并让训练目标、推理 prompt、解析器和 Legacy-FS 评价口径对齐。

Surface Memory 和 Slot Plan 不能作为当前论文的稳定性能贡献点。Surface Memory 在 DuEE-Fin HF 主后端单变量消融中没有可见正向贡献；Slot Plan 只有弱正向证据，且 ChFinAnn vLLM 消融中去掉 Slot Plan 反而略高。它们最多作为工程辅助或附录诊断，不应写成主方法创新。

## SFT 数据构造

训练数据不是把原始标注直接喂给模型，而是经过以下链条构造：

1. `stage_dataset()` 将不同数据集格式转换成统一 SARGE 视图：`doc_id / dataset / split / content / events`，并把 schema 转成 `event_types: [{event_type, roles}]`。
2. 每个训练文档构造成一个 instruction-output 样本。`prompt` 包含数据集、schema、文档正文、surface candidates 和输出约束；`output` 由 gold events 转成 canonical JSON event table。
3. 默认输出格式为 `minimal_text`：`{"events":[{"event_type":"...","arguments":{"角色名":["..."]}}]}`。
4. 训练时 prompt tokens 被 mask 为 `-100`，只对 assistant answer 部分计算 loss。

对应代码：

- `src/sarge/data/staging.py::stage_dataset`
- `src/sarge/models/sft_dataset.py::build_getm_sft_sample`
- `src/sarge/models/sft_dataset.py::gold_events_to_getm_output`
- `src/sarge/models/sft_dataset.py::build_sft_training_examples`

因此，论文里不要写“提出 LoRA SFT”。更准确的表述是：使用 LoRA 作为参数高效适配手段，创新边界在 schema-grounded event-table instruction tuning 的任务接口设计。

## Prompt 设计

当前 role-safe prompt 的有效设计点是：

- 从 schema 渲染 event type 和 role，不让模型自由发明标签。
- 要求输出单个 JSON object，禁止 markdown、解释性文字、重复文档或 schema。
- 要求 role 必须是当前 event type 的合法角色。
- 禁止翻译、折叠、别名化或规范化 schema label。
- 可使用 `{"events":` response prefix 降低 JSON 起始格式漂移。

这些设计支持“schema-grounded / role-safe JSON contract”的方法叙事。

## Slot Plan 为什么没有真正训练

当前 SFT 训练样本中 `slot_plan=None`。训练脚本只构造了 surface memory，未把 slot plan 放进 SFT prompt。

推理时的 Slot Plan 来自 `TrainPriorPlanner`：它用 train split 的 gold events 统计事件类型出现率、正例 record count mode 和 role prior，然后对 predict 文档生成一个 train-prior plan。这不是文档级 learned planner，也不是被 SFT 学会使用的计划。

这带来训练-推理不一致：模型训练时没有学过如何消费 slot plan，推理时突然看到 slot plan，可能忽略，也可能被误导。现有消融结果与此一致。

## 数据泄露检查

从当前主路径代码看，未发现 test/dev gold 泄露到模型推理 prompt 的证据：

- predict 文档以 `mode="predict"` 加载，`gold=None`。
- prompt builder 会拒绝 `gold`、`events`、`events_gold`、`raw_annotations`、`arguments`、`norm_text`、`event_id` 等 gold-visible key。
- Qwen/vLLM backend 在预测前递归拒绝 gold-visible 字段。
- Slot Plan fit 使用 train docs；predict 阶段要求目标文档不能暴露 gold。

需要注意：evaluation 阶段会按 predicted doc_ids 过滤 gold，这是 subset/run-level evaluator 对齐，不是模型输入。主表只使用 Legacy-FS / `legacy_doc2edag`，Unified-Strict、DocFEE 和 ExactRec 只作为诊断，不能混入主表。

## 消融证据

DuEE-Fin HF-4bit 主后端：

| Profile | Legacy-FS F1 | 结论 |
|---|---:|---|
| full | 0.7796 | 当前 DuEE-Fin 主路径 |
| no_surface_memory | 0.7812 | 去掉 Surface Memory 未造成可见下降 |
| no_slot_plan | 0.7758 | 仅小幅下降约 0.38pp，Slot Plan 只有弱正向证据 |

ChFinAnn vLLM BF16, `gpu_memory_utilization=0.80`：

| Profile | Legacy-FS F1 | P | R | Single F1 | Multi F1 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| full | 0.8547 | 0.8426 | 0.8671 | 0.8994 | 0.8064 | vLLM full control |
| no_surface_memory | 0.8538 | 0.8483 | 0.8595 | 0.9035 | 0.7999 | 精度升、召回降，净 F1 基本持平 |
| no_slot_plan | 0.8567 | 0.8403 | 0.8739 | 0.8963 | 0.8139 | 召回和 multi-event F1 更高，去掉 Slot Plan 略优 |

解释：

- Surface Memory 只是提示候选，不是硬约束。金融公告中金额、日期、机构名等高频重复，候选列表容易同时增加 TP 和 FP。
- Slot Plan 当前是 train-prior 弱提示，可能过度约束召回，尤其影响多事件展开。
- SFT 已经学到了大部分 schema/role 输出结构，两个提示模块的边际信息量不足。
- 语法合法性已不是主要瓶颈；当前瓶颈是精确值选择、record assignment 和 recall/precision 权衡。

## 论文处理建议

主方法建议收缩为：

> schema-grounded instruction SFT + role-safe JSON contract + canonical export/evaluation.

Surface Memory 可以作为浅层候选 grounding 的工程接口描述，但不要声称它稳定提升 F1。Slot Plan 不应作为核心贡献；如果保留，只能作为 train-prior auxiliary prompt 或负/中性消融分析。

如果未来要把 Slot Plan 做成真正创新，需要重新设计实验：

- gold-derived plan target 只用于 train supervision；
- dev/test 必须使用 predicted plan，不能使用 oracle plan 进主表；
- 报告 no-plan、weak train-prior plan、predicted plan、oracle-plan upper bound；
- 同时报告 plan accuracy、record count accuracy、multi-event F1、role omission/FP/FN。

在当前论文周期内，不建议为挽救 Slot Plan 临时加入训练计划。
