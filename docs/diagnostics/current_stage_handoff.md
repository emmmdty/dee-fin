# 当前阶段诊断交接摘要

> 日期：2026-05-13  
> 范围：CARVE v1.3 当前阶段性状态交接。  
> 结论边界：本文只总结已有仓库证据，不构成 hidden-test、final-test、SOTA 或完整 CARVE 实现报告。

## 已完成

- P0 文档冻结已完成：CARVE v1.3 被定位为冻结 proposal，`data/`、`baseline/`、method claim、phase gate、服务器路径和验证命令边界已写入仓库规则。
- P1 显存测量已完成：`gpu-4090` 上三个本地 safetensors encoder forward 测量均为 `ok`，当前证据见 `docs/measurements/p1_memory.md`。
- P4 toy allocation gate 已完成：多正例分配、NULL 列、确定性 record ordering、训练期 oracle injection、推理期禁用 oracle injection、`L_alloc` 与 share gate toy 行为通过。
- P5a toy EDAG gate 已完成：在受控错配样例中，allocation-aware route 能把 baseline 的 wrong-record 绑定改为 correct-record。
- R5b / P5b DuEE-Fin first long run 已完成：`DuEE-Fin-dev500` dev diagnostic 跑通，结果记录在 `docs/measurements/p5b_duee_fin_dev500_seed42.md`。

## 未完成

- 完整 P5b 三数据集机制门控未完成：ChFinAnn、DocFEE 行没有运行，`docs/measurements/p5b_decision_table.md` 不应创建或提升。
- P2、P3、P6、P7、P8、P9 未展开：根据 phase gate，除非 P5b 机制诊断支持继续，否则不应推进后续完整 CARVE 路线。
- 完整 CARVE encoder、EDAG decoder、Qwen verifier、hidden-test/final-test evaluation 均未完成。
- 当前 runner 未输出完整 P5b 指标族，尤其是 `misallocated_rate_eligible`、`misallocated_rate_total`、`ambiguous_excluded_rate` 和 inference candidate recall。

## 当前主要问题

- R5b DuEE-Fin dev diagnostic 不支持当前机制：baseline F1 为 `0.037311`，CARVE F1 为 `0.010883`，F1 delta 为 `-0.026428`，runner 标记 `No support`。
- CARVE route recall 略升，但 precision 明显崩坏：FP 从 baseline 的 `8,599` 增至 CARVE 的 `39,098`，说明当前 allocation/share diagnostic 更像过度分配或过度预测。
- P4/P5a 的 toy 成功不能外推到真实 dev：toy 只能证明机制形式和受控样例方向正确，不能证明真实数据有效。
- 当前 P5b 证据只覆盖 DuEE-Fin dev diagnostic，不能支撑 EMNLP-main、SOTA、final-test 或三数据集主表叙述。

## 可能的解决措施

- 先做 R5b failure analysis，不直接进入后续大阶段：
  - 分解 FP 来源：按 event type、role、candidate source、record index、share gate 输出统计。
  - 审计候选生成：区分 train-lexicon 命中与 role-regex 命中，检查是否产生大量低精度候选。
  - 审计 record-count estimate：确认过高的记录数估计是否导致候选被扩散到过多 record。
  - 校准 share gate 和 allocation 阈值：先用 dev diagnostic 做分析，不把调参结果写成 final claim。
  - 补齐 P5b 指标族：至少补出 `misallocated_rate_eligible`、candidate recall、ambiguous exclusion 和 hallucinated/ungrounded argument 侧边诊断。
- 如果 failure analysis 仍显示无支持，应考虑收缩方法叙述，回退到更简单的 ECPD-CRV / conservative allocation diagnostic，而不是强推完整 CARVE。
- 若要继续做三数据集 P5b，必须先定义新的阶段计划和 acceptance criteria，避免把单数据集 DuEE-Fin 结果扩展成全局结论。

## 交接硬边界

- 不修改 `data/` 和 `baseline/`，除非用户明确授权数据或 baseline phase。
- 不把 `docs/measurements/*_template.md` 当作证据。
- 不把 R5b dev diagnostic 写成 final-test、hidden-test、SOTA 或主表性能。
- 不启动远端 GPU 长任务，除非先报告 exact command、working directory 和 expected outputs，并得到明确授权。
- 服务器 `/data/TJK/DEE/dee-fin` 不应依赖 Git 状态；需要一致性时用本地 Git 加服务器 SHA256/hash 校验。
