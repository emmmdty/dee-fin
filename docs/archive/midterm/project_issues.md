# Fin-EKG 中期答辩问题清单

> 日期：2026-06-17。目标：给下一个窗口按优先级继续分析和修改。

## P0 - 远端代码同步/测试状态（已解决）

- 原现象：远端测试文件曾是旧版，`tests/rl/test_configs.py::test_expected_config_files_exist` 失败。
- 处理：已同步相关测试文件，并在本地/远端重跑 `pytest`。
- 当前结果：本地和远端均为 `187 passed, 2 skipped`。可以声称中期代码质量门全绿（GPU-heavy tests 仍按设计跳过）。

## P0 - ch1/SARGE 到中文图谱的数据契约未打通（已修复）

- 原现象：`data/processed/event_graph_zh/event_graph.json` 为 528 节点、0 边，metadata 显示 `edges_raw=495`、`edges_dropped_ungrounded=495`。
- 原因：`scripts/sarge_to_event_nodes.py` 只保留事件类型、参数、主体和时间，不保留 trigger、trigger evidence、argument evidence；启发式关系抽取器用节点证据生成边证据，默认 `require_evidence=True` 后无证据边会被 grounding 过滤。
- 处理：`scripts/sarge_to_event_nodes.py` 新增 `--source-docs`，从 DuEE-Fin-dev500 原文精确定位预测参数文本并写入 `argument_evidence`；已有预测 span 会被保留，但不使用 gold trigger。
- 当前结果：选定 SARGE pred `sarge_infer_DuEE-Fin-dev500_dev_20260519T043458Z` 重建后，`event_graph_zh` 为 677 节点、20683 条闭包后一致性图边，metadata 为 `edges_raw=498`、`edges_dropped_ungrounded=1`。
- 剩余口径：SARGE prediction 仍不导出 trigger span，因此 trigger/trigger_evidence 继续为空；中文图当前是 evidence-grounded heuristic relation graph，不应声称是 learned Chinese relation graph 的最终闭环。

## P1 - 下游交易配置曾指向不存在的图文件

- 现象：`configs/downstream/trading_selective.yaml` 原先指向 `data/processed/event_graph_zh/graph.json`，实际文件为 `event_graph.json`。
- 影响：直接运行 `scripts/evaluate_downstream_trading.py --config configs/downstream/trading_selective.yaml` 会 `FileNotFoundError`。
- 处理：本轮改为实际存在路径，并加测试保护。

## P1 - event_graph_zh manifest 固化本机绝对路径

- 现象：远端 `data/processed/event_graph_zh/manifest.json` 中曾出现 `/home/tjk/...`。
- 原因：`prepare_event_graph_zh` 使用 `str(Path)` 写 manifest。
- 影响：影响远端复现可信度，答辩追问时容易暴露环境混乱。
- 处理：本轮改为仓库相对路径，并刷新现有 manifest。

## P1 - lint 质量门不可用

- 现象：`uv run ruff check .` 会扫描 `data/raw` notebook 和 `docs/midterm` 一次性画图脚本，产生数百个与项目代码无关的问题。
- 影响：无法作为中期前的稳定质量门。
- 处理：本轮在 Ruff 配置中排除 `data/raw` 和 `docs/midterm`；项目代码仍通过 `src tests scripts` 覆盖。

## P2 - downstream selective trading 暂不适合作为正向卖点

- 验证：修正图路径后，CPU 跑通 `evaluate_downstream_trading.py`。
- 结果：`n_signals=270366`，整体 accuracy 约 0.480，`accuracy@0.25` 约 0.465，选择性覆盖越低越差。
- 中期口径：不要把该结果放入主线成果；可作为 ch3 风控/下游应用的后续工作。

## P2 - ch1 事件证据字段当前为空较多（部分修复）

- 观察：677 个中文事件节点中，673 个已有 `argument_evidence`；`trigger` 和 `trigger_evidence` 仍为空，因为当前 SARGE prediction 文件没有导出 trigger 字段。
- 影响：argument-level grounding 已支撑关系边保留；trigger-level grounding 仍需要上游 SARGE export 增强。
- 后续修复：在 SARGE 侧导出预测 trigger/span 后，转换层直接保真并重建图。

## P2 - 中期事件图展示曾过于像“卡片图”（已修复）

- 原现象：`fig9_event_graph_example.png` 只把事件字段堆在卡片里，看不出事件节点、属性节点和关系边。
- 处理：`docs/midterm/make_event_graph_example.py` 现在先构建 NetworkX `MultiDiGraph`，将事件、参与实体、时间、数量建成 typed nodes，再用 `pledger`、`pledged_company`、`time`、`shares`、`BEFORE`、`COREF` typed edges 连接。
- 当前结果：已生成 `fig9_event_graph_example.{png,graphml,json,cypher}`；其中 `.cypher` 可直接导入 Neo4j，`.graphml/.json` 可供 NetworkX/Cytoscape 等工具继续查看。

## 已确认可用成果

- ch2：GRPO-RLVR 在 MAVEN-ERE valid 上优于 SFT，主推 coref CoNLL 0.265 到 0.771，辅推 temporal precision 0.445 到 0.565。
- ch3：ICEWS14 上 path-RL MRR 0.360，优于 temporal-GNN 0.286 和 frequency 0.105。
- 本地 smoke 全链路可运行；远端训练/评测产物位于 `/data/TJK/Fin-EKG/runs`。
