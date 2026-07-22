# 设计：ch2 评测口径修复 + ch1 事件抽取统计（中期）

> 日期 2026-06-16 · 依据 `docs/MIDTERM_HANDOFF.md` 的 P1/P2 · 用户批准范围："尽量多做（含 ssh 重跑）"、P1 修法 = **A+C**。
> 注：本仓库非 git 仓库，故本 spec 仅落盘存档，不做 commit。

## 背景与问题（P1）

ch2 关系抽取评测 `relation_prf`（`src/finekg/core/eval/relation.py`）按 `(head, tail, type, subtype)` 精确匹配。
MAVEN-ERE 的 `temporal_relations` 在官方数据里**本就是传递闭包**，loader（`relations/data/maven_ere.py`）原样读入：
- temporal gold = **109933** 条（占全部 gold 87%），而 LLM 稀疏预测仅 4905 条
- → temporal recall 0.025、micro F1 **0.07 失真**，不可裸用、不可投稿

coreference 正常（gold 4029，P/R/F1 = 0.85/0.48/0.61，CoNLL 0.77），不受影响。
causal(9698)/subevent(2826) 的低 recall 是**真·抽取难度**（非闭包伪影），不做闭包。

## 决策（A+C）

- **A 闭包对齐预测**：对预测的 strict-temporal（`STRICT_TEMPORAL_SUBTYPES = {"", "BEFORE"}`）边做传递闭包后，再与（已闭包的）gold 匹配——贴合 MAVEN-ERE 官方打分器。预测链 a→b→c 自动获得 a→c 信用，recall 公平化。precision 也在闭包后预测集上算（伪边照样传播，公平）。
- **C 报告策略**：保留分类型 P/R/F1，头条仍主推 coref(0.61)/CoNLL(0.77)；micro **raw 与 closure-corrected 两个都报**，不藏数。

为什么不选 B（gold 传递归约）：改变任务语义（正确隐含边变假阳），不如 A 好辩护。

## 实现

### P1a — metric（core，TDD）
- `core/graph.py` 新增 pairs 级原语：`close_pairs(pairs) -> set[(h,t)]`（`nx.transitive_closure` 包一层）。
- `core/eval/relation.py`：`relation_prf(..., temporal_closure: bool = False)`。开启时：把 temporal strict 子类型的 pred/gold 抽成 (h,t) 对 → `close_pairs` → 以 canonical 子类型 `BEFORE` 重建归一化 key 后计数；非 strict temporal（OVERLAP 等）与其它族不变。**默认 False，向后兼容**，旧测试与 `make_figures.py` 不受影响。
- 闭包在语料级 pair 集上算是安全的：事件 id 以 `doc_id::` 命名空间，无跨文档边，语料闭包 == 各文档闭包并集。
- 测试：`tests/core/test_graph.py` 加 `close_pairs`（链→闭包、空、自环防护）；`tests/core/test_metrics.py` 加 closure 用例（稀疏链 vs 闭包 gold：raw recall 低、closure recall 高；非 strict 不被闭包；causal 不变）。

### P1b — 评测脚本（core）
- `scripts/evaluate_relations.py`：metrics 同时含 `relation_prf`（raw，旧 key 不动）与 `relation_prf_temporal_closed`。
- 加 `--dump-predictions PATH`：把每文档预测边 `{doc_id, edges:[(h,t,type,subtype,directed)]}` 落盘（小，~7k 边）。gold 可本地从 valid.jsonl 复算 → 日后改口径无需上 GPU。

### P1c — 离线重算 + CPU 验证
- `scripts/recompute_relation_metrics.py`：吃 pred dump + 本地 maven_ere gold → 离线算 raw/closed/coref，写 metrics JSON。
- 用 `configs/relations/heuristic_baseline.yaml` 在真实 maven_ere valid（或 valid_smoke）上 CPU 端到端验证闭包口径行为合理（temporal recall 在闭包后显著上升且 ≤1）。

### P2 — ch1 事件抽取统计（本地）
- 数据：`data/processed/event_graph_zh/event_graph.json` = **677 节点 / 20683 边**（参数 evidence 已从 SARGE 预测文本回填；trigger span 仍缺失）。
- 算节点级统计：事件总数、事件类型分布（Top-K）、时间/来源/触发词覆盖率等（取决于节点字段）。
- 产物：`docs/midterm/make_ch1_figures.py` + `docs/midterm/figures/fig_ch1_*.png`（1–2 张：类型分布等）+ `docs/midterm/table_ch1_event_extraction.md`。

### ssh 重跑（刷新 LLM 头条数）
- 前置已验证：`ssh gpu-4090` 通；card0/1/2 全空闲；SFT(`runs/relation_extractor_lora`)+GRPO(`runs/relation_grpo/phase2`) adapter、数据、垫片、配置、Qwen 模型均在。
- rsync 改动文件 → card1/card2 并行跑两 adapter 的 `evaluate_relations.py`（`--dump-predictions` + 垫片+离线前缀 + nohup `python -u`）→ 监视 → 拉回 JSON+dump → 更新 `docs/midterm/data/` 与图，并同步交接文档结果区。

## 测试与验收
- `uv run pytest`（现 ~127 测试）全绿 + 新增 closure 测试绿。
- raw 口径下数字与现有 `docs/midterm/data/eval_phase2.json` 一致（回归不破）。
- closure 口径给出公平 temporal F1（recall 明显 > 0.025）。
- ch1 统计表/图落盘且数与 json 一致。

## 风险 / 非目标
- 风险：服务器 LLM 评测 710 文档较慢；cpolar 隧道可能抖（监视脚本容忍 ssh 失败）；card0 或被 LiAo 占用→优先 card1/card2。
- 非目标：P3 风控（C1）、P4 多 seed/消融/下游选股（需更多算力，本轮不做）。
