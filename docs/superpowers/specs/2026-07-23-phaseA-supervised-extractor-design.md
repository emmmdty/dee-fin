# Phase A 实现设计稿 — Ch2 判别式监督关系抽取器

> 2026-07-23。承 `docs/phases/PHASE_A_ch2_supervised_extractor.md`(权威契约)+ 本会话探索现有代码后的**实现决策**。
> 目标:判别式 `@register("supervised")` 抽取器,金标节点上解 causal 召回 **0.4%(3/810)** 瓶颈,靶 causal F1 30–37 / subevent ~30。

## 关键决策(探索现有代码后确定)

1. **候选/标签/评测全复用 `relations/pairs.py`**(已实现且已测):
   - `candidate_pairs(doc, max_distance)` = 文档级全配对(有序对,PHASE_A 要求非仅相邻句)。
   - `pair_examples(doc)` = 候选宇宙 + gold 标签 = **pair-classifier 训练行**。
   - `pair_prf(...)` = pair-classification setting 下 per-family + micro P/R/F1(和文献 RoBERTa 51.8 可比)。
2. **评测走现有两段式,零新评测脚本**:`evaluate_relations.py --dump-predictions`(pipeline 跑 supervised,产边)→ `evaluate_relation_pairs.py`(CPU 离线 pair 打分,报 causal/subevent/temporal P/R/F1)。
3. **训练新建**:现有 `scripts/train_relation_extractor.py` 是 v3 生成式 LoRA(脚本自述"not the v4 Phase A discriminative extractor"),不复用。
4. **架构 = 复现 MAVEN-ERE 官方 RoBERTa pair-classification**:encode-once RoBERTa 编码整篇 → 触发词 mention 表示 `h_i` → 事件对特征 `[h_i; h_j; h_i⊙h_j; |h_i−h_j|]` → **三组独立头**(temporal / causal / subevent,各含 `NONE`+subtypes)。O(n) 次编码。(不选 cross-encoder 逐对 marker:O(n²)、非 baseline;留作后续开关)
5. **类不平衡**(causal/subevent 极稀疏):训练期**负例下采样 + 加权 CE 为主**,focal 作消融开关(PHASE_A"三选一并消融")。
6. **代码原则**(作者 2026-07-23 要求):简洁去冗余、**fail-fast**、无掩盖问题的 fallback/默认值——缺 gold/数据显式报错或计数跳过,不静默补默认。

## 组件

| 动作 | 路径 |
|---|---|
| 新建 | `src/finekg/relations/extractor/supervised.py`(`@register("supervised")`,torch **lazy**、CPU 可导入;`nn.Module` 打分器 + `extract()`) |
| 新建 | `scripts/train_supervised_relations.py`(判别式训练,GPU,checkpoint→`runs/relations/`) |
| 新建 | `configs/relations/supervised.yaml`(pipeline 选 supervised + extractor_kwargs 权重路径) |
| 新建 | `tests/relations/test_supervised.py`(**纯 CPU**) |
| 复用 | `pairs.py` / `evaluate_relations.py` / `evaluate_relation_pairs.py`(零改或极小改) |

## TDD 计划(本轮=本地,GPU 训练留服务器)

1. **测试先行(CPU,不实例化 torch)**:① registry 注册 `"supervised"` 存在;② `extract()` 契约——monkeypatch 打分器,验证 `candidate_pairs` 枚举 → 非 `NONE` 对生成 `RelationEdge`(subtype/directed/evidence/confidence 正确);③ 类不平衡负采样确定性;④ torch lazy——不装 torch 也能 `import supervised` 模块。
2. **实现** `supervised.py`:torch-guarded `nn.Module`(encode-once + 三头)+ `extract()`(候选→打分→边)。
3. **训练脚本 + config**;复用 `pair_examples` 造训练行。
4. **接线验证**:stub 预测 dump → `evaluate_relation_pairs.py` 打通(CPU)。
5. **本地验收**:`uv run pytest`(只增不改绿)+ `ruff` 0 + `finekg-smoke` 绿 → commit + push。
6. **服务器**(card 1 空闲):`git pull` → GPU 训练 → 回传 → 真实 causal/subevent F1 **如实**记 `docs/TODO.md`(降/未达也照报)。

## 止损(承 PHASE_A)

causal F1 <10% 且类不平衡/候选窗口/编码方式排查无果 → 维持受控模拟(`succession/cross_stage.py`),Ch2 收缩到一致性/修复/风控(Phase B 仍成立),Ch4 退受控扰动版。
