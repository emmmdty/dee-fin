# M2 结构感知编码 · 设计稿（2026-07-17）

> SPEC §4.2 的 M2 槽位落地。**未提交**（CLAUDE.md：提交仅在用户明确要求时）。
> 结论：第四路 = 单个 `reach_anchor` bit，经 `nn.Embedding(2, hidden)` 进 EeCE 第三个门，默认关。

## 1. 目标与约束

在已完成的自跑 SeDGPL 基线（EeCE 两级门融 instance/sent/type 三路）之上，给 EeCE 加**第四路结构流**
= 共享验证器的结构一致性信号，per 事件 token。`model.py::GatedFusion` 已抽出，故这是**组合非重写**。

硬约束：包/函数名不含 `ch1/2/3`；新组件走 registry/flag；`EventNode` schema 零新增（本改动只碰
succession 侧编码，不碰 `core.schema`）；`tests/core/test_propagation.py` 测试锁不动；**默认关 =
baseline 逐字节一致**（EeCE/SeDGPL 在关时不构造新模块，RNG 抽样序列与今天相同，现有 seed-209
基线 0.1867 直接充当 OFF 臂）。

## 2. 证据（决定信号内容的依据，需入论文消融）

**文献 lineage**：给事件因果 transformer 注结构，学界主流是距离/结构位置编码——HDSE（NeurIPS'24，
证明优于最短路距离，arXiv:2308.11129）、CAPE（NeurIPS'25 因果 DAG 位置编码，arXiv:2509.16629）、
Graphormer/GRPE（把结构偏置注进 attention）、ERGO（事件关系图 transformer，arXiv:2204.07434）、
结构感知 ECG 线性化（arXiv:2403.11129）。我们**停在输入流融合、不改 attention**，以守"组合非重写"；
attention-bias 注入列为 future work。

**CPU 预筛**（真实 CGEP-MAVEN valid=1908，特征在线性化存活子图上算，脚本见 scratchpad
`m2_feature_probe.py`；extrinsic 用 `runs/cgep/maven_sedgpl_ranks.json` 真实 gold 排名，按
seed-209 shuffle 逐条对齐）：
- **内在冗余**：proximity instance 内 std 仅 0.056（近常数，DsGL 排序已吃掉）；proximity×reach_anchor
  |ρ|=0.85，四特征两两 |ρ| 0.46~0.85 → 4 维过参数化。
- **外在相关（与真实 SeDGPL 难度）**：全部 |ρ|≤0.19；最强单特征 = `frac_reach_anchor` ρ=−0.19
  （anchor 有因果支撑时 median rank 11 vs 无支撑 32）。
- **多变量 5 折 CV R²（结构解释真实难度的比例）**：reach_anchor 单特征 0.032；+度 0.030（降）；
  +proximity 0.029（再降）；全加 0.047（上限）。→ **结构最多解释 ~3–5% 难度**，且 **reach_anchor 一个
  bit 就是全部 per-token 信息**（加任何 per-node 特征出样本无增益）。

**预注册结论**：M2 的 MRR 增益几乎必然是噪声/小量级（≤5% 天花板）。这不改动机——M2 是机制槽位，
如实报出（正/负）+ 上述 CPU 证据即完整一节；论文价值不依赖 M2 的 MRR（依赖 CS-CRP）。

## 3. 信号定义

`reach_anchor(event) ∈ {0,1}` = 该事件是否经**有向**因果/子事件边（存活子图）可达 anchor＝是否为本次
预测的"上游证据"。是 CS-CRP 可达性货币下沉到节点级，直接体现 evidence-grounded 身份。
- 覆盖 template 事件 + anchor；gold 是 mask、其句从不编码 → 不入特征，**无泄漏**。
- 在**线性化存活子图**上算（=编码器实际所见，与 sent/type 流同口径）。anchor 自身 reach=1。

## 4. 单元划分（各一职、可独立测）

1. `succession/structure.py`（新，纯 CPU/无 torch）：`event_reach_anchor(surviving_edges, query, event_nodes)
   -> list[int]`，顺序对齐 `event_token_nodes`。TDD 主战场。（预留可扩展：将来消融可加度/proximity 维。）
2. `encode.py`：`EncodedInstance` 加字段 `reach_anchor: list[int]`；`encode_instance` 填充（与
   `event_positions` 天然同序）。始终产出（廉价图事实），模型关时不读。
3. `model.py`：`EeCE(enable_structure=False)` — 开启时才构造 `reach_embed=nn.Embedding(2, h)`（**zero-init**）
   与 `gate_structure=GatedFusion(h)`；`forward` 多收可选 `reach_anchor`，`struct=reach_embed(reach)`，
   **`h3=gate_structure.residual(h2, struct)=h2+g3·struct`**（门控残差）。关时 forward≡两门版、构造≡今天。
   ⚠️**调试记录**：初版用插值门 `h3=g3·h2+(1−g3)·struct` + 默认 `nn.Embedding` 初始化 → GPU 实测 MRR
   **腰斩 0.1867→0.0883**。根因（服务器诊断 `diag_m2.py`）：默认 N(0,1) embedding 行范数 ~28 碾压融合
   `h2` 范数 ~8，init 时第三门扰动事件 embedding **185%**（`||h3−h2||/||h2||=1.85`），lr=1e-6 十轮救不回。
   修法＝zero-init（struct init=0）+ 残差（`h2+g·struct`，y=0 时恒等）→ init 扰动 **→0.0000**，ON 臂起点
   ＝baseline、只能学着"加"结构。对照工作模式：SeDGPL 对 `<a_i>` 新行专门 mean-init 正是"新 embedding 不能
   是随机噪声"。教训：给已调好的门控编码器加 embedding 流，**必须 no-op 起步**。
4. `sedgpl.py`：`SeDGPLPredictor(enable_structure=False)`；`build_sedgpl` 透传；`_forward` 关时不变、
   开时把 `reach_anchor` 转 tensor 传入 EeCE。
5. `evaluate_cgep.py`（及 selective 脚本）：加 `--structure-encoding` flag，默认关。

## 5. 数据流

```
instance → truncate/select 存活边 → event_reach_anchor → EncodedInstance.reach_anchor
        → LongTensor[n_tok] → EeCE.reach_embed(zero-init) → 门控残差 h3=h2+g3·struct → template encoder
```

## 6. 测试（TDD，先测后码）

- CPU `test_structure.py`：reach 有向性（上游=1、下游/旁支=0）、anchor 自达=1、与 `event_token_nodes`
  对齐、gold 不入、存活子图口径。
- torch 守卫（本地 skip / 服务器跑）：`EeCE(enable_structure=True)` 第三门形状；`enable_structure=False`
  时 EeCE 输出 ≡ 两门版（byte-identical 证据）；`nn.Embedding(2,h)` 接线。
- 全套只增不改（现 323 passed / 7 skipped）。

## 7. GPU A/B（等用户明确点头才起）

单臂：`--structure-encoding` ON，seed 209、单折 10ep ≈ 2.4h，对照 OFF（=现有 0.1867，同 seed）。
逐卡 `nvidia-smi` 核 free≥8GB、优先 card 1、nohup 脱离 ssh、本地轮询三态判活。
**止损**：持平/微动 → 如实记消融附录（配 §2 CPU 证据），不硬 tune。

## 8. 非目标

不改 attention（future work）；不加度/proximity 维（证据示无增益，仅留消融钩子）；不碰旧 TKG 线；
不写专利/论文正文。
