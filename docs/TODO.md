# Fin-EKG TODO / 实时状态

> 更新于 **2026-07-23**。本文件只记录 v4 的当前执行位置、已验证证据和下一步；设计定义见
> [`SPEC.md`](SPEC.md)，阶段验收见 [`phases/`](phases/README.md)，历史路线见 `archive/`。

## 当前结论

- **唯一主线**：v4 四章可信事件图谱构建（身份 → 结构 → 事实 → 传播/下游），headline 是 Ch4
  下游门控闭环修复与构建误差传播。
- **关键路径进展**：Phase A **已达标**（2026-07-24）——判别式 `supervised` 抽取器在金标节点上把 causal
  召回 0.4%→67.5%、causal F1 达 .250、subevent .213、temporal .338（`hallucinated=0`）。**当前关键路径转向
  Phase B**（全局一致解码 + repair trace + CRC 风控准入），并用 Phase A 的 predicted 图做真实图闭环。
- **执行状态**：P0 数据完成；**Phase A 代码完成并冒烟验证通过**（判别式 supervised 抽取器 + 训练脚本 +
  评测接线，CPU 测试全绿），全量训练进行中、**真实 F1 未出**。Phase B–E 的真实图实验依赖 A。
- **旧线定位**：SeDGPL、M1/M2、CS-CRP 和受控 cross-stage 扫描保留为 Ch4 可靠性模块；SARGE
  保留为金融应用层；旧实体中心 TKG 只在 tag `frozen-tkg-line`。
- **2026-07-23 联网复核**（结论并入 [`EXPERIMENTS.md`](EXPERIMENTS.md) + SPEC §4.5/§5）：数据/SOTA/baseline/
  评测协议全部真实可得，方案可执行；三个必修点——① **Phase A** 唯一硬骨头（0.4%→SOTA 30–37，有文献路线）；
  ② **Ch4 门控 oracle 澄清**（SPEC §4.5）；③ **新竞品 DeepRefine 2605.10488** 收窄 headline（SPEC §5）。
  Ch4 baseline 已补 2025 方法（Semantic Relation Experts 2506.06910 / 现代 LLM），弃用旧 Llama3/GPT-3.5。

## 已完成并有证据

### 工程与数据

- 冻结 schema、I/O、registry、图算法、通用评测与 calibration 原语已实现。
- MAVEN-ERE / MAVEN-Arg / MAVEN-FACT 公开 train/valid 已处理并在 WSL 与 4090 就位；三套数据的
  doc-id 对齐关系已核验。官方 test 标签隐藏，不进入本地调参。
- DocEE、It-Happened、ModaFact 已有 processed manifest；MATRES、RAMS、WikiEvents、ECB+ 当前仅 raw
  就位，尚无项目 processed 输出，不能写成“已预处理”。详见 `DATASET_SURVEY.md`。
- 2026-07-22 当前主干验证：`239 passed / 11 skipped`、Ruff 0 error、`finekg-smoke` 通过。

### Phase A 实施（2026-07-23，代码层已完成）

- `@register("supervised")` 判别式抽取器：文档级候选与标签**复用 `relations/pairs.py`**、torch-lazy（CPU 可
  导入/实例化）、`extract` 产 evidence-grounded 边；训练脚本 `train_supervised_relations.py`（确定性负采样 +
  逆频类权重 + 加权 CE）；`configs/relations/supervised.yaml` 接**既有两段式评测**
  （`evaluate_relations --dump-predictions` → `evaluate_relation_pairs`，零新写评测脚本）。
  CPU 测试 10 passed + 1 torch-skip；全量 pytest / ruff / finekg-smoke 全绿。
- **修复触发词定位缺陷**：MAVEN-ERE `trigger_word` 是小写形式而句子保留原始大小写，句首/专名触发词精确
  `find` 必然失配（实测 train_smoke 6/919 = 0.65%、valid_smoke 4/637 = 0.63%）；改**大小写不敏感 + 词边界**
  后降为 **0.00%**。该问题由训练侧 fail-fast 暴露——loader 对同样失配是容忍的（记 `span=(0,0)`）。
- 全量训练完成（2913 docs × 3 epochs，loss 4.12 → 2.49 → 2.04 → 1.76）。**首轮 pair-classification
  评测（valid 710 篇）如实结果：**

  | 关系 | P | R | F1 | 调阈值后最佳 F1 | 目标 |
  |---|---|---|---|---|---|
  | causal | .049 | **.675** | .091 | **.167**(thr .9) | ≥.25 ❌ **未达标** |
  | subevent | .043 | **.881** | .082 | **.206**(thr .95) | ≥.20 ✅ |
  | temporal | .191 | .575 | .286 | .317(thr .5) | — |

  - **召回瓶颈已破**：causal recall **0.4% → 67.5%**、subevent **0% → 88.1%**，`hallucinated=0`
    （判别式不产生端点不存在的幻觉边，相对生成式的结构优势）。
  - **但 precision 崩**：阈值扫到 .99 时 causal P 仅 **.240** → 模型未学出判别边界，**不是决策规则问题**，
    阈值救不回来。
  - **诊断**：负采样 3:1 vs 真实候选分布约 63:1，再叠加逆频加权 CE = **双重补偿**，把模型教成宁滥勿缺。
  - **类不平衡消融（已完成一轮 α 扫描，各配置最佳阈值的 F1）：**

    | neg30 配置 | causal F1 | subevent F1 | temporal F1 |
    |---|---|---|---|
    | α=1.0（inverse） | .161 | .202 ✅ | .316 |
    | **α=0.5** | **.234** | **.221** ✅ | **.397** |
    | α=0.25 | .232 | .219 ✅ | .416 |
    | α=0.0（none） | .186 | .041 ❌ | .407 |
    | 目标 | ≥.25 | ≥.20 | — |

    - α 曲线**倒 U 形，最优在 0.25–0.5**；per-family α（给 causal 更高 α）**反而降 causal F1**
      （.234→.219→.205）→ causal 瓶颈不在权重强度。**neg-ratio 在 α=1 时几乎无效**（逆频权重抵消负采样）。
    - **✅ 达标（2026-07-24）**：`neg30 · α=0.5 · 6 epochs` —— **causal F1 .250 / subevent .213 /
      temporal .338**（阈值 0.7，micro .311，`hallucinated=0`）。3→6 epochs 是决定性一步（loss 1.25→0.92
      仍在降＝3ep 欠拟合），把 causal .234→**.250** 推到目标下沿。交付 checkpoint = `runs/relations/supervised_maven`。

### Ch4 先行模块（来自 v3，降级复用）

- **SeDGPL 自跑基线**：CGEP-MAVEN 单折 MRR 0.1836 / strict 0.1265，n=1908。
- **M1 距离选边**：MRR 0.1889 / strict 0.1304；相对匹配重跑约 +0.002，属于噪声级，留作消融。
- **M2 结构编码**：MRR 0.1852 / strict 0.1290；无可信增益，留作负结果消融。
- **M3a 选择性预测**：ACI 各风险档覆盖达到目标；同覆盖下 SeDGPL 相比 frequency 的集合缩小
  约 43%–68%。
- **M3b 受控扫描**：真实 SeDGPL 排名下，naive coverage 随 reachability loss 下跌；预算方法在预留档
  更稳。它是受控证据，不等于真实 predicted/repaired 图闭环。

## v4 阶段状态

| 阶段 | 任务 | 当前状态 | 完成门槛 |
|---|---|---|---|
| P0 | 主数据与溯源 | ✅ 主干数据完成；扩展数据部分仅 raw | 主数据 hash/manifest 可核 |
| A | Ch2 判别式关系抽取 | ✅ **达标**（causal F1 .250 / subevent .213 / temporal .338；召回 .4%→67.5%） | causal F1 ≥25（目标 30–37），subevent ≥20 |
| B | 一致性、repair trace、风险准入 | 🟡 consistency/CRC 已有；repair trace 和真实图实验未做 | violation↓、分层 FNR、ECG 可重建率↑ |
| C | Ch1 规范事件节点 | ⬜ 未开始；schema/coref/calibration 可复用 | 检测 F1、CoNLL、误合并率、ECE |
| C2 | Ch1 跨文档泛化 | ⬜ 未开始；ECB+ raw 已有，CLES 未取 | ECB+/CLES 对比 SECURE/MEET/DIE-EC |
| D | Ch3 事实性与净化 | ⬜ 未开始；MAVEN-FACT train/valid 已就位 | macro-F1、预测图掉点、净化下游增益 |
| E | Ch4 闭环与三图传播 | 🟡 SeDGPL/受控扫描已有；闭环控制器未做 | repaired > predicted，三图误差曲线 |
| F | 端到端误差预算 | 🟡 通用传播原语已有；真实三段预算未做 | 显式前提下的界、分层 FNR、naive 对照 |
| G | 金融应用层 | 🟡 SARGE/CCKS adapter 已有；v4 迁移未做 | 金融构建→预测案例 |
| H | 多种子、消融、新颖性 | ⬜ 等 A–F 主结果 | mean±std、完整消融、投稿前新颖性扫 |
| I | 写作 | ⬜ 等主实验 | 初稿与终辩材料 |

## 下一步

1. ~~实现 Phase A 判别式抽取器~~ ✅ 代码完成、冒烟通过。**等全量训练结束跑两段式评测**，如实回填
   causal/subevent P/R/F1（对照生成式 0.4%）；未达标按 PHASE_A 止损条款处理。
2. 本地验收后，在启动远程长训练前单独给出完整命令、工作目录、选卡、时长和产物，并等待明确授权。
3. Phase A 达标后立即进入 B，先证明 predicted ECG 的 reachability 可用，再投入 C/D/E。
4. 多种子和进一步调 M1/M2 放到 Phase H；主闭环未通前不扩张实验面。
5. 每章开跑前照 [`EXPERIMENTS.md`](EXPERIMENTS.md) 定 baseline（新老搭配）+ 消融矩阵 + 评测档；Ch4 主表
   纳入 2025 近期方法（Semantic Relation Experts / 现代 LLM），不再用旧 Llama3/GPT-3.5。
6. 实现 Ch4 闭环控制器前，**先定死门控信号来源**（SPEC §4.5：无标签代理 or 离线诊断定位），别默认用金标 MRR。

## 止损与人工判断

- Phase A causal F1 <10% 且类不平衡、候选范围、编码方式均排查无果：保留受控模拟，Ch2 收缩为
  一致性/修复/风控，Ch4 收缩为受控误差传播。
- Ch3 净化无下游收益：保留事实性检测与 predicted-input 鲁棒性分析，不宣称净化有效。
- Ch4 repaired 不优于 predicted：退为一致性重排 + 误差传播分析，不更换指标掩盖负结果。
- Ch4 门控只能靠金标 MRR（无可用无标签代理）：显式改定位为「离线构建期质检工具」，不声称在线自愈（SPEC §4.5）。
- 发现 CS-CRP/reachability 组合有直接先例：重新限定或更换命名，不写“首次”。
