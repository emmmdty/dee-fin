# Fin-EKG TODO / 实时状态

> 更新于 **2026-07-23**。本文件只记录 v4 的当前执行位置、已验证证据和下一步；设计定义见
> [`SPEC.md`](SPEC.md)，阶段验收见 [`phases/`](phases/README.md)，历史路线见 `archive/`。

## 当前结论

- **唯一主线**：v4 四章可信事件图谱构建（身份 → 结构 → 事实 → 传播/下游），headline 是 Ch4
  下游门控闭环修复与构建误差传播。
- **当前关键路径**：Phase A——在金标事件节点上实现判别式 `supervised` 关系抽取器，解决生成式
  SFT+GRPO 探针的 causal recall 0.4%（3/810）和 subevent 0%（0/139）瓶颈。
- **执行状态**：P0 数据基本完成；Phase A 尚未实现。Phase B–E 的真实图实验依赖 A。
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
| A | Ch2 判别式关系抽取 | ⬜ 未开始；当前总瓶颈 | causal F1 ≥25（目标 30–37），subevent ≥20 |
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

1. 按 [`PHASE_A_ch2_supervised_extractor.md`](phases/PHASE_A_ch2_supervised_extractor.md) 先写 CPU
   候选/标签/registry 测试，再实现 lazy-import 的判别式抽取器。
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
