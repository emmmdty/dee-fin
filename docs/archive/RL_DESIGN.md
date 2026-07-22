> ⚠️ **主线已更新（2026-07-11）**：以 `docs/HANDOFF_2026-07-11.md` 为准。本文的**框架仍有效**
> —— verifier-as-reward 是算法创新点 **A1** 的载体（四分量复合奖励 + 课程 + GRPO）。但方法 A
> 需扩成**五分量**（新增「事理结构约束」奖励：时序传递闭包 / 因果无环 / 共指等价类 / 子事件层级），
> 方法 B（时序图路径 RL）服务的旧 ch3 TKG 外推**已降级为消融附录**。金融口径与实验矩阵已过时。

# 验证器即奖励:面向金融事件图谱构建与时序推理的强化学习设计

> Verifier-as-Reward: Reinforcement Learning for Evidence-Grounded Event Graph
> Construction and Temporal Reasoning.
>
> 本文档是 RL 增强部分的权威设计稿。代码不含章节标记;章节↔代码映射见
> `docs/RESEARCH_MAP.md`,架构机制见 `docs/ARCHITECTURE.md`。

## 1. 定位与动机

已有系统是"多智能体 + 可验证性"组织:专精 agent 提议,验证器(grounding /
consistency / faithfulness)在**推理时**把关。本设计把同一组验证器内核升级为
**训练时的奖励信号**,形成统一框架:

> **训练时:验证器 = 奖励(verifier-as-reward);推理时:验证器 = 门控(verifier-as-gate)。**

两个落点,各对应一个功能域:

| 落点 | 功能域 | 方法 | 训练信号 |
| --- | --- | --- | --- |
| A. 关系抽取 GRPO-RLVR | `relations` | SFT 后接 GRPO 后训练 | 证据接地 + 全局一致性 + 任务 F1(复合可验证奖励) |
| B. 时序图谱路径 RL | `forecasting` | 图上路径游走 MDP + 组相对基线 | 终点命中 + 干预忠实度 bonus + 势塑形 |

两者共享 `finekg/rl/` 的组相对优势、奖励组合、课程与塑形基底;
推理产物(LoRA adapter / 路径策略)直接接回现有 pipeline 与多智能体验证栈,
框架零改动(registry 挂接、schema 零新增字段)。

## 2. 方法 A:复合可验证奖励的 GRPO 关系抽取

### 2.1 形式化

沿用现有一次性全对抽取的提示 `build_relation_prompt`(事件列表含触发词,并附
原文摘录——evidence_quote 必须逐字摘自其中,否则接地不可满足)。对每个 prompt
$x$(一个事件窗口),从当前策略采样 $G$ 条补全 $\{y_1,\dots,y_G\}$,逐条计算复合奖励:

$$R(x, y) = w_f R_{\text{format}} + w_g R_{\text{ground}} + w_c R_{\text{consist}} + w_t R_{\text{task}}$$

| 分量 | 定义(全部复用现有内核) | 防 hacking 设计 |
| --- | --- | --- |
| $R_{\text{format}}$ | JSON 可解析且含 `relations` 列表;按合法条目占比缩放 | 权重最低(0.1),课程后期可降 |
| $R_{\text{ground}}$ | `ground_relations` 判定 evidence_quote 为原文子串的边占比 | quote 超长(> `max_quote_chars`)视为未接地;**含金标窗口空预测记 0**(不许靠不输出避罚),无金标窗口空预测=正确答案记 1(防"接地幻觉"反超诚实沉默) |
| $R_{\text{consist}}$ | $1-$ `GreedyConsistencySolver` 修复时的丢边率(因果/时序环冲突) | 用"违例率"而非加分制,闭包刷边无利可图;空预测同上(含金标 0 / 无金标 1) |
| $R_{\text{task}}$ | `relation_prf` 对金标边的 micro-F1 | 权重最大(0.4),外部评测以同一 F1 为准绳 |

组内相对优势(GRPO,无价值网络):

$$\hat A_i = \frac{R_i - \operatorname{mean}(R_{1..G})}{\max(\operatorname{std}(R_{1..G}), \epsilon_{\text{std}})}$$

目标为带 clip 与 KL 锚定(`beta`)的策略梯度;参考策略 = SFT adapter
(PEFT 下关 adapter 即参考模型,无第二份权重)。

### 2.2 课程学习(progressive sample mixing)

难度 = 窗口事件数。按 `phases: [{max_difficulty: 6, steps: 200}, {max_difficulty: 12,
steps: 300}, {max_difficulty: 24, steps: 300}]` 渐进混合(后期阶段包含前期样本),
每阶段续训同一 adapter,阶段末存 checkpoint。
长文档按 `window_events` 切窗,既控难度又控 prompt 长度。

### 2.3 训练配置(见 `configs/relations/grpo_rlvr.yaml`)

Qwen3-4B + LoRA(r=16)自 SFT adapter 热启;G=8、lr=1e-5、KL β=0.01、bf16、
梯度检查点;rollout 后端三级:`vllm_server`(双卡,默认)→ `vllm_colocate`
(单卡)→ `hf`(transformers generate,零版本耦合兜底)。逐分量奖励曲线落盘,
监控 reward hacking 与长度漂移。seeds 13/17/42 报 mean±std。

## 3. 方法 B:忠实度塑形的时序图谱路径 RL

### 3.1 MDP 定义

给定历史四元组集合 $\mathcal{G}$ 与查询 $q=(s_q, r_q, ?, t_q)$:

- **状态** $s_t = (q, e_t, \text{path}_{0..t})$,起点 $e_0 = s_q$;
- **动作** $A(s_t) = \{(r, e', t') : (e_t, r, e', t') \in \mathcal{G},\ t' < t_q\} \cup \{\text{STAY}\}$,
  含逆向边(`inv:` 前缀);按时间近因取 top-K(`max_actions`)截断;
- **转移** 确定性;固定 horizon $T$(默认 3),终点 $e_T$ 即预测答案。

### 3.2 奖励:命中 + 忠实度 + 势塑形

$$R(\tau) = c_h \mathbb{1}[e_T = o_{\text{gold}}] + c_f F(\tau)\,\mathbb{1}[e_T = o_{\text{gold}}] + c_s \sum_t \big(\gamma\,\Phi(s_{t+1}) - \Phi(s_t)\big)$$

- **忠实度 bonus** $F(\tau)$:把轨迹经过的 quads 从历史中消融,用 frequency
  代理重打分,取 `intervention_faithfulness(base, ablated)` —— 只有"承重"
  路径(消融后预测确实退化)才有 bonus,装饰性路径得 0。与推理时
  `FaithfulnessVerifierAgent` 用同一干预核,即"训练时奖励=推理时门控"。
- **势塑形** $\Phi(s)$ = 当前实体作为该查询答案的归一化频率-近因得分。
  基于势函数的塑形保证最优策略不变(Ng et al., 1999),仅缓解稀疏奖励。

### 3.3 策略与优化

小型 GRU 策略(实体/关系 embedding 64 维 + Δt sin-cos 编码 + GRU-128,约
1–2M 参数)。**组相对基线替代 TITer 的价值网络**:同一查询采 $G=16$ 条轨迹,
组内标准化优势 + REINFORCE + entropy bonus,无 critic(GRPO 思想在图游走
上的迁移,算法翻新点)。冷启动:frequency 启发式引导路径做行为克隆
warm-start,再进 RL。

### 3.4 推理与闭环

注册为 `@forecasters.register("path_rl")`:N 次 rollout 聚合终点得分 →
`Prediction.ranked`;最优路径逐步写成 `EvidenceLink(reference=quad.ref())`,
**直接可被现有 faithfulness verifier 消融、被 conformal calibrator 弃权**,
即插入 `multiagent_faithful` 编排为新的 reasoner。

## 4. 创新点口径(自圆其说,不 claim 全球首创)

1. **可验证性奖励化的统一框架**:证据接地、全局一致性、干预忠实度——原本是
   推理时验证器——被系统化为两章共用的训练奖励。区别于 AutoGraph-R1
   (arXiv:2510.15339,奖励=下游检索效用)与 EventRL(arXiv:2402.11430,
   奖励=纯 F1):我们的奖励显式编码*证据可核验性*与*图级一致性*。
2. **RL-TKG 老方法的 LLM 时代翻新**:TimeTraveler/TITer(2021)线引入
   (i)GRPO 式组相对基线(去价值网络)、(ii)干预忠实度塑形,并作为
   多智能体系统中 reasoner 的工具闭环接入。
3. **不完美验证器下的复合奖励设计**:违例率制、quote 长度上限、空预测零分
   等反 hacking 机制;讨论可引 noisy-verifier RLVR(arXiv:2510.00915)与
   composite verifiable rewards(arXiv:2509.15557)。

## 5. 实验矩阵

### relations(MAVEN-ERE 证方法;CCKS 金融因果中文落地)

| 系统 | 配置 |
| --- | --- |
| heuristic / LLM 零样本 / SFT | 既有 configs |
| **SFT+GRPO(主,3 seeds)** | `grpo_rlvr.yaml` |
| −R_ground / −R_consist | `ablation_grpo_no_grounding/ no_consistency.yaml` |
| 仅 R_task(裸 RLVR 对照) | 奖励面板只留 task_f1 |
| −curriculum / β=0 | 面板开关 |

指标:`relation_prf`(分族+micro)、`conll_coref_f1`、`consistency_report`、
grounding drop_rate;评测脚本零改动。

### forecasting(ICEWS14/FinDKG 证方法;自建中文事件图落地)

| 系统 | 配置 |
| --- | --- |
| frequency / temporal_gnn | 既有 configs |
| **path_rl(主,3 seeds)** | `path_rl.yaml` |
| random-walk(未训练) | `ablation_path_random_walk.yaml` |
| −shaping / −faithfulness / −warmstart | 对应 ablation yaml |
| −group-baseline(滑动均值) | trainer 开关 |

指标:MRR/Hits@k、`mean_faithfulness`、`faithfulness_aurc`、conformal
coverage(接入 `multiagent_faithful.yaml` 后)。

## 6. 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| GRPO 小批量不稳定 | SFT 热启、lr≤1e-5、KL 锚定、grad clip 0.5、组内 std 下限、课程、3 seeds、阶段 checkpoint |
| Reward hacking | §2.1 防护 + 逐分量曲线监控 + 定期抽查生成样例 |
| TRL–vLLM 版本耦合 | pin `trl>=0.17,<0.19`;上服务器先 50-step 冒烟验证 LoRA→vLLM 权重同步;三级 rollout 兜底 |
| 单卡显存不足 | G 8→4、completion 768→512、colocate sleep mode;最终落 `hf` 后端 |
| 路径 RL 稀疏奖励 | warm-start BC + 势塑形 + entropy bonus + 高重复查询子集起训 |
| 自建图证据质量 | path_rl 只在 verifier admitted 的边上游走(继承现有门控) |

## 7. 里程碑(约 3 个月)

| 周 | 阶段 | 退出标准 |
| --- | --- | --- |
| W1–2 | 本地 CPU:rl 基底 + 两域实现 + 测试 + 配置 | pytest/smoke/ruff 全绿 |
| W3–4 | 服务器:版本冒烟 + easy 桶小规模 GRPO | 奖励四分量同向上升,F1 ≥ SFT |
| W5–7 | GRPO 主训练 ×3 seeds + CCKS 混训 | F1/一致性/接地率增益成立 |
| W6–8 | 路径 RL(ICEWS14→FinDKG→自建图) | MRR 超 frequency,忠实度占优 |
| W9–10 | 双域消融 + 自建中文图闭环 | 消融表完整(mean±std) |
| W11–12 | 成文 + 缓冲 | configs+scripts 一键复现 |

## 8. 代码映射

| 模块 | 内容 |
| --- | --- |
| `src/finekg/rl/` | 域无关基底:`reward`(组合+trace)、`advantage`(组相对优势)、`shaping`(势塑形)、`curriculum`(课程) |
| `src/finekg/relations/rl/` | `rewards`(四分量)、`dataset`(切窗+DocStore)、`trl_adapter`(TRL 签名闭包,不 import trl)、`trainer`(GRPOTrainer 组装,lazy) |
| `src/finekg/forecasting/rl/` | `env`(时序合法路径 MDP)、`episodes`(轨迹/组采样)、`rewards`(命中+忠实度+势)、`policy`(stub + GRU)、`trainer`(组相对 REINFORCE,lazy)、`warmstart`(BC 冷启动) |
| `src/finekg/forecasting/forecaster/path_rl.py` | `@forecasters.register("path_rl")` 推理封装 |
| `scripts/train_relation_grpo.py` / `train_path_rl.py` | 服务器训练入口 |
| `configs/relations/grpo_rlvr*.yaml` / `configs/forecasting/path_rl*.yaml` | 主方法 + 消融面板 |
