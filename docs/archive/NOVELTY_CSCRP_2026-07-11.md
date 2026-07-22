# A2 新颖性复核：CS-CRP（跨阶段 conformal 风险传播）2026-07-11

> 承接 `docs/HANDOFF_2026-07-11.md`（CS-CRP=ch3 主打机制）与 `docs/NOVELTY_A1_2026-07-11.md`。
> 方法：`phd-skills:gaps` + 定向 arXiv 精读。CS-CRP 确切命题取自 `core/calibration/propagation.py`
> 与 `docs/patent/交底书.md` 模块六。

## 0. 结论（TL;DR）

**一般命题非新，具体组合未见先例，比 RL-奖励线干净。**

- ❌ 「首个 cross-stage / selective / pipeline conformal（甚至 under drift）」**不成立**：
  C-RAG(2402.03181, 2024 已做 pipeline conformal risk **under distribution shift**)、
  PASC(2605.18812)、SCRC(2512.12844)、CASCADE(2605.20468)、RAG conformal filtering(2511.17908) 已占。
- ✅ CS-CRP 的**确切组合未见先例**（置信度 MEDIUM）：构建阶段**召回**保证(CRC 边准入 FNR≤α_e)
  ⊗ 推理阶段**漂移自适应覆盖**保证 + **上游剪枝致金标不可达的 reachability 预算**(union bound)
  + **条件预算回收** α_pred'=(α_total−u)/(1−u)，实例化于事件因果图构建→后继推理(CGEP)。
- ✅ **印证「主打 CS-CRP、RL-奖励降级」**：RL-奖励被 MedCEG 直接抢先，CS-CRP 有真 delta。
- ⚠️ **撞名**：CS-CRP(Cross-Stage Conformal Risk Propagation) ≈ **SCRC**(Selective Conformal Risk
  Control, 2512.12844)，缩写与概念空间高度重叠——**建议改名**（见 §4）。

## 1. CS-CRP 的确切命题（复核对象）

把两个异质保证在单一预算 α_total=α_e+α_p 下组合成端到端选择性预测器：
1. 构建阶段：CRC 边准入，**召回**保证 FNR≤α_e（`relations/admission.py`）。
2. 推理阶段：**漂移自适应覆盖**保证 miss≤α_p（`core/calibration` 的 ACI/weighted 流式校准器）。
3. **关键**：边准入**移除候选**→ 丢掉金标边使答案**不可达**（推理校准器根本看不到的 miss）。
   为此单列 reachability 预算，union bound：P(miss)≤P(unreachable)+P(reason miss|reachable)≤α_e+α_p。
4. **条件回收**（`allocate_budget_conditional`）：用 held-out 准入结果证一个不可达率上界 u
   （Clopper-Pearson，CRC 界收紧），推理侧跑修正水平 α_p'=(α_total−u)/(1−u)，收紧预测集。
5. 推理侧**非可交换**（漂移自适应），区别于可交换 pipeline-CP。

## 2. 近邻逐条对比（关键差异表）

| 论文 | arXiv | 跨阶段组合 | 异质 recall⊗coverage | 上游剪枝致不可达+预算 | 漂移形式保证 | 条件回收 α'=(α−u)/(1−u) | 域 |
|---|---|---|---|---|---|---|---|
| **PASC** | 2605.18812 | ✔(联合覆盖) | �’(同质，单标量 max-nonconf) | ✘ | ✘(仅经验鲁棒) | ✘ | 多阶段 NLP/LLM |
| **SCRC** | 2512.12844 | ✔(选择+CRC) | ✘(同质 CRC on subset) | ✘(**样本级弃权**，非候选剪枝) | ✘(靠对称选择**保**可交换) | ✘(用积 α·ξ_LCB) | 医学影像/分类 |
| **CASCADE** | 2605.20468 | ✔(不确定性传播) | 异质**目标**(分类→回归) | ✘(只放宽区间宽度) | ✘ | ✘ | 临床(PD 剂量) |
| **C-RAG** | 2402.03181 | ✔(检索+生成) | ✘(证生成风险，非 recall⊗coverage) | ✘ | **✔(under shift)** | ✘ | RAG QA |
| **RAG conformal filter** | 2511.17908 | ✘(**单阶段**过滤) | ✘(仅检索**召回**保证) | ✘(明说不算丢证据代价) | ✘(要可交换，列 future work) | ✘ | RAG 上下文 |
| **Decomp. modular CP** | 2510.04406 | ✔(两阶段归因) | ✘(残差分解) | ✘ | ✘ | ✘ | 两阶段回归 |
| **CS-CRP（我们）** | — | ✔ | **✔** | **✔** | **✔** | **✔** | 事件因果图 CGEP |

**读表**：每篇都缺我们最右三格中的至少两格；无一篇同时做「recall⊗coverage 异质组合 + reachability
预算 + 漂移自适应 + 条件回收」。最独特、最硬的一格是**「上游剪枝致金标不可达」的预算化**——
SCRC(样本弃权)/RAG-filter(单阶段)/CASCADE(放宽区间)/PASC(同质)/C-RAG(证生成风险) 全无。

## 3. 存活 delta（scoped 主张，供写作参考，非最终措辞）

> 「据我们所知，首个将**构建阶段召回保证（CRC 边准入 FNR≤α_e）**与**推理阶段漂移自适应覆盖保证**
> 组合、并显式为**上游剪枝致金标不可达**留预算/条件回收的端到端选择性预测器，实例化于**事件因果图
> 构建→后继事件推理（CGEP）**。」——相关工作**必须逐条区分** C-RAG / PASC / SCRC / CASCADE。

## 4. ⚠️ 改名建议（不在本轮执行）

CS-CRP 与 **SCRC（2512.12844）** 缩写同族、概念邻近，reviewer 极易混淆。建议改名，突出真正独特的
机制（reachability-budgeted / recall-coverage composition），而非「cross-stage」「selective」这类已被占的词。
涉及 `docs/*`、`core/calibration/propagation.py`、`docs/patent/交底书.md` 多处 + §9 命名纪律 →
**待作者拍板后统一改**，本轮只记录风险。

## 5. 置信度与未尽

- 「一般 cross-stage/selective conformal 非新」= **HIGH**（≥7 措辞、C-RAG 早至 2024、SCRC/PASC/CASCADE 齐备）。
- 「CS-CRP 具体组合未占」= **MEDIUM**（领域 2025-2026 高速演进；SAFER 2510.10193 等未精读；投稿前应做一次
  穷尽 pipeline-CP / conformal-risk-composition 扫）。
- 论文相关工作：C-RAG(2402.03181)/SCRC(2512.12844) 已并入 §2 对比表；写作时以 **reachability 预算**
  为最硬差异点。（专利内容已归档 `docs/archive/patent/`，本课题不再安排专利写作。）
