# 数据集整理与获取清单(v4 · 2026-07-22)

> 目的:为四章各配"主库 + 广用有基线的第二/泛化库",全程**免费公开可下载**。
> 选择标准(据作者):**优先"别人也在用、能找到 baseline 对比"的**;新旧不卡死,**2018+ 即可**(比 2009 新);
> 免费公开;下载易得。获取工作流:**易下的直接下 → 难下/云盘的记录待作者下 → 上传服务器打包再传**。
> 决策图例:✅ 已在库 ｜ ⬇️ 现下(易得) ｜ 📝 记录待作者下(云盘/授权/工具链) ｜ 💤 备选不急。

## Ch1 — 事件节点(检测 / 论元 / 共指)

| 数据集 | 年 | 任务 | 语言 | 规模 | 许可/免费 | 广用/基线 | 下载 | 决策 |
|---|---|---|---|---|---|---|---|---|
| MAVEN | 2020 | 事件检测 | en | 4480 文档 | 免费(THU-KEG) | ✓✓ | 已在 maven_ere | ✅ |
| MAVEN-Arg | 2024 | 论元(162类/612角色) | en | 29万论元 | 免费 | ✓ | 已下 | ✅ |
| **RAMS** | 2020 | 文档级论元 | en | 9124 例/139类 | **Apache-2.0** | ✓✓ | nlp.jhu.edu/rams(tar) | ⬇️ |
| **WikiEvents** | 2021 | 文档级论元 | en | 50类/3.9k 事件 | **MIT** | ✓✓ | github raspberryice/gen-arg | ⬇️ |
| DocEE | 2022 | 文档级 EE(最大) | en+zh | 27k+ 文档 | 免费 | ✓ | github tongmeihan1995/DocEE(**数据走 Google Drive**) | 📝 |
| **ECB+** | 2014 | 跨文档事件共指(**标准库**) | en | 982 文档 | 免费 | ✓✓(基线最多) | github(cltl/ecbPlus, uCDCR) | ⬇️ |
| **GVC**(Gun Violence) | 2018 | 跨文档共指 | en | 510 文档 | 免费 | ✓ | uCDCR 统一包 | ⬇️ |
| WEC-Eng | 2021 | 大规模跨文档共指 | en | Wikipedia 派生 | 免费 | ✓ | github AlonEirew/extract-wec(或 HF) | 💤 |

## Ch2 — 事件关系(时序 / 因果 / 子事件)

| 数据集 | 年 | 任务 | 语言 | 许可/免费 | 广用/基线 | 下载 | 决策 |
|---|---|---|---|---|---|---|---|
| MAVEN-ERE | 2022 | coref+时序+因果+子事件 | en | 免费 | ✓✓ | 已在库 | ✅ |
| **MATRES** | 2018 | **时序关系(标准基准)** | en | **免费** | ✓✓(四大时序库之一,基线极多) | github qiangning/MATRES | ⬇️ |
| TORQUE | 2020 | 时序阅读理解 | en | 免费 | ✓ | github qiangning/TORQUE | 💤 |
| EventStoryLine | 2017 | 因果(篇章级) | en | 免费 | ✓✓ | github(已派生 ESC 在库) | ✅(ESC) |
| Causal-TimeBank | 2014 | 因果 | en | 免费* | ✓ | github | 💤 |
| CCKS-FinCausal | 2021 | 中文金融因果 | zh | 免费 | ✓ | 已在库 | ✅ |

## Ch3 — 事实性

| 数据集 | 年 | 任务 | 语言 | 许可/免费 | 广用/基线 | 下载 | 决策 |
|---|---|---|---|---|---|---|---|
| MAVEN-FACT | 2024 | 事实性 5 类 + 证据(最大) | en | 免费 | ✓ | 已下 | ✅(主) |
| **UDS-IH2 / It-Happened** | 2018 | 事实性(连续值,**标准基准**) | en | **免费** | ✓✓(基线多) | 已下+处理(`it_happened`,44315/5286/5099) | ✅ |
| ModaFact | 2025 | 事实性+情态(意语) | it | CC-BY-SA-4.0 | 新(基线少) | 已下+处理(HF dhfbk/modafact-ita) | ✅(跨语言 bonus) |
| ~~FactBank~~ | 2009 | 事实性(经典) | en | LDC 授权 | ✓✓ | LDC2009T23 | ❌ 弃用(无 LDC 会员) |

## Ch4 — 后继事件预测(CGEP)

| 数据集 | 年 | 任务 | 语言 | 下载 | 决策 |
|---|---|---|---|---|---|
| CGEP-MAVEN | 重建 | 事件因果图后继预测 | en | scripts/build_cgep.py | ✅ |
| CGEP-ESC | 2017 派生 | 后继预测(topic-CV) | en | 已在 sedgpl_esc | ✅ |

## 获取计划(进度 2026-07-22)

- **✅ 已下 + 已预处理 + 已上传 4090**(`preprocess_datasets.py --only <name>` 可复现,manifest 带 `source_sha256`):
  - Ch2 **MATRES**(13,577 时序关系)· Ch1 **RAMS**(9124)/ **WikiEvents**(246 文档)/ **ECB+**(29M 共指语料)。
  - Ch3 **MAVEN-FACT**(主)/ **ModaFact**(意语跨语言)/ **UDS-IH2 It-Happened**(2018 标准基准,44315/5286/5099)。
  - Ch1 **DocEE**:en normal(21966/2748/2771)已上传;**zh(36729)+ en_cross_domain 已本地处理、留 Ch1 用到时再传**(避免此刻 350M+ 过隧道)。
  - MAVEN-Arg / MAVEN-FACT / maven_ere 早已上传。
- **❌ 弃用**:**FactBank**(无 LDC 会员)· **ACE05**(LDC,论元已够,不用)。
- **💤 备选不急**:**GVC**(uCDCR 需 GVC 原始语料;ECB+ 已作主力共指库)· WEC-Eng。
- **✅ 本就在库**:MAVEN 四件套、ESC、CCKS-FinCausal、CGEP(重建)。
- **上传法**:raw/processed **tar 打包 → rsync --partial**(隧道抗断)+ 远端解包 + 双端 sha256;大件(docee zh/cross)按需再传。

## 为什么这样选(对齐作者标准)
优先 MATRES / RAMS / WikiEvents / ECB+ / UDS-IH2 —— 都是 **2018+ 或标准基准、被大量论文使用、能直接找到已发表 baseline 对比**;
ModaFact(2025)保留但降为**跨语言 bonus**(太新、基线少)。**FactBank 弃用**(LDC 无会员);**UDS-IH2(2018)作 Ch3 英文第二标准库**(广用有基线)。
