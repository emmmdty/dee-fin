# 数据集整理与获取清单（v4 · 2026-07-22）

> 目标：每章配置“主数据 + 有公开基线的泛化数据”。状态必须区分：
> **✅ processed 可用**｜**📦 raw 已下载、待预处理**｜**📝 未获取/需人工**｜**💤 备选**｜**❌ 弃用**。

## Ch1 — 事件节点（检测 / 论元 / 共指）

| 数据集 | 年 | 任务 | 语言 | 公开性 | 状态 | 决策 |
|---|---:|---|---|---|---|---|
| MAVEN | 2020 | 事件检测 | en | 免费 | ✅ WSL/4090 processed | 主数据 |
| MAVEN-Arg | 2024 | 论元（162 类/612 角色） | en | 免费 | ✅ WSL/4090 processed | 主数据 |
| RAMS | 2020 | 文档级论元 | en | Apache-2.0 | 📦 WSL/4090 raw | Phase C 前补预处理 |
| WikiEvents | 2021 | 文档级论元 | en | MIT | 📦 WSL/4090 raw | Phase C 前补预处理 |
| DocEE | 2022 | 长文档 EE | en+zh | 免费 | ✅ en；zh/cross 仅 WSL processed | 泛化数据 |
| ECB+ | 2014 | 跨文档事件共指 | en | 免费 | 📦 WSL/4090 raw | Phase C2 主数据 |
| CLES | 2024 | 中文跨文档事件抽取 | zh | 公开仓库 | 📝 未获取 | Phase C2 泛化 |
| GVC / WEC-Eng | 2018/2021 | 跨文档共指 | en | 免费 | 💤 | 不急 |

## Ch2 — 事件关系（时序 / 因果 / 子事件）

| 数据集 | 年 | 任务 | 语言 | 状态 | 决策 |
|---|---:|---|---|---|---|
| MAVEN-ERE | 2022 | coref+时序+因果+子事件 | en | ✅ WSL/4090 processed | 主数据 |
| MATRES | 2018 | 时序关系 | en | 📦 WSL/4090 raw | 泛化；待预处理 |
| ESC / EventStoryLine | 2017 | 因果/后继预测 | en | ✅ raw + CGEP loader | Ch4 副数据 |
| CCKS-FinCausal | 2021 | 中文金融因果 | zh | ✅ WSL/4090 processed | Phase G |
| TORQUE / Causal-TimeBank | 2020/2014 | 时序/因果 | en | 💤 | 不急 |

## Ch3 — 事实性

| 数据集 | 年 | 任务 | 语言 | 状态 | 决策 |
|---|---:|---|---|---|---|
| MAVEN-FACT | 2024 | 5 类事实性 + evidence | en | ✅ train/valid WSL/4090 processed | 主数据 |
| UDS-IH2 / It-Happened | 2018 | 连续事实性 | en | ✅ WSL/4090 processed | 第二标准库 |
| ModaFact | 2025 | 事实性+情态 | it | ✅ WSL/4090 processed | 跨语言 bonus |
| FactBank | 2009 | 事实性 | en | ❌ LDC 授权 | 弃用 |

## Ch4 — 后继事件预测

| 数据集 | 来源 | 状态 | 决策 |
|---|---|---|---|
| CGEP-MAVEN | 从 MAVEN-ERE 重建 | ✅ loader/基线/现有结果 | 主数据 |
| CGEP-ESC | EventStoryLine 派生 | ✅ `ESCSubWoRe.npy` | topic-CV 副数据 |

## 当前获取与同步事实

- **processed + manifest 已在 WSL/4090**：MAVEN-ERE、MAVEN-Arg、MAVEN-FACT、DocEE en、
  It-Happened、ModaFact，以及金融/旧线已有数据。
- **raw 已在 WSL/4090，但没有项目 processed 输出**：MATRES（约 6.1M）、RAMS（约 20M）、
  WikiEvents（约 15M）、ECB+（约 29M）。`preprocess_datasets.py` 当前没有这四个入口。
- **仅 WSL processed，按需再上传**：DocEE zh（36729）和 en cross-domain，避免当前额外传输 350M+。
- **未获取**：CLES；等 Phase C2 设计冻结后再取，避免继续无边界扩数据集。

## 选择理由

MAVEN 四件套提供同文档垂直主链；RAMS/WikiEvents/ECB+/MATRES/UDS-IH2 都有成熟公开基线，适合
做泛化。ModaFact 太新，只作 bonus。FactBank/ACE05 因 LDC 权限弃用。数据主链已经足够，当前优先级是
Phase A/B/C 的模型与评测，不是继续下载数据。
