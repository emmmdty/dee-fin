# data/raw 数据溯源（v4 主干 + 泛化数据）

> 更新于 2026-07-22。本文件是 `data/raw/` 下唯一纳入 Git 的文件；完整数据继续忽略。MAVEN-ERE、
> MAVEN-Arg、MAVEN-FACT 的 train/valid doc-id 集合已核对一致（train 2913、valid 710；ERE/Arg
> 另有 857 条无标签 test）。官方隐藏标签不进入本地调参。

## MAVEN 主干

| 数据集 | raw 路径 | 来源 | 本地公开切分 | 关键标注 | WSL/4090 |
|---|---|---|---|---|---|
| MAVEN-ERE | `maven_ere/` | THU-KEG release | 2913/710/857-unlabeled | events+coref+temporal+causal+subevent | ✅/✅ |
| MAVEN-Arg | `maven_arg/` | THU Cloud release | 2913/710/857-unlabeled | 论元、实体、negative triggers | ✅/✅ |
| MAVEN-FACT | `maven_fact/` | 作者 Google Drive release | 2913/710/— | CT+/CT-/PS+/PS-/Uu、evidence、论元、关系 | ✅/✅ |

MAVEN-FACT `valid.jsonl` 已于 2026-07-22 获取、处理并同步；旧文档中的“valid 待作者下载”已经失效。

## 泛化数据边界

| 数据集 | raw 状态 | processed 状态 |
|---|---|---|
| MATRES | WSL/4090 ✅ | 未实现 |
| RAMS | WSL/4090 ✅ | 未实现 |
| WikiEvents | WSL/4090 ✅ | 未实现 |
| ECB+ | WSL/4090 ✅ | 未实现 |
| CLES | 未获取 | 未实现 |
| DocEE | WSL/4090 ✅ | en 双端；zh/cross 仅 WSL |

raw 存在不代表 loader、切分和评测已经可用。只有 processed manifest 与对应代码同时完成，才可在
进度文档中标记“已预处理”。

## 应用层 / 兼容数据

- `sedgpl_esc/ESCSubWoRe.npy`：Ch4 ESC/topic-CV。
- `ccks_fin_causal/`：Phase G 中文金融因果。
- `event_graph_zh/`、Astock、CMIN-CN：历史/应用产物，不作 v4 主数据。
- ICEWS/FinDKG：冻结 TKG 线兼容数据，不进入 v4 主表。

## SHA-256

```text
a94f92c6ab509b58a8cab026d1984cf0b7095855a23e01f7e7d9c804e39a4761  maven_arg/train.jsonl
e68902654cf54217695539b6a9afe3d18322754e557cde253546f7e34bf36b73  maven_arg/valid.jsonl
6fbdd6150252ccd53b0496406ccd357c9af79bebf76a3bb19d1d6c7198b5f213  maven_arg/test.jsonl
190522b44f0702af030161924d7cb94c4a06bd5d6e2b40d79f8f1eaa5886bab7  maven_fact/train.jsonl
396fcf0779b67f0229f2cdaad4df0771682d9238a94b082d61659059b8dc7cff  maven_fact/valid.jsonl
```
