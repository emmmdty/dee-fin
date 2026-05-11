## 数据集
### DuEE-Fin
数据集说明：**DuEE-Fin** 是一个**中文金融新闻文档级事件抽取数据集**，包含约 **1.17 万篇文档、1.5 万+事件、13 类事件、92 类论元角色、8.1 万+事件论元**，支持一文多事件、多值论元和触发词标注等挑战场景。
- 论文及比赛引用：Li, X., et al. (2021) DuEE-fin: a document-level event extraction dataset in the financial domain released by Baidu. Retrieved from https://aistudio.baidu.com/aistudio/competition/detail/46.
- [论文地址](https://link.springer.com/chapter/10.1007/978-3-031-17120-8_14)
- 数据集下载： [DUEE-fin金融事件数据集_数据集-飞桨AI Studio星河社区](https://aistudio.baidu.com/datasetdetail/157875)
- 文件列表（去除 `License.docx` 和 `.DS_Store` 等无关文件）
	- `duee_fin_train.json`
		- 7015
		- 实际为jsonl格式，每一行为一条json
	 - `duee_fin_dev.json`
		- 1171
		- 实际为jsonl格式，每一行为一条json
	 - `duee_fin_test2.json`
		- 与test.json一致，无gold
		- 59394
		- 实际为jsonl格式，每一行为一条json
	- `duee_fin_event_schema.json`
	- `duee-fin 事件类型及对应角色.pdf`
	- `duee_fin_sample.json`
		- 500
		- 实际为jsonl格式，每一行为一条json
- `duee_fin_sample`

```json
{"text": "str，用`\n`换行", 
 "event_list": [
 	{"trigger": "收购", 
 	 "event_type": "企业收购", 
 	 "arguments": [
 	 	{"role": "披露时间", "argument": "晚间"}, 
 	 	{"role": "收购方", "argument": "海螺环境"}, 
 	 	{"role": "被收购方", "argument": "天河（保定）环境工程有限公司"}, 
 	 	{"role": "收购标的", "argument": "100%的股权"}, 
 	 	{"role": "交易金额", "argument": "3.24亿元"}]}], 
 "id": "0939e1d13dcf95f3540e3f3c2a4d1936", 
 "title": "海螺型材：参与收购天河环境 打造新的盈利增长点"}
```

### ChFinAnn
- 数据集说明：**ChFinAnn** 是一个**中文金融公告文档级事件抽取数据集**，包含 **32,040 篇上市公司公告、约 48,000 条事件记录、5 类股权相关事件、35 类论元角色**，其中约 **29%** 文档含多条事件记录，约 **98%** 事件论元跨句分布。
- github： [Doc2EDAG/Data.zip at master · shun-zheng/Doc2EDAG](https://github.com/shun-zheng/Doc2EDAG/blob/master/Data.zip)
	- 含数据集下载
- paper： [Doc2EDAG: An End-to-End Document-level Framework for Chinese Financial Event Extraction - ACL Anthology](https://aclanthology.org/D19-1032/)
- 文件列表（github仓库中的压缩包 `Datz.zip` 解压后）
	- `train.json`
	- `dev.json`
	- `test.json`
	- `sample_train.json`

### DocFEE
- 数据集说明：**DocFEE** 是一个**中文金融公告文档级事件抽取数据集**，包含 **19,044 篇公告文档、9 类事件、38 类论元**，平均文档长度约 **2,277 中文字符**，平均每篇文档 **1.86 个事件**，事件论元跨度平均超过 **960 字符**。
- 数据集下载： [Item - DocFEE: A Document-Level Chinese Financial Event Extraction Dataset - figshare - Figshare](https://figshare.com/articles/dataset/_b_DocFEE_A_Document-Level_Chinese_Financial_Event_Extraction_Dataset_b_/28632464)
- paper： [A dataset for document level Chinese financial event extraction | Scientific Data](https://www.nature.com/articles/s41597-025-05083-9)
- 文件列表
	- `train.jsonl`
	- `test.jsonl`
	- `sample.json`
	- `schema.json`
	- `README.pdf`
- `sample`

```json
{ "content": "str，用`<br>`分段", 
  "doc_id": "2071116", 
  "event_type": "公告篇章事件抽取增持0301", 
  "events": [ 
  		{ "event_id": "4805333", 
  		  "event_type": "股东增持", 
  		  "增持金额": "3503246", 
  		  "增持开始日期": "2019年1月12日", 
  		  "增持的股东": "吴锦华" }, 
  		{ "event_id": "4150985", 
  		  "event_type": "股东增持", 
  		  "增持金额": "4000040", 
  		  "增持开始日期": "", 
  		  "增持的股东": "吴锦华" } 
  	] 
}
```

<!-- BEGIN LOCAL_PROCESSED_SPLITS -->

## 本地处理后数据划分

本节记录由 `scripts/data_split/prepare_all_splits.py` 生成的可复现实验划分。复现命令：

```bash
python scripts/data_split/prepare_all_splits.py --project-root .
```

输出目录：

- `data/processed/ChFinAnn-Doc2EDAG/`
- `data/processed/DuEE-Fin-dev500/`
- `data/processed/DocFEE-dev1000/`

划分摘要：

- **ChFinAnn-Doc2EDAG**：`train` 25632 docs / 38088 events; `dev` 3204 docs / 4987 events; `test` 3204 docs / 4749 events
- **DuEE-Fin-dev500**：`train` 6515 docs / 8824 events; `dev` 500 docs / 674 events; `test` 1171 docs / 1533 events; `blind_test_unlabeled` 59394 docs / 0 events
- **DocFEE-dev1000**：`train` 17244 docs / 32393 events; `dev` 1000 docs / 1846 events; `test` 800 docs / 1236 events

处理原则：

- 不修改 `data/raw/` 下的原始文件。
- 不做去重；重复文本只在 `split_manifest.json` 中作为审计诊断报告。
- 不把 offset/drange 作为规范 gold 角色值抽取的必要条件。
- `ChFinAnn-Doc2EDAG` 保留既有 Doc2EDAG 风格官方 train/dev/test 划分，并在处理目录生成经验验证后的 `schema.json` 与 `schema_validation_report.json`。
- `DuEE-Fin-dev500` 使用 raw dev 作为本地 labeled test，从 raw train 中用 `seed=42` 的确定性分层 hash 方法抽取 500 条 dev，其余作为 train；raw test 仅保留为 `blind_test_unlabeled.jsonl`。
- `DocFEE-dev1000` 保留官方 raw test，从 raw train 中用 `seed=42` 的确定性分层 hash 方法抽取 1000 条 dev，其余作为 train。

重要提醒：

- `DuEE-Fin-dev500` 是公开数据上的确定性 dev500，本地复现目标不是恢复任何隐藏 dev id。
- ChFinAnn 原始 `schema.json` 只作为候选 schema 使用；若需引用处理后 schema，请使用 `data/processed/ChFinAnn-Doc2EDAG/schema.json` 和对应验证报告。
- `DocFEE-dev1000` 跨 split exact-text duplicate 审计：`dev__test`=0，`train__dev`=1，`train__test`=9；整条 JSON hash 对比为 `same_raw_json_line_hash`=0、`same_canonical_json_hash`=0，说明这些是 same-content/different-record 的审计项，不是整条 JSON 样本被复制到多个 split。

<!-- END LOCAL_PROCESSED_SPLITS -->
