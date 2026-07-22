# Datasets

This project keeps original releases under `data/raw/{dataset}` and all
project-ready files under `data/processed/{dataset}`. Regenerate processed data
with:

```bash
uv run python scripts/preprocess_datasets.py
```

Full public datasets are not committed; `data/fixtures/` contains tiny samples
for CPU tests.

## Processed Datasets

| Dataset | Paper/source | Role | Processed outputs |
|---|---|---|---|
| MAVEN-ERE | [EMNLP 2022](https://aclanthology.org/2022.emnlp-main.60/) | event coref/temporal/causal/subevent relations | `data/processed/maven_ere/*.jsonl` |
| MAVEN-Arg | [ACL 2024](https://aclanthology.org/2024.acl-long.224/) | document-level event argument extraction (162 types / 612 roles); same docs as MAVEN-ERE | `data/processed/maven_arg/*.jsonl` |
| MAVEN-FACT | [Findings EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.651/) | event factuality (CT+/CT-/PS+/PS-/Uu) + evidence; same docs as MAVEN-ERE | `data/processed/maven_fact/*.jsonl` |
| CCKS-2021 financial causality | Tianchi/CCKS financial causality release | Chinese financial cause/effect extraction | `data/processed/ccks_fin_causal/*.jsonl` |
| ICEWS14 | TiRGN/TLogic-style ICEWS14 split | TKG forecasting benchmark | `data/processed/icews14/*.tsv` |
| FinDKG | [ICAIF 2024 / arXiv](https://arxiv.org/html/2407.10909v2), [repo](https://github.com/xiaohui-victor-li/FinDKG) | financial TKG forecasting | `data/processed/findkg/*.tsv` |
| CMIN-CN | [ACL 2023](https://aclanthology.org/2023.acl-long.679/) | downstream Chinese stock movement | `data/processed/cmin_cn/manifest.json`, `stocks.tsv` |
| Astock | [FinNLP 2022](https://aclanthology.org/2022.finnlp-1.24/) | stock-specific news movement/trading | `data/processed/astock/{train,val,test,ood}.tsv` |

## Split Policy

| Dataset | Academic/source split | Project split |
|---|---|---|
| MAVEN-ERE | Official train/valid/test; official test labels are hidden | train 2,913; valid 710; test_unlabeled 857; local evaluation should use valid |
| MAVEN-Arg | Official train/valid/test; test labels hidden (CodaLab) | train 2,913; valid 710; test_unlabeled 857 (**identical doc ids to MAVEN-ERE**) |
| MAVEN-FACT | Only train/valid public; official test hidden | train 2,913; valid 710 (**identical doc ids to MAVEN-ERE**); no local test |
| CCKS-2021 | 7,000 labeled train + 1,000 unlabeled eval; no public labeled test | train_all 7,000; seed-42 local train/val/test = 5,600/700/700; eval_unlabeled 1,000 |
| ICEWS14 | Current repo uses the 365-day split common in TLogic/TiRGN-style TKG experiments | train 63,685; valid 13,823; test 13,222; combined 90,730 |
| FinDKG | Official repo split | train 119,549; valid 11,444; test 13,069; combined 144,062 |
| CMIN-CN | Release is organized by 300 CSI300 stocks, news dates, and price series | manifest inventory only; no current project loader |
| Astock | Official train/val/test/ood CSV files | same split normalized to TSV |

ICEWS14 has multiple published preprocessing variants. Do not compare results
against papers using a different ICEWS14 split unless the split statistics match.

## Formats

MAVEN-ERE keeps the official JSONL format. The loader reads `events`,
`temporal_relations`, `causal_relations`, and `subevent_relations` directly.

CCKS JSONL uses:

```json
{"text_id": "...", "text": "...", "causal_pairs": [{"cause": {}, "effect": {}, "cause_span": [], "effect_span": []}]}
```

Temporal KG TSV uses:

```text
subject<TAB>relation<TAB>object<TAB>timestamp
```

Every processed dataset directory contains `manifest.json` with source,
split/count, and reproducibility notes.

## Candidate Extensions

More aligned future datasets worth checking before expanding the benchmark:

| Dataset | Why it may fit | Current decision |
|---|---|---|
| CFTE / DocFEE / CFinDEE-style Chinese financial event extraction datasets | More domain-aligned than MAVEN-ERE for Chinese finance event extraction | list only; not included in current processed outputs |
| TGB `tkgl-icews` | More standardized temporal graph benchmark packaging | list only; current project keeps the existing ICEWS14 split for continuity |

## Licenses

Respect each source license and citation requirement. FinDKG is GPL-3.0. This
repo should only commit fixtures and scripts, not full raw/processed datasets.
