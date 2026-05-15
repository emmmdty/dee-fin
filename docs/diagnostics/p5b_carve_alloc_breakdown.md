# P5b CARVE Allocation Failure Diagnostic

Run: `/home/tjk/myProjects/masterProjects/DEE/dee-fin/runs/carve/p5b_duee_fin_dev500_seed42_r3typegate`

## Verdict

**Dominant failure: noise records (record-count over-prediction).** Sinkhorn is allocating to N record columns where most should be empty. The lexical record_count fallback overestimates.

## Summary

- TP arguments: **102**  
- Total FP arguments: **1,107**  
- Argument-level precision: **0.0844**

### FP breakdown

| Category | Args | Share of FP |
|---|---:|---:|
| Noise records (unmatched pred records, all args FP) | 880 | 79.5% |
| Misallocation (matched record, wrong (role,value)) | 227 | 20.5% |
| Share-gate excess (overlapping subset of above) | 0 | 0.0% |

Noise record count: **297**

Note: noise records and misallocation FPs partition the total FP space. 
Share-gate excess is an *additional* attribution that may overlap with either category — 
any duplicated value in an unmatched record is counted in both noise_fp and share_gate_excess.

### FN breakdown

| Category | Count |
|---|---:|
| Gold value in doc text, missing from pred | 1,824 |
| Gold value not in doc text (pathological) | 47 |

## Per-event-type breakdown (sorted by total FP args)

| Event Type | Pred recs | Gold recs | Matched | TP args | Noise FP | Misalloc FP | Share excess | Cand. miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 质押 | 114 | 68 | 30 | 34 | 284 | 101 | 0 | 443 |
| 解除质押 | 94 | 45 | 26 | 35 | 215 | 66 | 0 | 258 |
| 股东减持 | 34 | 29 | 2 | 2 | 139 | 10 | 0 | 155 |
| 股份回购 | 21 | 41 | 4 | 5 | 54 | 10 | 0 | 205 |
| 亏损 | 31 | 26 | 6 | 6 | 40 | 9 | 0 | 85 |
| 高管变动 | 22 | 47 | 7 | 9 | 33 | 12 | 0 | 230 |
| 股东增持 | 8 | 16 | 1 | 1 | 38 | 5 | 0 | 74 |
| 中标 | 16 | 21 | 6 | 8 | 16 | 10 | 0 | 83 |
| 企业收购 | 18 | 27 | 0 | 0 | 25 | 0 | 0 | 104 |
| 公司上市 | 8 | 22 | 0 | 0 | 20 | 0 | 0 | 55 |
| 企业融资 | 8 | 10 | 1 | 1 | 9 | 2 | 0 | 34 |
| 企业破产 | 5 | 6 | 0 | 0 | 6 | 0 | 0 | 13 |
| 被约谈 | 2 | 24 | 1 | 1 | 1 | 2 | 0 | 85 |

## Recommended Next Step

- Improve the record-count estimator before re-running P5b. The lexical fallback (`_estimate_record_count`) is currently the bottleneck. Options:
  - Use R3 v5's coref count (requires inference-time mention extraction)
  - Cap record_count at min(lexical_estimate, k_clip=16) — likely already done
  - Run a lightweight regression on document features to predict record_count
