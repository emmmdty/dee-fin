| Method | MRR | Hits@1 | Hits@10 | Note |
|---|---|---|---|---|
| Frequency | 0.105 | 0.044 | 0.220 | naive baseline |
| Temporal-GNN | 0.286 | 0.192 | 0.467 | simplified neural baseline |
| Recurrency | 0.356 | 0.283 | 0.488 | CPU copy baseline |
| Path-RL | 0.360 | 0.284 | 0.494 | path policy |
| RE-GCN | 0.380 | 0.286 | 0.565 | 时间感知过滤 |
| 融合搜索 | **0.411** | 0.309 | 0.609 | ws=1.0, wc=0.3 |
