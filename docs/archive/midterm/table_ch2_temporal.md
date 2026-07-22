| model | temporal P | temporal R | temporal F1 |
|---|---|---|---|
| SFT | **0.445** | 0.005 | 0.010 |
| GRPO-RLVR | **0.565** | 0.025 | 0.048 |

> Headline temporal signal is **precision** (ordering accuracy on predicted pairs). Recall/F1 are not comparable — MAVEN-ERE temporal gold is a transitive closure (dense, n_gold≈110k) while the extractor is sparse by design. Closure-aware scoring (`relation_prf(temporal_closure=True)`) is implemented and unit-tested, but is a **no-op** on the current model: its BEFORE predictions don't chain, so closing them adds 0 edges. A denser / transitively-structured extractor would benefit.
