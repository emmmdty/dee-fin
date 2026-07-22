| Split | Factor | Setting | F1 | P | R | F1(S) | F1(M) | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | SFT | HF-4bin no-SFT | 3.3 | 44.8 | 1.7 | 3.7 | 3.0 | HF no-SFT test baseline completed after earlier docs were written. |
| test | SFT | vLLM-bf16 no-SFT | 11.3 | 37.2 | 6.7 | 13.8 | 9.4 | vLLM no-SFT test baseline. |
| test | backend | HF-4bin + LoRA k1 | 78.0 | 76.6 | 79.3 | 79.3 | 77.5 | main DuEE-Fin backend |
| test | backend | vLLM-bf16 + LoRA k1 | 75.0 | 74.8 | 75.2 | 78.6 | 73.2 | backend cross-check |
| test | decoding | vLLM-bf16 + LoRA k4 T0.7 | 73.1 | 69.2 | 77.5 | 75.8 | 72.5 | Sampling decoding ablation. |
| test | LRD | no-LRD | 78.0 | 76.6 | 79.3 | 79.3 | 77.5 | Primary DuEE-Fin test result; no-LRD remains the main path. |
| test | LRD | safe-anchor tau=0.90 | 78.0 | 76.7 | 79.3 | 79.4 | 77.5 | LRD diagnostic; gain is negligible and it is not the main method. |
| dev | LRD | invalid k4-pool diagnostic | 33.5 | 21.3 | 78.6 | 31.7 | 35.5 | not comparable; FP explosion from candidate-pool misuse |
