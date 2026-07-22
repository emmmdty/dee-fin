| Split | Factor | Setting | F1 | P | R | F1(S) | F1(M) | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | SFT | vLLM-bf16 no-SFT | 24.8 | 61.2 | 15.6 | 38.9 | 6.8 | No-SFT test baseline. |
| test | backend | HF-4bin + LoRA k1 | 86.0 | 84.4 | 87.7 | 89.9 | 81.8 | main ChFinAnn backend |
| test | backend | vLLM-bf16 + LoRA k1 | 85.5 | 84.1 | 86.9 | 89.8 | 80.8 | backend cross-check |
| test | decoding | vLLM-bf16 + LoRA k4 T0.7 | 84.2 | 79.5 | 89.5 | 87.5 | 80.7 | Sampling decoding ablation. |
