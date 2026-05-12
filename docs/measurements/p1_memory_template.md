# P1 Memory Measurement Report Template

> **Status**: template. Fill after P1 is executed.  
> **Purpose**: record measured GPU memory for CARVE encoder configurations under the local 1–2×RTX 4090 constraint.

## Environment

| Item | Value |
|---|---|
| Date | TBD |
| Host | TBD |
| GPU model(s) | TBD |
| CUDA version | TBD |
| Python environment | TBD |
| PyTorch version | TBD |
| Transformers version | TBD |
| Repository commit | TBD |
| Dataset split commit / hash | TBD |

## Measurement Protocol

- No paper claim may use estimated memory numbers.
- Measure peak allocated and reserved GPU memory.
- Record batch size, sequence length, gradient checkpointing, precision, and optimizer.
- If a run OOMs, record the exact command and failure point.

## Encoder Measurements

| Dataset | Encoder | Seq/window | Batch | Grad checkpointing | Precision | Peak allocated | Peak reserved | Status | Notes |
|---|---|---:|---:|---|---|---:|---:|---|---|
| ChFinAnn | Chinese-RoBERTa | 512 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DuEE-Fin | Chinese-RoBERTa | 512 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DocFEE | Lawformer | 4096 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DocFEE | Longformer-Chinese | 4096 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DocFEE | RoBERTa-window + role memory | 512 windows | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Fallback Decision

If DocFEE encoder + CARVE modules exceed 22 GB on one RTX 4090, apply:

1. F1 — gradient checkpointing + smaller batch + gradient accumulation
2. F2 — frozen encoder + cached sentence representations
3. F3 — 512-window Chinese-RoBERTa + role-summary memory
4. F4 — compare measured Lawformer vs Longformer-Chinese memory

| Fallback level used | Reason | Evidence |
|---|---|---|
| TBD | TBD | TBD |

## Commands

```bash
# paste commands here
```

## Notes

TBD
