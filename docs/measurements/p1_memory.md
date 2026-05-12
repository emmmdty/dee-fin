# P1 Memory Measurement

> Status: measured evidence from P1. Do not edit manually without rerunning the command.

- Host: `ubuntu`
- Git commit: `3b5338790c3844e2e52a90776f32d99983e02dc8`
- Python: `/home/TJK/.conda/envs/tjk-feg/bin/python`

| Model | Status | Seq Len | Batch | Peak Allocated GB | Peak Reserved GB |
|---|---:|---:|---:|---:|---:|
| chinese-roberta-wwm-ext | ok | 512 | 1 | 0.407 | 0.455 |
| lawformer | ok | 512 | 1 | 0.531 | 0.945 |
| longformer-chinese | ok | 512 | 1 | 0.419 | 0.965 |
