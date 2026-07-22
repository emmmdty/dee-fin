> ⚠️ **主线已更新（2026-07-11）**：以 `docs/HANDOFF_2026-07-11.md` 为准。**代码↔功能的映射仍有效**，
> 但 forecasting 一节的主任务已从 TKG 外推（ICEWS/FinDKG MRR）改为 **CGEP**（基座 SeDGPL，
> 代码在 `src/finekg/succession/`）；re_gcn / path_rl / hybrid 那条 TKG 线**冻结为消融附录**。

# Research map (chapters ↔ code)

Code is organized by **function**, not by thesis chapter — package and function
names never contain `ch2`/`ch3`. This file is the only place the mapping lives,
so renaming chapters never touches code.

> ⚠️ The confirmed thesis spine is the **3-chapter** line (extraction → relations →
> reasoning). The old `Fin-EKG-legacy` **4-chapter** plan and its OG-LANS narrative are
> **obsolete — do not follow them**. Chapter 1 results are summarized in `docs/chapter1/`.
>
> 🧭 顶层三章统一设计见 `docs/HANDOFF_2026-07-11.md`（单一权威；旧 `THESIS_DESIGN.md` 已归档
> 至 `docs/archive/`）。SARGE（第一章）以 git subtree 迁入 `external/sarge/`；Fin-EKG 运行期仍
> 仅通过冻结契约 `core.io.event_nodes_from_sarge` 消费其 canonical 输出。**CS-CRP**（漂移鲁棒
> 跨阶段 conformal 风险传播）现为 ch3 CGEP 的 M3 head，见 HANDOFF §3 与 `RISK_CONTROL_DESIGN.md`。

| Thesis stage | Code (domain-named) | Public benchmark (method) | Chinese landing (no new labels) |
|---|---|---|---|
| Event records = **nodes** (Chapter 1, SARGE, upstream) | `core.io.event_nodes_from_sarge` | — | DocFEE / DuEE-Fin / ChFinAnn |
| Event relations & graph = **edges** | `finekg.relations` | MAVEN-ERE (coref/temporal/causal/subevent) | CCKS financial causality |
| Temporal reasoning & forecasting | `finekg.forecasting` | FinDKG / ICEWS14 (MRR/Hits) | self-built event graph + CMIN-CN/Astock |

Cross-cutting property (the thesis identity): **evidence-grounded / verifiable**
— `evidence` on every edge, `evidence_chain` on every prediction.

## Experiment configs

Relation stage (`configs/relations/`):
- `heuristic_baseline.yaml` — torch-free lower bound (CPU)
- `llm_grounded_consistent.yaml` — main method (LLM + grounding + consistency)
- `ablation_no_grounding.yaml` — `require_evidence: false`
- `ablation_no_consistency.yaml` — `consistency: identity`
- `ccks_causal_zh.yaml` — Chinese financial causality landing
- `grpo_rlvr.yaml` — GRPO post-training with the composite verifiable reward
- `ablation_grpo_no_grounding.yaml` / `ablation_grpo_no_consistency.yaml` — reward ablations

Forecasting stage (`configs/forecasting/`):
- `frequency_baseline.yaml` — torch-free recurrence baseline (CPU)
- `temporal_gnn.yaml` — neural forecaster (PyG, server)
- `self_built_event_graph.yaml` — forecasting over our own event graph (loop closure)
- `path_rl.yaml` — path-RL forecaster (group-relative REINFORCE + faithfulness shaping)
- `ablation_path_no_shaping.yaml` / `ablation_path_no_faithfulness.yaml` /
  `ablation_path_no_warmstart.yaml` / `ablation_path_random_walk.yaml` — path-RL ablations

## Reproduce a result

```bash
# relation stage (CPU baseline today; swap config for the LLM method on the server)
uv run python scripts/evaluate_relations.py --config configs/relations/heuristic_baseline.yaml \
    --path data/processed/maven_ere/valid.jsonl

# build the financial event graph, then forecast over it (loop closure)
uv run python scripts/build_event_graph.py \
    --nodes data/raw/event_graph_zh/event_nodes.jsonl \
    --output data/processed/event_graph_zh/event_graph.json
uv run python scripts/evaluate_forecasting.py --config configs/forecasting/self_built_event_graph.yaml
```

## Multi-agent upgrade (the `agents` domain)

Both stages have an agentic upgrade that keeps the same inputs/outputs but builds
the result with a society of agents, gated by a shared **faithfulness verifier**
(the cross-stage "currency"). Function-named — no chapter markers leak into code.

| Thesis stage | Multi-agent pipeline | Agents (CPU stub → server) | New configs |
|---|---|---|---|
| edges | `MultiAgentRelationPipeline` | proposer committee → grounding-verifier (per-edge faithfulness + abstention) → consistency-arbiter | `relations/multiagent_grounded.yaml`, `relations/ablation_no_verifier.yaml` |
| reasoning | `MultiAgentForecastingPipeline` | retrieval → reasoner committee → faithfulness-verifier (counterfactual intervention) → calibrator (conformal coverage + faithfulness abstention) | `forecasting/multiagent_faithful.yaml` |

New metrics (`finekg.core.eval.faithfulness`, `finekg.core.calibration`): edge/path
intervention faithfulness, risk-coverage / AURC, ECE, split-conformal coverage.

Run them (CPU stub on fixtures; swap `extractor: llm` / a `temporal_gnn` reasoner on the server):

```bash
uv run python scripts/run_agent_pipeline.py --stage relations \
    --config configs/relations/multiagent_grounded.yaml --path data/fixtures/maven_ere/sample.jsonl
uv run python scripts/run_agent_pipeline.py --stage forecasting \
    --config configs/forecasting/multiagent_faithful.yaml --path data/fixtures/findkg/sample.tsv
uv run python scripts/evaluate_faithfulness.py \
    --config configs/forecasting/multiagent_faithful.yaml --path data/fixtures/findkg/sample.tsv
```

## Verifier-as-reward RL upgrade (the `rl` domains)

Design doc: `docs/RL_DESIGN.md`. The verifiers above stop being inference-time
gates only and become *training* rewards — same kernels, two roles:

| Thesis stage | RL method | Reward (verifier kernels) | Train / evaluate |
|---|---|---|---|
| edges | GRPO post-training of the LLM extractor (`finekg.relations.rl`, TRL + LoRA) | format + grounding + consistency + gold F1 | `scripts/train_relation_grpo.py` → adapter into `llm_grounded_consistent.yaml` |
| reasoning | path RL over the temporal graph (`finekg.forecasting.rl`, `@forecasters.register("path_rl")`) | hit + intervention-faithfulness bonus + potential shaping | `scripts/train_path_rl.py` → `evaluate_forecasting.py` with `path_rl.yaml` |

```bash
# server: GRPO with verifiable rewards (vllm rollout on GPU 0, training on GPU 1)
CUDA_VISIBLE_DEVICES=0 uv run --extra serve --extra rl trl vllm-serve --model Qwen/Qwen3-4B-Instruct-2507
CUDA_VISIBLE_DEVICES=1 uv run --extra llm --extra rl python scripts/train_relation_grpo.py \
    --config configs/relations/grpo_rlvr.yaml --output runs/relation_grpo

# server (or CPU for small graphs): faithfulness-shaped path RL
uv run --extra gnn --extra rl python scripts/train_path_rl.py \
    --config configs/forecasting/path_rl.yaml --output runs/path_rl

# local CPU: the stub-policy path_rl runs inside finekg-smoke and the test suite
uv run python scripts/evaluate_forecasting.py \
    --config configs/forecasting/ablation_path_random_walk.yaml --path data/fixtures/icews14/sample.tsv
```

## Verifier-as-risk-controller upgrade (the `calibration` / `admission` modules)

Design doc: `docs/RISK_CONTROL_DESIGN.md`. The same verifiers now also become a
**risk controller** with finite-sample guarantees, robust to the distribution
drift that voids static split conformal — the verifier's third duty (gate /
reward / risk-controller), spanning both stages and coupled across them.

| Thesis stage | Risk-control method | Guarantee | Train / evaluate |
|---|---|---|---|
| reasoning | drift-robust conformal (`core.calibration`: `aci` / `weighted` / `crc`) wired into `MultiAgentForecastingPipeline` (streaming, online feedback) | long-run coverage 1−α under shift | `scripts/evaluate_calibration.py` with `conformal_{split,aci,weighted,crc}.yaml` |
| edges | CRC edge admission (`finekg.relations.admission`) | gold FNR ≤ α (retain ≥ 1−α of true relations) | `crc_edge_admission.yaml` / `ablation_no_edge_admission.yaml` |
| coupling (C3) | edge confidence masks the path walk (`TemporalPathEnv(min_confidence=…)`, `event_graph_confidence`) | walk only over admitted edges | inside `self_built_event_graph` runs |
| counterfactual (C4) | model-consistent faithfulness (`build_path_reward_scorer(proxy=…)`) | bonus uses the real forecaster's ablation | `faithfulness_proxy` in `path_rl` config |
| downstream | event-driven selective trading (`finekg.forecasting.downstream`) | accuracy@coverage | `scripts/evaluate_downstream_trading.py` with `downstream/trading_selective.yaml` |

New metrics (`finekg.core.calibration.metrics`): rolling coverage, drift-coverage
gap, set-size efficiency, accuracy@coverage; CRC empirical risk.

```bash
# headline: which calibrator holds 1-alpha coverage under drift? (CPU)
uv run python scripts/evaluate_calibration.py \
    --config configs/forecasting/conformal_aci.yaml --path data/fixtures/icews14/sample.tsv

# downstream: selective trading on the financial event graph (needs the graph + movements)
uv run python scripts/evaluate_downstream_trading.py --config configs/downstream/trading_selective.yaml
```
