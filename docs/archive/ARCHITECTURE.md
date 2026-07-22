# Architecture & upgrade path

The MVP is built so that scaling a stage from baseline to full method is **adding
an implementation**, never reworking the framework. Three mechanisms enforce
this.

## 1. Frozen contracts (`finekg.core.schema`)

The only types crossing stage boundaries:

```
event records ──► EventNode ──► RelationEdge / EventGraph ──► TemporalQuad / ForecastQuery ──► Prediction
   (SARGE)          node            relation stage (edges)         forecasting stage            ranked + evidence chain
```

Rule: **extend by adding optional fields; never repurpose an existing field.**
Every node, edge and prediction carries `evidence` / `evidence_chain`, so the
"evidence-grounded" property holds end to end.

## 2. Pluggable registries (`finekg.core.registry`)

Each swappable component owns a `Registry`. Implementations self-register; configs
select one by name. Current registries:

| Registry | Baseline (CPU, torch-free) | Full method (GPU, server) |
|---|---|---|
| `relation_extractors` | `heuristic` | `llm` (LoRA, evidence-grounded) |
| `consistency_solvers` | `identity` (ablation), `greedy` | add e.g. `ilp` here |
| `forecasters` | `frequency`, `path_rl` (stub policy) | `temporal_gnn` (PyG), `path_rl` (trained GRU policy) |
| `relation_reward_components` | `format` / `grounding` / `consistency` / `task_f1` (all CPU) | — (rewards stay CPU; only the GRPO trainer is GPU) |

Adding a method:

```python
@relation_extractors.register("my_method")
class MyExtractor(RelationExtractor):
    def extract(self, nodes, context=None): ...
```

No change to pipelines, configs schema, or callers — only a new name to select.

## 3. CPU/GPU layering

`core` + the heuristic baselines run anywhere with no torch. Neural code
(`relations/extractor/llm.py`, `forecasting/model/temporal_gnn.py`,
`forecasting/forecaster/neural.py`) imports torch **lazily**, so the whole
package imports on a CPU box; only instantiation needs the `llm`/`gnn` extras.
This is why `uv run pytest` and `finekg-smoke` are fully green locally while the
GPU path waits for the server.

The RL stack follows the same split: rewards, the path MDP environment,
curriculum and advantage math are pure CPU (`finekg/rl`, `relations/rl/rewards`,
`forecasting/rl/{env,episodes,rewards,warmstart}` and the stub policy), while
the GRPO trainer (`relations/rl/trainer`, needs `llm` + `rl`) and the GRU path
policy/trainer (`forecasting/rl/{policy,trainer}`, needs `gnn` + `rl`) stay
behind lazy imports / availability guards.

## Data flow (two stages, one loop)

```
RelationPipeline:   nodes ─► extractor ─► grounding (anti-fabrication)
                          ─► consistency (acyclic causal / closed temporal) ─► EventGraph
ForecastingPipeline: TemporalGraphDataset ─► time split ─► forecaster.fit(history)
                          ─► predict(future, history<t) ─► MRR/Hits + evidence chain
bridge:             EventGraph ──(event_graph_to_dataset)──► TemporalGraphDataset   # closes the loop
```

## Where things live

```
core/        schema, io (+SARGE adapter), graph algos, eval metrics, registry, config
relations/   data loaders, extractor/, grounding/, consistency/, rl/ (GRPO-RLVR), pipeline
forecasting/ data loaders (+bridge), model/ (PyG), forecaster/, rl/ (path RL), pipeline
rl/          stage-agnostic RL base: composite rewards, group advantage, shaping, curriculum
scripts/     download / build_event_graph / evaluate_* / train_*  (function-named)
configs/     one YAML per experiment (baseline / main / ablations)
agents/      multi-agent orchestration protocol (Agent/Blackboard/Orchestrator)
```

## 4. Multi-agent orchestration (`agents` domain)

The relation and forecasting stages each have an agentic upgrade built on a
stage-agnostic substrate (`finekg.agents.protocol`: `Agent` / `Blackboard` /
`Stage` / `Orchestrator` / `Verifier`). Role agents self-register in `agent_roles`
and wrap the existing components, so the three mechanisms above still hold:

- **Same contracts**: agents consume/produce the frozen schema; a verifier only
  annotates the new optional fields (`faithfulness`, `admitted`, `abstained`,
  `coverage_set`, `intervention_delta`).
- **Registry**: `agent_roles` (proposer / grounding_verifier / consistency_arbiter
  / history_retrieval / forecast_reasoner / faithfulness_verifier /
  forecast_calibrator). New impls register a name; pipelines pick by config.
- **CPU/GPU layering**: the orchestration, the symbolic arbiter, the conformal
  calibrator and the *grounding* faithfulness proxy run on CPU with stub agents;
  the LLM proposer/verifier and the GNN reasoner are the only GPU pieces.

The verifier is the connective tissue — edge-level faithfulness (mask the evidence
span) for relations, path-level faithfulness (ablate the evidence chain) for
forecasts: one intervention idea, two granularities. Old methods are re-cast as
tools the agents consult (the consistency solver is the arbiter's tool; the
temporal GNN is a reasoner's tool).

```
MultiAgentRelationPipeline:    nodes ─► proposer committee ─► grounding-verifier
                                     ─► consistency-arbiter ─► EventGraph
MultiAgentForecastingPipeline: query ─► retrieval ─► reasoner committee
                                     ─► faithfulness-verifier ─► calibrator ─► Prediction
```

## 5. Verifier-as-reward (`rl` domains)

The training-time mirror of the verifiers (design doc: `docs/RL_DESIGN.md`):
**at training time the verifier is the reward; at inference time it stays the
gate.** `finekg/rl` is the stage-agnostic base (composite rewards with
per-component traces, GRPO-style group-relative advantages, potential-based
shaping, easy-to-hard curriculum); each stage contributes its own rewards:

- `relations/rl` — GRPO post-training of the LLM extractor where the reward is
  format + evidence grounding + global consistency + gold F1, every component
  reusing the inference kernels (`ground_relations`, the greedy solver,
  `relation_prf`). Ablating a component = deleting a line in the config.
- `forecasting/rl` — a TITer-style path walk (`path_rl` in `forecasters`) whose
  policy is trained with group-relative REINFORCE; hits earn an
  intervention-faithfulness bonus (the same counterfactual the path verifier
  runs), shaping densifies the sparse signal. The walked path *is* the evidence
  chain, so existing verifier/calibrator agents gate it unchanged.

The three mechanisms above still hold: schema untouched (the evidence chain is
the interface), `path_rl` and the reward components are registry entries, and
everything except the two trainers runs on CPU.
