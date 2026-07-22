# Paper outline — Verifier-as-Risk-Controller for financial event-graph reasoning

> Target: SCI journal (Knowledge-Based Systems / Expert Systems with Applications /
> Information Fusion) or CIKM/AAAI. Scope = 方案A (ch3-anchored, verifier unified).
> Method/results sourced from `docs/RISK_CONTROL_DESIGN.md` + `docs/RL_DESIGN.md`;
> numbers marked **[TBD]** pending T6/eval. This is an outline (key points per
> section) to convert to prose; positioning reflects the 2026-06 frontier scan.

## Working title
A Verifier with Three Duties: Drift-Robust, Risk-Controlled Reasoning over
Evidence-Grounded Financial Event Graphs

## Contribution statement (the defensible delta)
The individual primitives (adaptive conformal, path-RL on TKGs, CRC selective
prediction) are each active areas; our contribution is their **integration into a
single verifier** that gates inference, shapes RL rewards, *and* controls risk —
spanning graph construction and temporal reasoning — on an evidence-grounded
Chinese financial event graph, with finite-sample guarantees that survive
distribution drift. Concretely:
1. A **streaming, drift-adaptive conformal controller** (split/ACI/weighted/CRC)
   wired into multi-agent forecasting with online feedback (C1).
2. **CRC edge admission** bringing an FNR≤α recall guarantee to graph
   construction (C2), with **cross-stage uncertainty propagation** masking the
   reasoning action space to admitted edges (C3).
3. **Model-consistent counterfactual faithfulness** as both an abstention gate
   and an RL reward (C4), unifying the verifier's training and inference roles.
4. A **compute-equivalent evaluation** (`--cache-ranks`): one forecaster pass
   replays all calibrators, making the drift study tractable on large series.
5. ★ **CS-CRP — drift-robust cross-stage conformal risk propagation**: compose the
   construction-stage CRC recall guarantee (FNR≤α_e) with the reasoning-stage
   drift-adaptive coverage guarantee (1−α_p) into one selective predictor that models
   the admission→reachability tradeoff, holding composed coverage ≥1−α under drift.
   Extends exchangeable pipeline-CP (PASC 2605.18812, CASCADE 2605.20468) to the
   non-exchangeable + heterogeneous recall⊗coverage + temporal event-graph setting.
   *(Patent: add a claim before any public release.)* Full design in `THESIS_DESIGN.md §4`.

## 1. Introduction
- Motivation: financial event reasoning needs *reliable* predictions; point
  forecasts without risk control are dangerous under market regime shift.
- Gap: static conformal assumes exchangeability → coverage breaks under drift;
  construction-stage errors propagate; no unified verifier across stages.
- Contributions (the 4 above) + a self-built evidence-grounded Chinese financial
  event graph closing the extraction→relation→reasoning loop.

## 2. Related work (positioned — frontier scan 2026-06)
- **Conformal under shift / time series**: TCP (arXiv 2507.05470), ACI
  (Gibbs–Candès), CORE (RL×CP, OpenReview 7puF5JOkKk), dynamic-GNN CP (2405.19230),
  conformalized link prediction (KDD'24). → we *wire* these into a financial
  event-graph system and couple them across construction+reasoning, not a new CP.
- **TKG forecasting (RL/LLM)**: TITer/TimeTraveler, RECIPE-TKG (2505.17794),
  Self-Exploring LMs (2509.00975 — RL+explainability but *no* conformal / faithfulness
  reward / finance), Chain-of-History. → ours adds risk control + faithfulness.
- **Selective prediction / CRC for LLMs**: SCRC (2512.12844), SCOPE (2602.13110),
  conformal abstention. → we use CRC for *edge admission*, a construction-stage
  guarantee.
- **Faithful KG reasoning + abstention**: GCR (2410.13080), SAFE (2604.01993),
  KNOWGUARD — all multi-hop QA, not temporal forecasting + risk control + finance.
- **Financial event KG / event-driven prediction**: FinKario (2508.00961, construction
  only), Janus-Q (2602.19919, RL trading reward), Fin-R1 (financial reasoning).
  → ours centers *formal risk guarantees*, not return modeling.
- Honest boundary to state: downstream trading signal is weak (future work); the
  selective-prediction value is demonstrated on the forecasting task itself.

## 3. Preliminaries
- TKG forecasting setup (s, r, ?, t) extrapolation; MRR/Hits.
- Split conformal + nonconformity = gold rank; coverage 1−α; exchangeability.

## 4. Method — the verifier's three duties
- 4.1 Shared verifier kernel: grounding / consistency / faithfulness.
- 4.2 Duty 1 — gate (inference-time abstention on unfaithful forecasts).
- 4.3 Duty 2 — reward (GRPO-RLVR for relations; faithfulness-shaped path-RL).
- 4.4 Duty 3 — risk controller: C1 streaming drift-adaptive conformal (state not
  clipped → telescoping coverage); C2 CRC edge admission (FNR≤α); C3 confidence
  masking the path walk; C4 model-consistent counterfactual faithfulness.
- 4.5 System: multi-agent (proposer/grounding-verifier/arbiter; retrieval/reasoner/
  faithfulness-verifier/calibrator) over the event graph; cross-stage coupling.

## 5. Experimental setup
- Data: ICEWS14/18/05-15, FinDKG (forecasting); MAVEN-ERE, CCKS-2021 (relations);
  self-built Chinese financial event graph (SARGE → nodes); Astock/CMIN (downstream).
  ICEWS14 uses the TiRGN/TLogic 365-day split (do not resplit).
- Baselines: frequency; temporal_gnn; path-RL (ours); split vs aci/weighted/crc.
- Metrics: MRR/Hits; **coverage_drift_gap** (headline), conformal coverage,
  set-size efficiency, accuracy@coverage; relation P/R/F1, coref CoNLL, FNR.
- Implementation: Qwen3-4B+LoRA, TRL/GRPO, PyG; 4×RTX4090; `--cache-ranks` for the
  drift study.

## 6. Results (placeholders)
- 6.1 C1 headline: split's coverage_drift_gap ≫ aci/weighted under drift. **[TBD: T6
  on ICEWS14; stronger-drift ICEWS18/05-15 variant if needed]**
- 6.2 Forecasting: path-RL > temporal_gnn > frequency (ICEWS14 MRR 0.360/0.286/0.105,
  done). FinDKG/ICEWS18 GNN **[partial]**.
- 6.3 Relations: GRPO-RLVR ≫ SFT (CoNLL coref 0.27→0.77, temporal precision
  0.45→0.57, micro F1 5.7×, done).
- 6.4 Ablations (single-seed): −grounding/−consistency/−shaping/−faithfulness/
  −edge-admission. **[TBD]**
- 6.5 Risk-coverage: accuracy@coverage rises with selectivity on forecasting. **[TBD]**

## 7. Conclusion & future work
- Unified verifier (gate/reward/risk-controller) with drift-robust guarantees on a
  financial event graph. Future: denser event graph from news (SARGE on CMIN/Astock),
  stronger downstream trading signal, non-monotone risk (FDR).

## Key references (to expand into .bib)
TCP 2507.05470 · ACI Gibbs–Candès 2021 · CORE · dynamic-GNN CP 2405.19230 ·
RECIPE-TKG 2505.17794 · Self-Exploring LM 2509.00975 · SCRC 2512.12844 ·
GCR 2410.13080 · SAFE 2604.01993 · FinKario 2508.00961 · Janus-Q 2602.19919 ·
Angelopoulos CRC 2022 · TiRGN/TLogic (ICEWS14 split).
