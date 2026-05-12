# CARVE v1.3 — Frozen Method Design

> **Status**: method proposal, not implemented.  
> **Repository evidence**: data splits and evaluator tracks only.  
> **Implementation status**: no CARVE model components implemented yet.  
> **Literature status**: recheck before paper submission.  
> **Schema assumption**: schema contains only event types, roles, and arguments by default.  
> **Version**: CARVE v1.3 frozen proposal; supersedes EASV v1, ECPD-CRV v0, CARVE v1.1, and CARVE v1.2.  
> **Last updated**: 2026-05-12.  
> **Freeze status**: method proposal freeze. Do not add new modules before P5b. Only factual corrections, repository-consistency fixes, or evidence-based revisions from P0/P1/P4/P5a/P5b are allowed.

Evaluator track wording:

- ChFinAnn: legacy-doc2edag + unified-strict
- DuEE-Fin: historical-compatible / Doc2EDAG-style if supported + unified-strict
- DocFEE: docfee-official + unified-strict

The repository does not currently contain CARVE / EASV / ECPD-CRV model code. No CARVE component has been implemented, measured, released, or reproduced in this repository.

Required boundary statements:

```text
Status: method proposal, not implemented.
Repository evidence: data splits and evaluator tracks only.
Implementation status: no CARVE model components implemented yet.
Literature status: recheck before paper submission.
Schema assumption: schema contains only event types, roles, and arguments by default.
```

---

## 0. Expert Change Audit and Final Decisions

| ID | Expert request | Final decision |
|---|---|---|
| E1 | Add an optional positive-coverage regularizer for multi-positive mentions in `L_alloc`. | Accepted. Multi-positive marginal likelihood can concentrate probability mass on one positive column. CARVE keeps the regularizer optional and ablates it separately. |
| E2 | Make gold-record column ordering deterministic. | Accepted. Deterministic canonical sorting is required for reproducible `Y_{t,rho}` construction and byte-identical outputs. |
| E3 | Use training-only oracle injection for missing gold mentions; disable at inference; report candidate recall. | Accepted. It prevents early `L_alloc` collapse while exposing the train-inference candidate gap through candidate-recall reporting. |
| E4 | Report both total and eligible misallocation rates. | Accepted. Eligible denominator is the primary diagnostic because ambiguous repeated values are common in financial documents and can dilute total-rate interpretation. |
| E5 | Quantify P5b decision criteria. | Accepted with a decision-profile format rather than a single pass/fail threshold. |
| E6 | Reorder GPU fallback policy: checkpoint/freeze/cache/windowing before model replacement. | Accepted. Real memory-saving steps should precede swapping Lawformer and Longformer-style encoders. |

---

## 1. One-Sentence Positioning

CARVE moves record grouping from post-hoc matching or per-record independent decoding into:

1. **Before decoding**: a differentiable cross-record allocation prior, proposed as a Sinkhorn-regularized soft competition matrix and directly supervised by `L_alloc`.
2. **During decoding**: conflict-aware EDAG candidate scoring, with an independent learned share gate.
3. **After decoding**: set-level allocation verification with train-only model-induced hard negatives.

These components are connected through role-level multi-instance evidence anchoring and produce a mechanism-diagnostic loop focused on **misallocation**, not generic hallucination.

---

## 2. Related-Work Positioning

| Work | CARVE differentiation |
|---|---|
| Doc2EDAG, GIT, IPGPF | Their per-record path expansion or iteratively parallel generation focuses on record-internal decoding. CARVE injects a directly supervised cross-record argument-to-record allocation prior before and during decoding. |
| DE-PPN | DE-PPN uses Hungarian matching as a training-time set-matching loss over whole records. CARVE uses a differentiable single-argument-to-record prior at both training and inference. |
| SEELE | SEELE uses schema descriptions as encoder-side attention bias. CARVE adds argument-to-evidence-sentence multi-instance grounding. |
| CAINet, TGIN, TER-MCEE | Their main mechanisms are graph or multi-channel structures. CARVE is an OT-style allocation-prior and verifier route, orthogonal to graph modeling. |
| ExpDEE, SEA | They use sentence clues or type-aware decoding. CARVE turns evidence into explicit multi-instance grounding supervision. |
| Doc2Event, SALE | They follow generative or schema-valid decoding. CARVE defaults to EDAG-style mention selection and treats generation only as optional. |
| MoE-ML-CNEE (CCL 2025) | LLM fine-tuning / MoELoRA route versus CARVE's discriminative allocation-verification route. Do not claim resource superiority without reproduction. |
| JLF (IEEE TNNLS 2026, PubMed indexed, DOI 10.1109/TNNLS.2026.3678139) | Paper existence is indexed; code availability, split compatibility, metric compatibility, and reproducibility remain unverified. CARVE is framed as a differentiable allocation-prior and verifier method, orthogonal to joint-learning / topology-decomposition framing. |
| DocFEE | CARVE does not compete with DocFEE's LLM-SFT route as an LLM extractor. Qwen-4B, if used, is only an appendix enhanced verifier over short evidence, not a full-document extractor. |

---

## 3. CARVE v1.3 Architecture

### 3.1 Input and Encoding

- Document: `D = (s_1, ..., s_T)`.
- Schema: `S = {(t, R_t)}`, where `R_t` is the role list for event type `t`.
- The repository schema is assumed to contain only event types, roles, and arguments. No mutual-exclusion, entity-type, or role-sharing constraints are used by default.
- ChFinAnn / DuEE-Fin: `hfl/chinese-roberta-wwm-ext`, 512-token sliding windows with 64-token overlap, sentence pooling, and a 2-layer global transformer.
- DocFEE: encoder selected on dev from candidates such as `thunlp/Lawformer` and `schen/longformer-chinese-base-4096`. Do not write either as the best choice before dev comparison and P1 memory measurement.
- Outputs: token representations `x_{i,j}`, sentence representations `h_i`, and global pooled representation `g`.

### 3.2 Weak Evidence Labeling

For each gold record `r = (t, {(rho, v_rho)})`:

- Align each role value `v_rho` to the source document with the exact NFKC + whitespace-collapse normalization used by the evaluator.
- Define `PosSent(r, rho) = {i : s_i contains v_rho}`.
- Construct sentence × event-type labels `y_ev[i,t]` and sentence × event-type × role labels `y_ev[i,t,rho]`.

**Required limitation**: these labels are deterministic projections from gold event records, not human evidence annotations. They must not be reported as evidence-F1 main results. They are used only as training supervision, downstream record-F1 support, and diagnostics.

### 3.3 Bidirectional Role-Evidence Anchoring

Forward heads:

- Sentence × type BCE head.
- Sentence × type × role BCE head.

Backward grounding head:

\[
L_{\mathrm{ground}}^{\mathrm{MI}}
=
-\sum_{r,\rho}\log \sum_{i\in PosSent(r,\rho)} \hat p_{r,\rho,i}
\]

Hard-EM is only a proposed robustness variant and is not the default. First-match sentence is only a diagnostic field and tie-breaker, not the main loss target.

### 3.4 Mention Extraction and Record Planner

- Mention extractor: BIO-CRF over selected evidence sentences.
- Record planner: truncated Poisson softmax head predicting `n_t` for each event type.

### 3.5 Cross-Record Soft Allocation Layer

#### 3.5.1 Design Clarification

`A_{t,rho}` is not a hard assignment matrix and not a strict one-to-one solver. It is a differentiable allocation prior used to bias EDAG candidate scoring. A mention may still be selected by multiple records when the learned share gate permits it. Sinkhorn is used to model cross-record competition under entropic regularization, not to impose exclusivity. Do not claim row-normalized Sinkhorn naturally solves legal cross-record sharing.

#### 3.5.2 Sinkhorn-Based Competition Prior

For each event type `t` and role `rho`, define candidate mentions `M_{t,rho}` and record count `n_t`.

\[
C_{t,\rho}[m,j]
=
-\left(\alpha s_{\mathrm{ev}}(m,t,\rho)+\beta s_{\mathrm{state}}(m,\hat R_j)\right),
\quad j\le n_t
\]

\[
C_{t,\rho}[m,n_t+1] = -b_{\mathrm{NULL}}
\]

where `alpha`, `beta`, and `b_NULL` are learnable scalars.

\[
\tilde A_{t,\rho}=\exp(-C_{t,\rho}/\tau), \quad
A_{t,\rho}=\mathrm{Sinkhorn}(\tilde A_{t,\rho}, \text{row marginals}, \text{column marginals})
\]

The matrix is used as a differentiable allocation prior during training and inference.

#### 3.5.3 Candidate Set Construction and Oracle Injection

Training-time candidates:

```text
M_train(t,rho) =
  TopK_evidence(t,rho)
  union MentionExtractor(selected_sentences)
  union OracleInject(gold_values(t,rho,doc))
```

Inference-time candidates:

```text
M_infer(t,rho) =
  TopK_evidence(t,rho)
  union MentionExtractor(selected_sentences)
```

Proposed requirements for a later implementation phase:

- Define a boolean flag `oracle_inject`.
- Training default: `oracle_inject=True`.
- Inference: forced `oracle_inject=False`.
- Report candidate recall before and after oracle injection.
- Stage B hard-negative mining must not use oracle injection.

### 3.5.4 `L_alloc`: Multi-Positive Allocation Supervision

#### Deterministic Gold-Record Column Ordering

For each document, event type, and role:

1. Sort gold records of event type `t` by gold record ID if present.
2. Otherwise sort by canonical role-value tuple: role names in schema order, normalized values, and raw record index as final tie-breaker.
3. Output must be deterministic under a fixed seed.

Construct:

```text
Y_{t,rho} in {0,1}^{|M_train(t,rho)| x (n_t^* + 1)}
```

For each sorted gold record column `k`, if role `rho` has normalized value `v`, set `Y[m,k]=1` for all candidate mentions `m` whose normalized value equals `v`. Mentions not assigned to any gold record are assigned to the NULL column.

#### Multi-Positive Marginal Allocation Loss

\[
L_{\mathrm{alloc}}^{\mathrm{ML}}
=
-\sum_{(t,\rho)}\sum_{m\in M_{t,\rho}^{train}}
\log \sum_{j:Y_{t,\rho}[m,j]=1} A_{t,\rho}[m,j]
\]

#### Optional Positive-Coverage Regularizer

For shared mentions with multiple positive columns:

\[
L_{\mathrm{alloc}}^{\mathrm{cov}}
=
-\sum_{(t,\rho)}\sum_{m\in S_{t,\rho}}
\frac{1}{|P(m)|}\sum_{j\in P(m)}
\log(A_{t,\rho}[m,j]+\epsilon)
\]

where:

- `S_{t,rho}` contains mentions with two or more positive columns.
- `P(m) = {j : Y[m,j]=1}`.
- `epsilon = 1e-8`.

Proposed objective:

\[
L_{\mathrm{alloc}} = L_{\mathrm{alloc}}^{\mathrm{ML}} + \mu L_{\mathrm{alloc}}^{\mathrm{cov}}
\]

with `mu` swept on dev, for example `{0, 0.1, 0.2, 0.5}`. This term is optional and must be ablated separately.

#### Key Differentiation

DE-PPN uses Hungarian matching over whole records during training. CARVE uses a differentiable single-argument-to-record allocation prior at both training and inference.

Safe claim:

> To our knowledge, CARVE is the first DEE framework that uses a differentiable argument-to-record allocation prior during both training and inference, with multi-positive supervision and sharing handled by an independent gate.

### 3.5.5 Learned Share Gate

\[
g_{t,\rho}(m)
=
\sigma\left(W_{\mathrm{share}}[\mathrm{typeEmb}_t;\mathrm{roleEmb}_\rho;\mathrm{mentionRepr}_m]+b_{\mathrm{share}}\right)
\]

Gold sharing label:

A mention is considered shared if the same normalized surface role-value appears as the same role value in two or more gold records of the same event type within the same document. This is a surface-level sharing signal, not entity coreference. No coreference resolution or entity linking is assumed.

### 3.5.6 Schema Hard Constraints

The proposed main method applies no schema-derived hard exclusion because the schema files are assumed to contain only event types, roles, and arguments. Optional role-constraint configs, if introduced later, are manually specified experimental assumptions and must be reported separately. Headline results must not depend on them.

### 3.6 Conflict-Aware Constrained EDAG Decoder

```text
for each event type t passing type gate:
    n_t = planner(t, D)
    initialize records R_hat_1 ... R_hat_n
    used_args[rho] = {}

    for rho in schema_order(R_t):
        M = inference-time candidate set, with oracle injection disabled
        A = SinkhornPrior(M, n_t)
        g = ShareGate(M, t, rho)

        for record j in 1..n_t:
            for mention m in M:
                score(m | R_hat_j, rho) =
                    base_score(m, R_hat_j, rho)
                    + lambda_alloc * log A[m,j]
                    + lambda_share * g(m) * indicator(m in used_args[rho])
                    - lambda_compete * (1 - g(m)) * indicator(m in used_args[rho])

            choose best mention or NULL
            update used_args and record state
```

### 3.7 Long-Document Role-Summary Memory

For DocFEE or documents exceeding encoder length, this proposal would:

- Split into 4096-token chunks with overlap.
- Compute role-aware pooling per chunk.
- Aggregate chunk-level role vectors with a small GRU.
- Decode from fixed-size role states.

All memory claims are estimated until P1 measurement.

### 3.8 Allocation Verifier

Proposed default verifier: V1 lightweight transformer (~30M parameters). Qwen-4B QLoRA verifier is appendix-only if implemented in a later phase.

Inputs:

- Predicted record set.
- Argument pointers.
- Evidence sentences.
- Allocation-prior statistics.
- Share-gate statistics.

Outputs:

- Record label: support / partial / unsupported.
- Per-record misallocation score.
- Set-level allocation score.

Negative mining:

- Stage A, train split only: swap / merge / split / role-shift / hallucination synthetic negatives.
- Stage B, train predictions only: high-confidence wrong records, near-miss Hungarian matches, and evaluator-rejected records from train predictions only.
- Dev is used only for threshold selection, early stopping, hyperparameter selection, and diagnostics.

### 3.9 Proposed Inference

- Prune records below verifier threshold `tau_v`.
- If all records are pruned and planner predicts at least one record, perform at most one constrained re-decoding pass.
- Do not use oracle injection at inference.

### 3.10 Training Objective

\[
\mathcal{L}
=
\lambda_1L_{ev,t}
+\lambda_2L_{ev,t,\rho}
+\lambda_3L_{ment}
+\lambda_4L_{plan}
+\lambda_5L_{path}
+\lambda_6L_{\mathrm{ground}}^{MI}
+\lambda_7L_{cov}
+\lambda_8L_{verify}
+\lambda_9L_{sinkhorn\_reg}
+\lambda_{10}L_{share}
+\lambda_{11}L_{alloc}
\]

Proposed training stages:

1. Stage 1: freeze verifier; train evidence, pointer, mention, planner, Sinkhorn, share gate, EDAG, and allocation heads. Oracle injection is enabled.
2. Stage 2: unfreeze verifier and allow path / pointer fine-tuning. Stage B hard negatives are mined from train predictions without oracle injection.

---

## 4. Diagnostics

### 4.1 Three-Way Diagnostic Split

| Metric | Definition |
|---|---|
| hallucinated argument rate | Predicted argument value is not in the source text after normalization. |
| ungrounded argument rate | Predicted value appears in text, but pointer lands on a low-evidence sentence. |
| misallocated argument rate | Predicted value appears in text and is grounded, but is assigned to the wrong record. |

### 4.2 Misallocation Rate with Dual Denominators

Step 1: Align predicted and gold records within each document × event type using Hungarian matching under unified-strict role-value overlap.

Step 2: Categorize each predicted argument.

- Hallucinated: normalized value not in document text.
- Ungrounded: normalized value appears, but pointer lands on a low-evidence sentence.
- Candidate for misallocation: value appears, pointer is grounded, and the role-value pair exists in at least one gold record.
- Ambiguous repeated: the same role-value pair exists in two or more gold records of the same event type in the document. Mark as ambiguous-excluded and report separately.
- Misallocated: candidate is true, ambiguous repeated is false, and the aligned gold record does not contain the role-value pair.
- Eligible: neither hallucinated nor ungrounded nor ambiguous repeated.

Report:

```text
misallocated_rate_total    = #misallocated / #total_predicted_arguments
misallocated_rate_eligible = #misallocated / #eligible_grounded_nonambiguous_predicted_arguments
ambiguous_excluded_rate    = #ambiguous_repeated / #total_predicted_arguments
```

Primary main-paper diagnostic: `misallocated_rate_eligible`.

---

## 5. GPU Constraint and Fallback Policy

All memory numbers are estimated until P1.

### 5.1 Estimated Memory

| Module | Estimated memory |
|---|---|
| Chinese-RoBERTa-wwm-ext | 8–12 GB |
| Lawformer / Longformer-Chinese | 14–18 GB |
| Evidence + pointer heads | negligible |
| Sinkhorn allocation + `L_alloc` | <1 GB |
| Share gate | <1 GB |
| Mention CRF | <1 GB |
| EDAG decoder | 2–3 GB |
| Role-summary GRU | <1 GB |
| Verifier V1 | 3–4 GB |
| Qwen-4B QLoRA verifier | estimated only; must be measured in this repository |

### 5.2 Fallback Order

If P1 shows DocFEE encoder + CARVE modules exceed 22 GB on a single RTX 4090:

1. Enable gradient checkpointing, reduce batch size, and use gradient accumulation.
2. Freeze the encoder and train CARVE heads on cached sentence representations.
3. Use 512-window Chinese-RoBERTa with role-summary memory.
4. Compare measured memory of Lawformer and Longformer-Chinese and pick the lower-memory option if needed.

Future decisions should be recorded in `docs/measurements/p1_memory.md` only after P1 is actually measured. `docs/measurements/p1_memory_template.md` is a template, not evidence.

### 5.3 Things Not to Do

- No Qwen-7B/13B full SFT.
- No LLM inference over more than 8k context.
- No end-to-end differentiable EDAG path selection.
- No new GNN.

---

## 6. Qwen-4B Position

- Not part of the headline method.
- Default verifier is V1 lightweight transformer.
- Qwen-4B QLoRA verifier is appendix-only.
- The paper framing is allocation-verification, not LLM ability.

---

## 7. Risk Register

| Risk | Mitigation |
|---|---|
| Sinkhorn appears like a small Doc2EDAG add-on | Emphasize `L_alloc`, training+inference use, and argument-level allocation. |
| Multi-positive allocation collapses to one positive column | Optional positive-coverage regularizer and ablation. |
| Multi-instance grounding unstable | Hard-EM robustness variant and first-match warmup. |
| Share gate label noisy | Limitations and share-gate-off ablation. |
| `L_alloc` target noisy due to repeated surface forms | Limitations, `L_alloc`-off ablation, ambiguous repeated diagnostics. |
| Oracle injection creates train-inference gap | Candidate recall before/after injection, Stage B without oracle. |
| Verifier looks like post-processing | Train-only hard negatives and verifier-linked re-decoding. |
| Memory exceeds 4090 limits | Fallback policy F1–F4. |
| JLF / MoE-ML-CNEE outperforms CARVE | Frame CARVE as complementary allocation-verification method with explicit diagnostics. |
| P5b fails | Do not force EMNLP main; fallback to simpler ECPD-CRV / EASV or target lower-risk venues. |

---

## 8. Paper Claim Draft

### Candidate Titles

1. *CARVE: Cross-record Allocation and Role-grounded Verification for Document-Level Event Extraction*
2. *Beyond Per-Record Decoding: Learning a Differentiable Cross-Record Argument Allocation Prior for Chinese Financial Event Extraction*
3. *From Hallucination to Misallocation: Diagnosing and Resolving Cross-Record Argument Binding Errors in Document-Level Event Extraction*

### Abstract-Level Claim

Document-level event extraction in Chinese financial documents is dominated by per-record decoding paradigms—path expansion, parallel prediction, or generation—that treat each record independently at decoding time and rely on Hungarian-style set losses to align predictions to gold at training time only. We argue that the dominant remaining errors are not hallucination but **misallocation**: arguments correctly identified and supported by evidence yet bound to the wrong record. We propose CARVE, which (i) supervises sentence-by-role evidence with weakly-projected labels and trains argument-to-evidence pointers under a multi-instance marginal likelihood objective, (ii) introduces, to our knowledge, the first DEE framework that uses a differentiable argument-to-record allocation prior during both training and inference—a Sinkhorn-regularized soft competition matrix supervised by a multi-positive allocation loss with an optional positive-coverage regularizer, paired with an independent learned share gate, and (iii) verifies the predicted record set with a lightweight allocation verifier trained on synthetic negatives and train-only model-induced hard negatives. We separate hallucinated, ungrounded, and misallocated diagnostics with explicit ambiguous-repeated handling and dual denominators.

### Contributions

1. Differentiable cross-record allocation prior with multi-positive supervision and an optional coverage regularizer.
2. Multi-instance evidence anchoring with bidirectional pointers and transparent training-only oracle-injection reporting.
3. Misallocation-aware verification with train-only hard-negative mining and ambiguity-aware diagnostics.

### Do Not Claim

- First schema-aware DEE.
- First evidence-grounded DEE.
- First LLM-based DEE on DocFEE.
- SOTA on ChFinAnn.
- Solving hallucination.
- Absolute "first differentiable allocation in DEE."
- Concrete JLF or MoE-ML-CNEE numbers.
- Resource superiority over MoE / LLM fine-tuning routes.
- Sinkhorn naturally solves legal sharing.
- Oracle injection has no inference impact.
- Hard-EM is generally better than marginal likelihood.

---

## 9. Construction Order

| Phase | Build | Acceptance |
|---|---|---|
| P0 | Weak evidence labeler design freeze: PosSent, sharing labels, multi-positive `Y`, deterministic canonical sort, unit-test plan. | Documentation freeze only. No code is implemented in Phase 0. |
| P1 | Encoder interface and memory measurement plan. | Future forward pass and memory results recorded in future `docs/measurements/p1_memory.md`. |
| P2 | Evidence and pointer heads. | Future evidence BCE and pointer MI-loss checks. |
| P3 | Mention CRF and record planner. | Future mention F1 and planner MAE checks. |
| P4 | Sinkhorn allocation, `L_alloc`, positive coverage regularizer, share gate, oracle-injection pipeline. | Future toy behavior and projection tests. |
| P5a | Conflict-aware EDAG toy unit test. | Future toy misallocation comparison. |
| P5b | Dev diagnostic gate on multi-event subset. | Future decision profile recorded in `docs/measurements/p5b_decision_table.md`. |
| P6 | Long-document role-summary memory. | Future memory check against P1 limit or fallback range. |
| P7 | V1 verifier and train-only negative mining. | Future synthetic-dev and train-only checks. |
| P7' | Appendix Qwen-4B verifier. | Future measured memory and training-loss checks only if authorized. |
| P8 | End-to-end inference and evaluator plug-in. | Future dev inference and scoring check. |
| P9 | Diagnostics hooks. | Future JSON-output stability check. |

---

## 10. P5b Decision Profile

For each dataset's dev multi-event subset, compare with vs. without Sinkhorn allocation prior + `L_alloc` + share gate.

| Dataset | Delta misallocated_rate_eligible | Delta record F1 | Inference candidate recall | Signal |
|---|---:|---:|---:|---|
| ChFinAnn | TBD | TBD | TBD | TBD |
| DuEE-Fin | TBD | TBD | TBD | TBD |
| DocFEE | TBD | TBD | TBD | TBD |

Per-row signal:

- **Strong**: `misallocated_rate_eligible` decreases by at least 1.0 absolute point and record F1 does not drop by more than 0.5 absolute.
- **Weak**: `misallocated_rate_eligible` decreases but by less than 1.0, or record F1 drops by 0.5–1.0 absolute.
- **No support**: `misallocated_rate_eligible` does not decrease or record F1 drops by more than 1.0 absolute.

Venue decision:

- Three strong signals: EMNLP main stretch is justified.
- Two strong + one weak: EMNLP main or Findings, depending on final baseline strength.
- Three weak: Findings/COLING only if record F1 is stable and diagnostics are visually consistent; otherwise COLING / SCI Q2.
- One strong + two weak or any no-support signal: COLING / SCI Q2, or simplify back to ECPD-CRV.
- Two or more no-support signals: CARVE mechanism fails; do not force main-conference framing.

---

## 11. Self-Check

1. CARVE is a method proposal above evaluator-only documentation.
2. GPU feasibility is estimated and must be measured in P1 before any paper claim.
3. The proposal covers ChFinAnn, DuEE-Fin, and DocFEE.
4. The evaluator-track wording follows the repository boundary in this document header.
5. All six final expert fixes are included as proposal decisions.
6. Repository facts are not overstated as implementation evidence.
7. Literature notes are cautious and must be rechecked before submission.
8. P5b is the future decision point for whether CARVE is EMNLP-main stretch, Findings/COLING, or fallback.

---

## 12. Archive Plan

```text
docs/method/carve_method_design_v1_3.md
docs/method/carve_method_design_v1_2.md
docs/method/easv_v1.md  # deprecated and intentionally removed from this checkout
docs/method/ecpd_crv_v0.md
docs/method/README.md
docs/measurements/p1_memory_template.md
docs/measurements/p5b_decision_table_template.md
docs/phase/README.md
docs/phase/p0_documentation_freeze.md
docs/phase/p1_memory_measurement_plan.md
docs/phase/p4_allocation_toy_validation_plan.md
docs/phase/p5a_edag_toy_gate_plan.md
docs/phase/p5b_dev_diagnostic_gate_plan.md
```

`docs/method/README.md` should state that all method documents are proposals only and that no CARVE / EASV / ECPD-CRV model components are implemented yet.
