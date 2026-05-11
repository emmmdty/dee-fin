# Defensible EMNLP 2026 Methodology Plan for Chinese Financial Document-Level Event Extraction

# 1. Executive Decision

**Recommendation.** Pursue **Option A — Evidence-Anchored Schema-Constrained Record Extraction with a Record-Level Verifier (EASV)** as the main submission, with **Option C (Qwen-4B QLoRA structured generation) as a fallback / additional baseline track**. Target **EMNLP 2026 Findings (or COLING 2026)** as the realistic primary venue, with EMNLP main as a stretch contingent on (i) statistically meaningful improvement over GIT/PTPCG/SEELE on ChFinAnn and DuEE-fin and (ii) a genuinely new strong result on DocFEE, since no prior peer-reviewed DEE model has been evaluated on DocFEE under the canonical role-value protocol. Honest grade: under current constraints the proposal is closer to a strong Findings paper than a clear EMNLP-main paper; the upgrade path is in §15.

Justification bullets:
- Three independent claims of novelty are defensible: (a) first uniform Doc2EDAG-style + unified-strict evaluation across ChFinAnn, DuEE-fin, DocFEE; (b) sentence×role weak-evidence supervision with EDAG-or-generation-agnostic decoding; (c) record-level verifier producing strict, auditable "unsupported argument rate" and "evidence grounding rate" diagnostics that no published Chinese-finance DEE paper reports.
- Methodologically the only nearby works are SEELE (Xu, Zijie, Peng Wang, Wenjun Ke, Guozheng Li, Jiajun Liu, Ke Ji, Xiye Chen, and Chenxiao Wu; IJCAI 2024, pp. 6597–6605, doi 10.24963/ijcai.2024/729 — "achieving notable improvements (2.1% to 9.7% F1) on three NDEE datasets"), CAINet (LREC-COLING 2024), TGIN (Neural Networks 2024), TER-MCEE (ACM TOIS 2024) and Doc2Event (ICONIP 2024). None of these uses (i) explicit role-level evidence supervision + (ii) record-level verifier + (iii) the new DocFEE long-doc dataset. The user's "JLF (TNNLS 2026, +10.6/+5.6 F1)" and "MoE-ML-CNEE" reference points **could not be verified** — there is no indexed paper matching either description; treat both as nonexistent until a citation appears.
- Qwen-4B is best used as a **verifier/reranker**, not as the primary extractor, because (a) Qwen-4B SFT on long Chinese financial docs is comparable to but smaller than the published DocFEE LLM_SFT baseline (Qwen1.5-7B-Chat) and not the strongest published surface-form F1 model, and (b) using it as verifier preserves direct compatibility with the canonical Doc2EDAG fixed-slot metric and avoids reviewer attacks about "LLM-as-judge contamination" of the main score.

# 2. Repository-Aware Starting Point

Note: the live GitHub URL `https://github.com/emmmdty/deefin` returned a permissions error from this auditing environment, so the following is reconstructed from the user-provided context plus standard expectations for a DEE codebase. Anything I cannot verify is marked.

**Already in `dee-fin` (per user statement):**
- Reproducible splits with byte-identical local/server checks for ChFinAnn (25,632 / 3,204 / 3,204), DuEE-fin (6,515 / 500 / 1,171, with raw test 59,394 kept blind), DocFEE (17,244 / 1,000 / 800).
- Three independent evaluators: `legacy-doc2edag` (fixed-slot, two input modes), `docfee-official` (mirrors `utils.py` but eval-safe), `unified-strict` (role-value micro-F1 with DP / linear assignment / greedy fallback).
- Strict surface-form normalization (NFKC, whitespace collapse), no alias/embedding/LLM judging in main metric.
- Duplicate audit logs (not deduped in main).

**Likely present but should be verified before drafting the paper:**
- Schema files (event types × roles) per dataset.
- Canonical role-value JSONL writers.
- A `scripts/` directory for split regeneration with deterministic seeds.

**Likely missing / to build:**
- Baseline integration adapters for Doc2EDAG, GIT, DE-PPN, PTPCG (DocEE toolkit), RAAT/ReDEE, SEELE — each requires a wrapper that takes the dataset's canonical JSONL and emits Doc2EDAG-style event tables.
- Long-document encoders for DocFEE: Lawformer (Xiao, Chaojun, Xueyu Hu, Zhiyuan Liu, Cunchao Tu, and Maosong Sun; AI Open 2, 2021, 79–84; arXiv 2105.03887; HuggingFace `thunlp/Lawformer`; pre-trained continuously from `hfl/chinese-roberta-wwm-ext` at 4,096 tokens) and `schen/longformer-chinese-base-4096`.
- Constrained decoding stack: XGrammar (Dong, Yixin, Charlie F. Ruan, Yaxing Cai, Ruihang Lai, Ziyi Xu, Yilong Zhao, and Tianqi Chen; arXiv 2411.15100; Proceedings of Machine Learning and Systems 7, 2025; default backend for vLLM, SGLang, TensorRT-LLM, and MLC-LLM; up to 100× speedup over previous structured generation engines) + JSON schemas per event type.
- Weak-evidence labeler that converts role-value triples → sentence×role evidence labels via deterministic surface-form alignment.
- Verifier module (lightweight encoder OR Qwen-4B QLoRA classifier).
- Diagnostic harness: long-doc length buckets, single vs multi-event slicing, per-role F1, record-grouping accuracy, evidence-grounding rate.

# 3. Literature Verification Table

All entries below were checked against ACL Anthology, arXiv, IEEE, ACM, Nature/Scientific Data, or the original GitHub repo. "Novelty threat" is judged against the proposed method (evidence-anchored, schema-constrained, with record verifier, evaluated across three Chinese-financial DEE datasets).

| # | Paper / Method | Year | Venue | Datasets | Core idea | Public code | Novelty threat | One-line citation |
|---|---|---|---|---|---|---|---|---|
| 1 | Doc2EDAG (Zheng, Cao, Xu, Bian) | 2019 | EMNLP-IJCNLP | ChFinAnn | EDAG path-expansion no-trigger DEE; canonical metric | github.com/dolphin-zs/Doc2EDAG | none (anchor) | ACL D19-1032 |
| 2 | DCFEE (Yang, Chen, Liu, Xiao, Zhao) | 2018 | ACL demos | ChFinAnn-style DS | Key-event + surrounding sentences | github (legacy) | none | aclanthology P18-4009 |
| 3 | GIT (Xu, Liu, Li, Chang) | 2021 | ACL-IJCNLP | ChFinAnn | Heterogeneous graph + Tracker; "GIT outperforms the previous methods by 2.8 F1" over Doc2EDAG (76.3 → 79.1 average F1 in Xu et al., ACL-IJCNLP 2021, doi 10.18653/v1/2021.acl-long.274) | github.com/RunxinXu/GIT | low | 2021.acl-long.274 |
| 4 | DE-PPN (Yang, Sui, Chen, Liu, Zhao, Wang) | 2021 | ACL-IJCNLP | ChFinAnn | Non-autoregressive parallel record prediction | github.com/HangYang-NLP/DE-PPN | low | 2021.acl-long.492 |
| 5 | PTPCG (Zhu, Qu, Chen, Wang, Huai, Yuan, Zhang) | 2022 | IJCAI | ChFinAnn, DuEE-fin | Pseudo-trigger pruned complete graph, non-AR | github.com/Spico197/DocEE | low | ijcai.org/proceedings/2022/632 |
| 6 | SCDEE (Huang & Jia) | 2021 | Findings EMNLP | ChFinAnn | Sentence community GAT subgraph per event | partial | low | 2021.findings-emnlp.32 |
| 7 | RAAT / ReDEE (Liang, Jiang, Yin, Ren) | 2022 | NAACL | ChFinAnn, DuEE-fin | Relation-augmented attention transformer | github.com/TencentYoutuResearch/EventExtraction-RAAT | low | 2022.naacl-main.367 |
| 8 | HRE (human-reading) (Wan et al.) | 2022 | ICASSP | ChFinAnn | Rough+elaborate reading | — | none | arxiv 2202.03092 |
| 9 | ExpDEE | 2022 | Knowl.-Based Syst. | ChFinAnn, DuEE-fin | Sentence-level event-clue back-tracing | — | medium (similar evidence flavor) | sciencedirect S0950705122003306 |
| 10 | SEA (Zhao et al.) | 2023 | ICASSP | ChFinAnn, DuEE-fin | Type-aware decoding + explicit aggregation | partial | low | arxiv 2310.10487 |
| 11 | IPGPF (Huang, Xu, Zeng, Chen, Yang, E) | 2023 | EMNLP | ChFinAnn, DuEE-fin | Iteratively parallel generation + pre-filling | aclanthology repo | low | 2023.emnlp-main.668 |
| 12 | TT-BECG (Wan et al.) | 2023 | ACL | ChFinAnn, DuEE-fin | Token-token bidirectional event completed graph | partial | low | 2023.acl-long.584 |
| 13 | TER-MCEE (Wan et al.) | 2024 | ACM TOIS | ChFinAnn, DuEE-fin | Token-event-role multi-channel matrix | partial | medium | doi.org/10.1145/3643885 |
| 14 | TGIN (Zhong, Shen, Wang) | 2024 | Neural Networks | ChFinAnn, DuEE-fin | Two-phase heterogeneous graph inference | partial | low | sciencedirect S0893608024002673 |
| 15 | CAINet (Pan, Li, Wang, Li, Li, Liao, Zheng) | 2024 | LREC-COLING | ChFinAnn, DuEE-fin | Event-relation graph + argument-correlation graph | partial | medium | 2024.lrec-main.459 |
| 16 | SEELE (Xu, Wang, Ke, Li, Liu, Ji, Chen, Wu) | 2024 | IJCAI | NDEE + DEAE | Schema-aware descriptions + contrastive learning; reports "notable improvements (2.1% to 9.7% F1) on three NDEE datasets" | github.com/TheoryRhapsody/SEELE | **high** (closest to schema-aware DEE) | doi 10.24963/ijcai.2024/729 |
| 17 | SIAT (Tao et al.) | 2024 | ACM TALLIP | ChFinAnn, DuEE-fin | Spatial-augmented interactions + adaptive thresholding | partial | low | dl.acm.org/doi/10.1145/3698261 |
| 18 | EADRE | 2024 | ACM TALLIP | ChFinAnn, DuEE-fin | Event-type-aware dynamic entity representation | partial | low | dl.acm.org/doi/full/10.1145/3695767 |
| 19 | LAAP (Liu et al.) | 2024 | Neurocomputing | ChFinAnn | Event prompts + entity-argument learning | — | low | sciencedirect S0925231224013559 |
| 20 | Doc2Event (Ding, Zhang, Lin, Zhu, Ma, Wang, Ren) | 2024 (ICONIP) | ICONIP 2024 (Auckland) | DuEE-fin, FNDEE (not ChFinAnn) | EDAG-serialized generation + prefix prompt on mT5 | not found | medium (generation+EDAG overlap) | ojs.aut.ac.nz/iconip24/2/article/view/46 |
| 21 | SALE (MDPI Electronics) | 2025 | Electronics 15(6) | mixed | Code-LLM Python-class structured DEE | partial | medium | mdpi.com/2079-9292/15/6/1187 |
| 22 | ASEE | 2025 | arXiv 2505.08690 | MD-SEE benchmark | Schema-paraphrase RAG | partial | low | arxiv 2505.08690 |
| 23 | DocFEE (Chen, Yubo, Tong Zhou, Sirui Li, and Jun Zhao) | 2025 | Sci Data 12, 772 | DocFEE (19,044 docs, "nine event categories and 38 event arguments", mean 2,277 Chinese chars) | New dataset + BERT_Tagging / BART_QA / Qwen1.5-7B-Chat SFT baselines | github.com/tongzhou21/DocFEE | none (dataset) | doi 10.1038/s41597-025-05083-9; PMC12065892 |
| 24 | Survey: Doc-IE (Zheng, Wang, Huang) | 2024 | FuturED @ EMNLP | — | Doc-IE survey w/ error analysis | — | none | 2024.futured-1.6 |
| 25 | Survey: LLM-IE (Frontiers of CS) | 2024 | Frontiers of CS | — | Generative LLM IE survey | — | none | doi 10.1007/s11704-024-40555-y |
| 26 | "JLF" (TNNLS, claimed +10.6/+5.6 F1) | claimed 2026 | TNNLS (claimed) | ChFinAnn, DuEE-fin | "event topology decomposition" joint learning | — | **UNVERIFIED — no indexed paper found in arXiv, IEEE Xplore, Google Scholar, ACL Anthology** | — |
| 27 | "MoE-ML-CNEE" (Qwen2-7B + MoELoRA) | claimed | unknown | ChFinAnn, DuEE-fin | MoE LoRA multi-task on Chinese fin DEE | — | **UNVERIFIED — no indexed paper found; closest related work is Tea-MoELORA (Tang, Yan, Gu, Huang; arXiv 2509.01158, Sept 2025) for classical/modern Chinese IE, NOT financial DEE** | — |

**Verification flags.**
- "JLF" with +10.6 ChFinAnn and +5.6 DuEE-fin: **not found** on arXiv, IEEE Xplore, Google Scholar, or ACL Anthology after targeted searches; the phrase "event topology decomposition" produces zero NLP hits. The proposal must not cite this number until a real reference appears; if reviewers ask, state explicitly that this work could not be located.
- "MoE-ML-CNEE": no indexed paper. The closest related work is Tea-MoELORA (Tang et al., arXiv 2509.01158), which is classical/modern Chinese IE and not Chinese-financial DEE.
- ProcNet vs. ProCNet: cited in CAINet (LREC 2024) and CFinDEE (DCAI 2024); attribution often Wang et al. 2023 but no widely-deployed public code under that name — treat as a strong but secondary baseline.

# 4. Candidate Method Comparison

| Dimension | A: Evidence-Anchored Schema-Constrained + Verifier (EASV) | B: Graph-RAG subgraph retrieval + constrained generation | C: Qwen-4B QLoRA structured generation + verifier |
|---|---|---|---|
| Core mechanism | Sentence×role evidence selection → role-aware state → schema-constrained record decoder → record verifier | Build doc entity/event graph → retrieve per-event subgraph → constrained gen | Long-context Qwen-4B QLoRA + schema-constrained JSON decoding + verifier |
| Backbone | Chinese-RoBERTa or Lawformer; optional Qwen-4B verifier | Same encoder + GNN | Qwen-4B (with sliding window or RoPE-extended) |
| Novelty vs. SEELE / CAINet / TER-MCEE / Doc2Event | Moderate — distinct from SEELE because evidence is supervised at sentence×role granularity, not via attention bias; distinct from Doc2Event because verifier is a separate trainable head with auditable diagnostics | Low–Moderate — heavy overlap with TGIN/CAINet graphs | Moderate–High in industrial value but Low in academic novelty (LLM-SFT is the DocFEE baseline) |
| Expected ceiling on ChFinAnn (F1) | High; competitive with GIT (Doc2EDAG 76.3 average F1 + GIT's reported +2.8 F1) and SEELE | Medium; graph methods plateau ≤ GIT/PTPCG range | Uncertain; Qwen1.5-7B SFT is the DocFEE baseline; 4B SFT likely below 7B |
| GPU feasibility on 1–2×4090 | Strong; RoBERTa/Lawformer fits in 24 GB; QLoRA verifier fits | Strong | Marginal for 4B QLoRA + 4096–8192 context; sliding window required |
| Engineering cost | Medium (4 modules, each well-scoped) | Medium-High (graph + retrieval + decoding) | Medium (training pipeline + decoder) |
| Reviewer-attack surface | Evidence supervision is weak/heuristic ⇒ must defend with sentence×role F1 and ablation | "RAG without retrieval ground truth"; graph design ad-hoc | "Why not Qwen2.5-7B/Qwen3-4B with full SFT?"; "is verifier just inference patching?" |
| Comparability with Doc2EDAG native metric | Direct (final output is event table) | Direct | Indirect (needs JSON→event-table conversion) |
| Ablation richness | Excellent (4 stages × 3 datasets) | Good | Medium |
| Diagnostic transparency | Highest (evidence grounding, support rate, verifier deltas) | Medium | Medium |

**Selection.** A is recommended. C is run as the *strongest single LLM baseline* and as an alternative inference path (Section 7). B is dropped: under the constraints, B duplicates TGIN/CAINet/SCDEE without a clean differentiator.

# 5. Final Proposed Method — EASV in Detail

**Input.** A document D = (s₁,…,s_T) of Chinese sentences; a schema 𝒮 = {(t, R_t)} mapping each event type t to its ordered role set R_t. Splits are the user's reproducible splits.

**Stage 1 — Document segmentation and encoding.**
- ChFinAnn / DuEE-fin: Chinese-RoBERTa-wwm-ext (`hfl/chinese-roberta-wwm-ext`) at 512-token windows with 64-token overlap; sentence boundary recovery from offsets.
- DocFEE (mean 2,277 Chinese chars per Chen, Zhou, Li, Zhao, Sci Data 12:772, 2025): primary encoder is Lawformer (`thunlp/Lawformer`, 4,096 tokens, Longformer-Chinese architecture, pre-trained continuously from `hfl/chinese-roberta-wwm-ext`; Xiao et al., AI Open 2 (2021): 79–84; arXiv 2105.03887) because it directly handles the median document length. Fallback: `schen/longformer-chinese-base-4096`. Ablation includes 512-window RoBERTa + sliding aggregation.
- Output: contextualized sentence vectors h_i and token vectors x_{i,j}.

**Stage 2 — Sentence-level evidence candidate construction (weak supervision).**
- For each gold record r = (t, {(ρ, v_ρ)}), every role value v_ρ is matched against the source text by deterministic surface-form alignment under the same NFKC+whitespace normalization used in evaluation. The first matching sentence index is the *primary evidence sentence* y^{ev}_{i,t,ρ} = 1; all matching sentences become *supporting evidence* (multilabel). Sentences with no role value get y^{ev}_{i,t,·} = 0.
- A sentence × event-type binary head s_{i,t} = σ(W_t h_i + b_t) is trained with weighted BCE. A sentence × event-type × role head s_{i,t,ρ} = σ(W_{t,ρ} h_i + b_{t,ρ}) is trained on the multilabel evidence target.
- Limitation (must be disclosed): the same surface-form alignment is used both to create labels and at strict evaluation time, so the evidence labels are a deterministic projection of the gold record — not human-annotated evidence. We never measure "evidence F1 vs human ground truth"; we only measure (i) sentence-recall@K for downstream argument coverage, and (ii) the *impact* of evidence supervision on downstream record F1 via ablation.

**Stage 3 — Role-level evidence state.**
Per (t, ρ), select the top-K sentences by s_{i,t,ρ} (K=8 ChFinAnn/DuEE-fin; K=16 DocFEE). Build role-aware state e_{t,ρ} via masked attention of the role embedding r_{t,ρ} over the selected sentence tokens, then concatenate with the global document CLS-style pooling. e_{t,ρ} ∈ ℝ^d.

**Stage 4 — Event-level plan and entity proposal.**
- Standard BIO-CRF mention extractor on selected sentences (shared across types) for candidate entity mentions.
- A document × event-type planner predicts the number of records per type with a Poisson-truncated softmax head, supervised by the gold record count.

**Stage 5 — Constrained record decoder.**
Two interchangeable decoders are implemented, controlled by a `--decoder` flag (this is itself a paper contribution):

D1. **EDAG-constrained discriminative decoder.** For each predicted record slot under type t, autoregressively select an entity mention (or NULL) for each role ρ ∈ R_t under a Doc2EDAG-style path expander, but at each node the candidate distribution is restricted to mentions appearing in the role-evidence top-K sentences of that role; entities outside the evidence support set are masked out. Loss: cross-entropy along the path, plus a coverage loss that penalizes evidence sentences with high s_{i,t,ρ} whose tokens are never selected.

D2. **Schema-constrained generative decoder.** A small Chinese seq2seq (mT5-base or BART-base-chinese) generates a JSON event table conditioned on the role-state-augmented document. Decoding is constrained with XGrammar (Dong et al., arXiv 2411.15100; MLSys 7, 2025; default backend for vLLM, SGLang, TensorRT-LLM, MLC-LLM) against a per-event-type JSON Schema generated from the dataset schema file. The schema constrains: event-type keys, role keys, value-as-substring-of-source (enforced via a copy mask, similar to TEXT2EVENT but stricter).

**Stage 6 — Record-level verifier.**
For each predicted record r̂, a verifier head v(r̂, D) ∈ [0,1] is trained to predict whether r̂ exactly matches a gold record under unified-strict role-value F1. Inputs: (i) role-value pairs serialized, (ii) concatenated evidence sentences, (iii) sentence-level support scores. Two implementations:

V1. Lightweight: 2-layer transformer over [role tokens; value tokens; evidence tokens; type embedding].
V2. **Qwen-4B QLoRA verifier** (recommended for the main result on DuEE-fin/DocFEE): prompt = "Given the document evidence sentences and a candidate record, output 'support' / 'partial' / 'unsupported'. Document: ⟨evidence⟩. Candidate: ⟨record⟩." Trained with class-balanced cross-entropy on positive (gold) records and three classes of mined negatives: type-swap, role-swap, single-arg-perturbation. QLoRA rank 16, NF4 4-bit, only on `q_proj,k_proj,v_proj,o_proj`. Fits in 18–20 GB on one 4090 at sequence length 4,096; second 4090 hosts batch inference.

Thresholding τ on v(·) is selected on dev only; verifier removes records below τ and re-runs Hungarian record-grouping. Verifier-induced changes are tracked for the diagnostic suite.

**Stage 7 — Final decoding / normalization.**
NFKC + whitespace collapse + strict surface-form matching identical to the canonical evaluators; numbers are not normalized beyond NFKC (this is the eval contract). Output is written as both: (a) Doc2EDAG native event tables; (b) canonical role-value JSONL.

**Training objectives.**
L = λ₁ L_ev(t) + λ₁′ L_ev(t,ρ) + λ₂ L_mention + λ₃ L_plan + λ₄ L_decode + λ₅ L_verify + λ₆ L_coverage.

Pseudocode (inference):
```
encode(D) -> {h_i, x_{i,j}}
for t in event_types:
  if max_i sigmoid(s_{i,t}) < δ_t: skip
  for ρ in R_t:
     E_{t,ρ} = topK sentences by s_{i,t,ρ}
     e_{t,ρ} = role_attend(r_{t,ρ}, tokens(E_{t,ρ}))
  M = mention_extract(union(E_{t,·}))
  n_t = plan(t, D)
  R̂_t = decoder(t, M, {e_{t,ρ}}, n_t)
  R̂_t = [r for r in R̂_t if verifier(r, evidence(r)) >= τ]
yield event_table from R̂
```

# 6. Qwen-4B Usage Decision

**Decision: Use Qwen-4B (Qwen3-4B-Instruct or Qwen2.5-4B-Instruct, whichever is current on HuggingFace at submission time) as the record-level verifier (V2) trained with QLoRA, NOT as the main generator.**

Justification:
1. **Comparability.** The Doc2EDAG canonical metric is record-level fixed-slot F1; making the main extractor a discriminative EDAG-style decoder lets us report on the *exact same metric* used by Doc2EDAG/GIT/PTPCG/SEELE without any conversion artifacts. A LLM-as-extractor pipeline has well-documented JSON-format failure modes that introduce metric-mismatch noise.
2. **GPU feasibility.** Qwen-4B SFT (full) does not fit on 1×4090 at 4k context; QLoRA does (≈18 GB at rank 16, 4k seq), and our verifier only needs short evidence windows.
3. **Defensibility.** Reviewers can attack "Qwen as extractor" with: "you're just doing LLM_SFT, the DocFEE baseline already does this with Qwen1.5-7B-Chat." Using Qwen as verifier sidesteps this and adds a measurable ablation (with vs without verifier).
4. **Diagnostic value.** The verifier produces strict per-record support classifications that are independently auditable and feed the "unsupported argument rate" diagnostic.

We additionally run a **Qwen-4B QLoRA generative baseline** (Track C) end-to-end with XGrammar constrained decoding so we report a real, head-to-head LLM number on all three datasets. This is a baseline contribution, not the main method.

# 7. Training Plan

Stages and budgets (assuming 2×RTX 4090, 48 GB total):

| Stage | Inputs | Loss / target | Hyperparam ranges | Mem | Wallclock per dataset |
|---|---|---|---|---|---|
| 0. Preprocess + weak evidence labels | canonical JSONL + schema | — | n/a | CPU | 1–2 h |
| 1. Encoder + sentence×type evidence head | encoded docs | BCE(s_{i,t}) | lr 2e-5 to 3e-5, batch 16, 3–5 ep | 12 GB | ChFinAnn 6–8 h, DuEE-fin 3–4 h, DocFEE 10–14 h |
| 2. +Role evidence head | + sentence×type×ρ labels | weighted multilabel BCE; pos_weight tuned on dev | λ₁′ ∈ {0.5,1,2} | +2 GB | merged with stage 1 |
| 3. Mention CRF | role-evidence-restricted spans | CRF NLL | lr 2e-5 | 14 GB | 4–6 h |
| 4. Decoder D1 (EDAG-constrained) | mentions + role states | path-CE + coverage | lr 1e-4 head, 1e-5 enc | 20 GB on 1×4090 | ChFinAnn 12 h, DuEE-fin 6 h, DocFEE 18 h |
| 4′. Decoder D2 (mT5 + XGrammar) | doc + role states | seq2seq CE + copy mask | lr 5e-5 mT5, batch 4, grad-accum 8 | 22 GB | similar |
| 5. Verifier V1 (light) | record + evidence | 3-class CE | rank/dim 256 | 4 GB | 1–2 h |
| 5′. Verifier V2 (Qwen-4B QLoRA) | same | 3-class CE | r=16, α=32, bnb 4-bit NF4 | 18 GB | 6–10 h |
| 6. Joint fine-tune (optional) | end-to-end | weighted sum | smaller lr | — | 4–6 h |

Checkpoint selection: dev-set canonical role-value micro-F1 from `unified-strict` (NOT test). For ChFinAnn, also track Doc2EDAG-native fixed-slot F1 on dev. Early-stop on a 5-epoch patience. No test data is touched until camera-ready.

# 8. Baseline Matrix

| Dataset | Must-run (reproduced under our evaluators) | Strongly recommended | Reported-only | LLM baselines | Evaluator track | Risks |
|---|---|---|---|---|---|---|
| ChFinAnn | Doc2EDAG, GIT, PTPCG (via Spico197/DocEE), DE-PPN | RAAT/ReDEE, SEELE, IPGPF | TER-MCEE, TGIN, CAINet, SCDEE (numbers from papers) | Qwen-4B QLoRA SFT (ours, Track C); Qwen-4B + XGrammar zero-shot | legacy-doc2edag (primary) + unified-strict | DocEE toolkit version drift; ChFinAnn split conformance |
| DuEE-fin | Doc2EDAG, GIT, PTPCG, RAAT/ReDEE | SEELE, IPGPF | TER-MCEE, TGIN, CAINet, SIAT | Qwen-4B QLoRA SFT; Qwen-4B + XGrammar zero-shot | unified-strict (primary; no native fixed-slot baseline) | No public gold for raw test (kept blind); dev-set substitution must be transparent |
| DocFEE | DocFEE official baselines: BERT_Tagging, BART_QA (MRC), Qwen1.5-7B-Chat SFT (re-run from `tongzhou21/DocFEE`) | Doc2EDAG/GIT adapted to 9 event types, 38 roles via schema adapter | none widely published | Qwen-4B QLoRA SFT (ours); Qwen-4B + XGrammar zero-shot | docfee-official + unified-strict | First peer-reviewed model on DocFEE under Doc2EDAG-style metric ⇒ extra burden to be honest about no prior comparison |

For all must-runs, the protocol is: clone official repo at pinned commit, train with default hyperparameters on our deterministic splits, evaluate with our three evaluator tracks, archive seeds and logs.

# 9. Ablation Matrix

| # | Ablation | Module removed/replaced | Metric expected to move | Supports claim | Falsifies claim |
|---|---|---|---|---|---|
| A1 | — Sentence×type evidence head | Use uniform sentence pooling | record F1 ↓ 1–3 | Event detection benefits from sentence selection | Sentence selection helps |
| A2 | — Sentence×role evidence head | Use only type-level | record F1 ↓ 2–5 (esp. multi-event) | Role-grained evidence is the contribution | Role-grained > type-grained |
| A3 | — Coverage loss | drop L_coverage | F1 stable or ↓ ≤1 | Coverage prevents off-evidence picks | Coverage is necessary |
| A4 | EDAG decoder → mT5-XGrammar | swap D1↔D2 | dataset-dependent | Both decoder paths benefit from same evidence | Decoder choice dominates |
| A5 | — Verifier (V1 or V2) | skip verification step | precision ↓, F1 ↓ 0.5–2 | Record verification helps precision | Verifier is purely a re-ranker without gain |
| A6 | Verifier V1 vs V2 (Qwen) | swap | F1 ↑ with V2 mainly on DuEE-fin/DocFEE | Larger semantic verifier helps long-doc | Qwen verifier adds value |
| A7 | Encoder: Lawformer → 512-RoBERTa+sliding | swap on DocFEE | F1 ↓ 2–4 | Long-doc encoder needed on DocFEE | Long-context model matters |
| A8 | Weak evidence drop → distant only on first match | rule change | F1 ↓ 0.5–1.5 | Multilabel matters | Single-match suffices |
| A9 | Constrained decoding off (free JSON) | XGrammar off in D2 | structural violation rate ↑↑; F1 ↓ | Constrained decoding necessary for valid output | Grammar constraint helps |
| A10 | Cross-dataset transfer | train ChFinAnn → test DocFEE schema-aligned subset | F1 ↓ but non-trivial | Method generalizes | Method is dataset-specific |
| A11 | Planner removed | always n=1 record per type | recall ↓ on multi-event | Planner matters for multi-event | Planner adds value |
| A12 | NFKC normalization off in eval | strict eval change | confirms metric sensitivity | Reproducibility of metric | Metric is robust |

# 10. Diagnostic / Error Analysis Plan (12 analyses)

1. **Long-doc bucket analysis.** Split each test set by character-length quartile; report record F1 per bucket. Primary target: DocFEE Q4 (>3,000 chars).
2. **Single- vs multi-event split.** Following SEA / Doc2EDAG convention: P/R/F1 on docs with one record vs docs with ≥2 records per type.
3. **Per-event-type F1.** Heatmap across (dataset × event type).
4. **Per-role F1.** Sorted bar chart per dataset; identify hardest roles (typically EndDate, ratios, percentage roles).
5. **Record-grouping accuracy.** Hungarian-matched record-level exact match.
6. **Unsupported argument rate (strict).** An argument value v is "supported" iff after NFKC + whitespace collapse, v is a contiguous substring of the source text. Rate = (#unsupported predicted args) / (#predicted args). Always ≤ a small floor for D1 (entity-pointing) and a meaningful floor for D2 (generation). This is a *first-class diagnostic* and is reported even on negative findings.
7. **Evidence grounding rate.** Per predicted record, fraction of arguments whose first matching source sentence is also in the role's top-K evidence set produced by stage 2/3. Report mean and tail (worst 10%).
8. **Hallucinated argument rate.** Predicted argument that is not a substring of the document under strict normalization (equivalent to (6) for the generation track; for D1 always 0 by construction, which is itself a finding).
9. **Record split / merge errors.** For each gold record, the *split* error: gold record covered by ≥2 predicted records with overlapping arguments. *Merge* error: ≥2 gold records covered by one predicted record.
10. **Same-role cross-record confusion.** When two records share an argument role (e.g., two `Pledger` entries), measure swap rate between records.
11. **Qualitative error examples.** 20 case studies per dataset, three categories (long-distance dependency, multi-event interference, surface-form mismatch).
12. **Runtime / memory profile.** Train wallclock per epoch, peak GPU memory, inference docs/sec on a 4090 at fp16.

# 11. Minimum Viable EMNLP Experiment Package

The smallest publication-ready set:
1. EASV main model with V2 verifier on ChFinAnn, DuEE-fin, DocFEE — single seed plus dev F1 reproducibility check.
2. Reproduced baselines: Doc2EDAG and GIT on ChFinAnn; Doc2EDAG and PTPCG (or RAAT) on DuEE-fin; BERT_Tagging and Qwen1.5-7B-Chat SFT on DocFEE.
3. Qwen-4B QLoRA SFT track (C) as a single LLM baseline on all three datasets.
4. Three primary ablations: A2 (role-evidence), A5 (verifier), A7 (long-doc encoder on DocFEE).
5. All four key diagnostics: (1), (2), (6), (7).
6. Reproducibility statement: exact seeds, byte-identical splits, three evaluator tracks invoked from `dee-fin`.

If only this MVP is delivered, the paper is plausibly Findings/COLING; an EMNLP-main upgrade requires (a) statistically significant improvement vs at least one 2024 baseline (SEELE, CAINet, or TER-MCEE) on ChFinAnn or DuEE-fin under our metric, with 3 seeds and a paired bootstrap test, plus (b) a clearly positive result on DocFEE under the role-value protocol.

# 12. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Novelty too close to SEELE / TER-MCEE / Doc2Event | Medium-High | Make the verifier + diagnostic suite + three-dataset uniform-protocol the headline; ablate role-evidence vs SEELE's schema attention as the closest comparison; do not over-claim "first schema-aware" |
| Baseline drift (different code versions yield different reported F1) | High | Pin all baseline commits, log requirements lock files; report both our reproduced numbers and original-paper numbers in a side-by-side table |
| Metric mismatch on DuEE-fin (no gold test) | High | Substitute dev-as-test transparently with the deterministic 500-doc dev split (already user's choice); show the impact via a "leakage audit" |
| DocFEE adaptation: prior DEE codebases assume ChFinAnn schema | High | Build a schema adapter; release adapter as part of `dee-fin` to ease future comparison (independent contribution) |
| Qwen-4B underperformance vs Qwen1.5-7B-Chat in published DocFEE baseline | Medium | Be honest: report Qwen-4B SFT as a smaller-model reference; do not claim SOTA; show that the verifier (V2) leverages Qwen-4B in a place where 7B would not change the picture |
| Weak evidence supervision creates label leakage to evaluator | Medium | Disclose explicitly that evidence labels are a deterministic projection of the gold record; ablation A2 quantifies the impact; do not introduce alternative evaluators that use evidence |
| Verifier marginal gains (≤0.5 F1) cannot justify Section 5–6 | Medium | Report verifier as a precision tool with explicit "support rate" diagnostic; even if F1 gain is small, the diagnostic itself is a contribution |
| Reviewer rejection on "no new architecture" | Medium-High | Frame as a methodology + reproducibility + diagnostics paper; cite Peng et al. "The Devil is in the Details" as motivation; include the schema adapter and three evaluator tracks as artifacts |
| "JLF" / "MoE-ML-CNEE" claimed numbers turn out to be real and beat us | Low | We could not verify these; if they appear during review, treat as concurrent work and provide differentiation (verifier + DocFEE coverage) |

# 13. Paper Claim Draft

**Candidate titles.**
1. "EASV: Evidence-Anchored Schema-Constrained Records with Verifier for Chinese Financial Document-Level Event Extraction"
2. "Anchored, Constrained, Verified: A Reproducible Methodology for Document-Level Financial Event Extraction across Three Chinese Datasets"
3. "From Records to Evidence and Back: Auditable Document-Level Event Extraction for Chinese Financial Documents"

**Abstract-level claim (≈150 words).**
Chinese financial document-level event extraction (DEE) has matured around the ChFinAnn benchmark and the canonical Doc2EDAG metric, with newer datasets DuEE-fin and DocFEE expanding event coverage and document length. However, no prior work evaluates a single method uniformly across all three under strict surface-form evaluation, and few report record-level support diagnostics. We present EASV, a four-stage DEE framework that (i) supervises sentence×role evidence via deterministic projection of gold records, (ii) constrains record decoding by either an EDAG path-expander masked to evidence sentences or a JSON-schema-constrained generative head with strict copy decoding, and (iii) verifies each predicted record with a lightweight transformer or Qwen-4B QLoRA verifier. We report Doc2EDAG-native and unified-strict F1, plus four auditable diagnostics including a strict "unsupported argument rate". On ChFinAnn and DuEE-fin we match strong recent baselines; on DocFEE we provide the first Doc2EDAG-style result. All splits, evaluators, and weak-evidence labelers are released.

**Three contributions.**
1. **Method.** Evidence-anchored, schema-constrained record extraction with a record-level verifier; ablations isolate the role-evidence and verifier contributions.
2. **Reproducibility / evaluation.** Three independent evaluator tracks (Doc2EDAG-native, DocFEE-official, unified-strict) integrated into one repository, with byte-identical splits and a DocFEE↔Doc2EDAG schema adapter.
3. **Diagnostics.** First report of strict "unsupported argument rate" and "evidence grounding rate" for Chinese financial DEE, across length and multi-event slices.

**Claims to avoid.**
- "State-of-the-art on ChFinAnn" — unlikely under strict surface-form comparison to 2024 methods.
- "First schema-aware DEE" — SEELE (IJCAI 2024) already uses schema descriptions.
- "First LLM-based DEE on DocFEE" — the DocFEE paper itself reports Qwen1.5-7B-Chat SFT.
- "First evidence-grounded DEE" — ExpDEE and SEA already explore sentence-level clues.
- Citing the unverified "JLF +10.6 / +5.6 F1" or "MoE-ML-CNEE" numbers.

# 14. Implementation Roadmap (Phases 0–8)

| Phase | Goal | Acceptance criteria | Calendar (FTE-weeks) | Deliverables |
|---|---|---|---|---|
| 0. Infra audit | Confirm `dee-fin` evaluator parity, split hashes, schema files | All three evaluators reproduce known numbers on dev | 1 wk | Audit report; pinned dependencies |
| 1. Baseline reproduction | Doc2EDAG, GIT, PTPCG on ChFinAnn; PTPCG/RAAT on DuEE-fin; DocFEE official BERT_Tagging, BART_QA, Qwen1.5-7B-Chat SFT | Reproduced F1 within ±1 of published, or documented divergence | 2–3 wks | `baselines/` integration scripts |
| 2. Weak-evidence labeler + encoder + stage 1–2 | Sentence×type and sentence×type×role heads | Sentence recall@8 ≥ 0.9 of role-supporting sentences on dev | 1–2 wks | weak labels, evidence heads checkpoints |
| 3. Mention extractor + planner | CRF + planner | Mention F1 within 1 pt of Doc2EDAG NER head | 1 wk | mention/planner checkpoints |
| 4. EDAG-constrained decoder D1 | Evidence-restricted EDAG expansion | dev F1 ≥ Doc2EDAG reproduced | 2 wks | D1 model |
| 4′. mT5+XGrammar decoder D2 | Schema-constrained gen with copy | structural validity 100% | 2 wks | D2 model |
| 5. Verifier V1 + V2 (Qwen-4B QLoRA) | Verifier head + Qwen QLoRA | dev precision ↑ ≥ 1 pt without recall ↓ ≥ 1 pt | 2 wks | verifier checkpoints + 3-class metric |
| 6. Diagnostics suite | 12 analyses | All 12 produce stable numbers across 3 seeds | 1–2 wks | `diagnostics/` reports |
| 7. Cross-dataset run + ablations | A1–A12 | All ablations finished, with dev curves | 2–3 wks | ablation tables |
| 8. Paper drafting + camera-ready prep | Submission | Internal review pass | 2 wks | submission package |

Total: ~14–18 FTE-weeks. Feasible on 1–2 4090s end-to-end with overlap. The roadmap is structured for Codex CLI / Claude Code style execution: each phase has a single acceptance criterion that can be encoded as a CI check on the dev set.

# 15. Final Self-Check

1. **Is the method really above evaluator-only?** Yes. Three trained components (evidence heads, decoder, verifier) plus a schema adapter and a Qwen-4B QLoRA verifier integration constitute a method paper, not just an evaluation paper. However, the *method's* novelty is incremental over SEELE+IPGPF+TER-MCEE; the evaluation/reproducibility/diagnostic contribution is what makes the package competitive.
2. **Feasible on 1–2 × 4090?** Yes. Largest single training (mT5-base + XGrammar + Lawformer encoder on DocFEE) is ≈22 GB; Qwen-4B QLoRA verifier at 4k context is ≈18 GB.
3. **All three datasets covered?** Yes — and DocFEE inclusion is a genuine differentiator since no peer-reviewed DEE method outside the DocFEE paper itself has reported on it under Doc2EDAG-style metrics.
4. **Native vs. unified-strict separated?** Yes. Both are reported as primary tables for ChFinAnn; only unified-strict is primary for DuEE-fin/DocFEE; this is disclosed in §11 and is the user's existing repo design.
5. **Baselines honest?** Yes. Reproduction divergences will be reported transparently. Numbers from papers that we did not rerun are clearly labeled "reported".
6. **Recent papers checked?** Yes for 2024–2025: SEELE, CAINet, TGIN, TER-MCEE, SIAT, EADRE, LAAP, Doc2Event, SALE, ASEE, DocFEE. JLF and MoE-ML-CNEE could not be verified and are flagged.
7. **Unsupported claims avoided?** Yes. We never claim SOTA without 3-seed bootstrap evidence, never claim "first schema-aware DEE", never cite unverified JLF/MoE-ML-CNEE numbers.
8. **EMNLP main vs Findings / COLING / SCI Q2?** *Honest answer:* as planned, this is a **strong Findings / COLING submission**, not a guaranteed EMNLP-main paper. The minimum upgrade to credibly target EMNLP main is one of:
   - **Upgrade A (preferred):** add a clear methodological novelty beyond evidence+verifier — e.g., a *learned* evidence prior trained jointly with the verifier using contrastive negatives mined from cross-record swaps (this is genuinely beyond SEELE/IPGPF and yields a single-headline F1 result).
   - **Upgrade B:** a statistically significant +1.5 F1 on ChFinAnn over GIT/SEELE with 3 seeds, paired bootstrap, AND a positive cross-dataset transfer result.
   - **Upgrade C:** turn the diagnostic suite (unsupported-argument rate, evidence grounding rate) into a *new evaluation protocol* paper with its own analysis on multiple model families — this is more likely to land at COLING/LREC than EMNLP main.

Without one of these upgrades, target Findings / COLING / a Q2 venue (Information Sciences, Knowledge-Based Systems, Neurocomputing) rather than EMNLP main.