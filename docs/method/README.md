# Method Design Versions

All documents in this directory are proposals only. The `dee-fin` repository currently provides repository evidence for data splits and evaluator tracks only.

Evaluator track wording:

- ChFinAnn: legacy-doc2edag + unified-strict
- DuEE-Fin: historical-compatible / Doc2EDAG-style if supported + unified-strict
- DocFEE: docfee-official + unified-strict

No CARVE / EASV / ECPD-CRV model components, training scripts, Sinkhorn allocation layer, share gate, grounding heads, EDAG decoder, verifier, or diagnostics harness have been implemented yet.

## Version Status

| Document | Presence | Status | Notes |
|---|---|---|---|
| `easv_v1.md` | Present | Superseded proposal | Earlier pipeline-style proposal. Keep for design history only. |
| `ecpd_crv_v0.md` | Not present in this checkout | Superseded proposal if restored | Intermediate cross-record-aware hard-mask proposal. |
| `carve_method_design_v1_1.md` | Not present in this checkout | Superseded proposal if restored | Early CARVE allocation-prior draft. |
| `carve_method_design_v1_2.md` | Not present in this checkout | Superseded proposal if restored | Added `L_alloc`, share-gate definitions, and diagnostic clarifications. |
| `carve_method_design_v1_3.md` | Present | Current recommended proposal | Frozen method proposal. Not implemented. |

## Next-Step Execution Policy

Do not implement CARVE end-to-end yet.

The next step is the minimal validation chain:

```text
P0 -> P1 -> P4 -> P5a -> P5b
```

The future P5b result, when measured, should be recorded in `docs/measurements/p5b_decision_table.md`. Until then, `docs/measurements/p5b_decision_table_template.md` is only a template and contains no measured result.

The future P1 memory result, when measured, should be recorded in `docs/measurements/p1_memory.md`. Until then, `docs/measurements/p1_memory_template.md` is only a template and contains no measured GPU number.

## Repository Fact Boundary

The following must not be stated as already implemented until corresponding code and tests exist:

- CARVE model components
- EASV model components
- ECPD-CRV model components
- weak-evidence labeler
- Sinkhorn allocation layer
- share gate
- EDAG decoder
- allocation verifier
- Qwen-4B verifier
- diagnostics harness
- end-to-end training scripts
- end-to-end inference scripts
- reproduced full baseline matrix
- measured GPU memory numbers
