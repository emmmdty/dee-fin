#!/usr/bin/env python
"""GRPO post-training for the relation extractor with verifiable rewards (server / CUDA).

The verifier-as-reward stage (docs/archive/RL_DESIGN.md §2): after SFT
(`scripts/train_relation_extractor.py`), sample G completions per event-window
prompt and reward them with the verifier kernels — format + evidence grounding
+ global consistency + gold F1. Phases follow the easy-to-hard curriculum from
the config; each phase continues from the previous adapter and saves a
checkpoint, its phase-local per-component reward means, and the windowed
per-component reward curve (watch that curve: one component collapsing while
the total climbs is reward hacking).

The final adapter dir drops into `adapter_path` of
`configs/relations/llm_grounded_consistent.yaml` — evaluation is unchanged.

GPU + the `llm` + `rl` extras required (the default rollout backend also
expects `trl vllm-serve` on a second GPU; see the `rollout:` section):
    uv run --extra llm --extra rl python scripts/train_relation_grpo.py \
        --config configs/relations/grpo_rlvr.yaml \
        --output runs/relation_grpo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finekg.core.config import load_config
from finekg.relations.data import load_ccks_causal, load_maven_ere
from finekg.relations.rl.dataset import build_grpo_dataset, to_rows
from finekg.relations.rl.rewards import build_relation_reward
from finekg.relations.rl.trl_adapter import TrlRewardAdapter
from finekg.rl.curriculum import phase_indices, phases_from_config, seeded_order


def _load_docs(loader: str, path: Path) -> list:
    if loader == "maven_ere":
        return list(load_maven_ere(path))
    if loader == "ccks_causal":
        return list(load_ccks_causal(path))
    raise ValueError(f"unknown relations loader: {loader!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train", type=Path, help="override data.path from the config")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--base-model",
        type=str,
        help="override relations_rl.base_model (e.g. a local offline model dir)",
    )
    parser.add_argument(
        "--resume-from-phase",
        type=int,
        default=0,
        help="skip curriculum phases < N, chaining the warm-start from their saved "
        "adapter dirs under --output (re-run phases 1+ after a crash without redoing phase 0)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    section = cfg["relations_rl"]
    if args.base_model:
        section["base_model"] = args.base_model
    curriculum = section.get("curriculum", {})

    docs = _load_docs(cfg["data"]["loader"], args.train or Path(cfg["data"]["path"]))
    samples, store = build_grpo_dataset(
        docs, window_events=int(curriculum.get("window_events", 12))
    )
    print(f"[grpo] {len(samples)} window prompts from {len(docs)} documents")

    composite = build_relation_reward(section["rewards"])
    reward_fn = TrlRewardAdapter(composite, store)
    print(f"[grpo] reward components: {composite.names}")

    phases = phases_from_config(
        curriculum.get("phases", [{"max_difficulty": 1e9, "steps": 0}])
    )
    buckets = phase_indices([s.difficulty for s in samples], phases)
    seed = int(section.get("grpo", {}).get("seed", 42))

    # Heavy stack stays behind the entry point.
    from finekg.relations.rl.trainer import build_grpo_trainer

    adapter_path = section.get("sft_adapter_path") or None
    args.output.mkdir(parents=True, exist_ok=True)
    for i, (phase, bucket) in enumerate(zip(phases, buckets, strict=True)):
        phase_dir = args.output / f"phase{i}"
        if i < args.resume_from_phase:
            # Already trained in a previous invocation: chain the warm-start from its
            # saved adapter so phase N resumes from phase N-1 without redoing phase 0.
            if phase_dir.exists():
                adapter_path = str(phase_dir)
            print(
                f"[grpo] phase {i}: skipped (resume-from-phase {args.resume_from_phase}), "
                f"warm-start chain -> {adapter_path}"
            )
            continue
        if not bucket:
            print(f"[grpo] phase {i}: no samples <= difficulty {phase.max_difficulty}, skipped")
            continue
        order = seeded_order(len(bucket), seed + i)
        rows = to_rows([samples[bucket[j]] for j in order])
        print(
            f"[grpo] phase {i}: {len(rows)} prompts (difficulty <= {phase.max_difficulty}), "
            f"steps={phase.steps or 'full epoch'}, warm-start={adapter_path or 'fresh LoRA'}"
        )
        trainer = build_grpo_trainer(
            model_name=section["base_model"],
            rows=rows,
            reward_fn=reward_fn,
            cfg=section,
            output_dir=str(phase_dir),
            sft_adapter_path=adapter_path,
            max_steps=phase.steps or None,
        )
        trainer.train()
        trainer.save_model(str(phase_dir))
        # The vLLM-server weight-sync group is one-shot per server lifetime; release it
        # so the next phase's GRPOTrainer can re-init against the same server. Without
        # this the server keeps the phase-N group and phase N+1's init_communicator
        # blocks until a 300s NCCL store timeout (DistStoreError), killing the run.
        client = getattr(trainer, "vllm_client", None)
        if client is not None:
            client.close_communicator()
        adapter_path = str(phase_dir)
        # Phase-local means (not cumulative across phases) plus the windowed
        # per-component curve — the reward-hacking watch and the W3-4 gate.
        (args.output / f"reward_means_phase{i}.json").write_text(
            json.dumps(reward_fn.phase_means(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        reward_fn.mark_phase()
        (args.output / "reward_curve.json").write_text(
            json.dumps(reward_fn.curve(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "adapter_path": adapter_path,
                "reward_means": reward_fn.component_means(),
                "reward_curve": "reward_curve.json",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[grpo] final adapter -> {adapter_path}")
    print("[grpo] evaluate: set it as adapter_path in llm_grounded_consistent.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
