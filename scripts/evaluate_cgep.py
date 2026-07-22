#!/usr/bin/env python
"""Cross-validate a successor-event predictor on CGEP-ESC or CGEP-MAVEN.

ESC is evaluated by cross-validation over EventStoryLine *topics*, not documents:
documents inside one topic narrate the same story, so a document split leaks the
causal chain across train and test. SeDGPL's released `ESCSubWoRe.npy` carries no
train/valid/test keys at all (its `load_data.py` expects some anyway), so which
split produced its published ESC MRR of 19.6 was unknown.

Measured here (5-fold), it was a leaky one. Our reimplementation scores MRR
0.0599 +/- 0.0138 under topic CV and 0.1802 +/- 0.0089 on a document split -- the
latter matching the published 0.196. Learning rate is not the cause (1e-6 -> 5e-6
*lowers* topic-CV MRR to 0.0701) and neither is candidate difficulty (the
`random` baseline is 0.0286 vs 0.0288 across the two splits). What moves is what a
document split shares: the mention prior (`frequency` goes 0.0217 -> 0.0491) and
the topic's causal chain. Report topic CV; keep `--split-mode document` for that
comparison only.

Both tie-break conventions are reported: `mrr` is SeDGPL's (ties go to gold),
`mrr_strict` charges gold for every tie. The gap is not small -- 0.1802 vs 0.1204
on the document split -- so the published figure is inflated twice over.

    uv run python scripts/evaluate_cgep.py --dataset esc --predictor frequency
    uv run python scripts/evaluate_cgep.py --dataset maven --predictor frequency
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev

from finekg.succession.data.cgep import CgepInstance, build_cgep, iter_documents
from finekg.succession.data.esc import load_cgep_esc, topic_folds
from finekg.succession.linearize import EDGE_BUDGET
from finekg.succession.predictor import evaluate, successor_predictors

_ESC = Path("data/raw/sedgpl_esc/ESCSubWoRe.npy")
_MAVEN = Path("data/processed/maven_ere")

_HEADLINE = ("mrr", "hits@1", "hits@3", "hits@10", "mrr_strict", "hits@1_strict")


def _esc_folds(n_folds: int, split_mode: str, seed: int = 209):
    """Topic folds (the EventStoryLine convention) or document folds (leaky).

    A document-level split puts documents from the *same story* on both sides:
    ESC's topics are one narrative each, so train and test then share the causal
    chain and much of the event vocabulary. It is included only to test whether
    the published ESC number depends on that leakage -- never as our protocol.
    """
    if not _ESC.exists():
        raise SystemExit(f"missing {_ESC}; see docs/GPU_RUNBOOK.md")
    by_topic = load_cgep_esc(_ESC)

    if split_mode == "topic":
        folds = []
        for train_topics, test_topics in topic_folds(list(by_topic), n_folds):
            folds.append((
                [i for t in train_topics for i in by_topic[t]],
                [i for t in test_topics for i in by_topic[t]],
            ))
        return folds

    by_doc: dict[str, list[CgepInstance]] = {}
    for instances in by_topic.values():
        for instance in instances:
            by_doc.setdefault(instance.doc_id, []).append(instance)
    docs = sorted(by_doc)
    random.Random(seed).shuffle(docs)
    folds = []
    for fold in range(n_folds):
        test_docs = set(docs[fold::n_folds])
        folds.append((
            [i for d in docs if d not in test_docs for i in by_doc[d]],
            [i for d in docs if d in test_docs for i in by_doc[d]],
        ))
    return folds


def _maven_folds() -> list[tuple[list[CgepInstance], list[CgepInstance]]]:
    """MAVEN keeps its official split: train fits, valid reports (test is hidden)."""
    train, _ = build_cgep(iter_documents([str(_MAVEN / "train.jsonl")]))
    valid, _ = build_cgep(iter_documents([str(_MAVEN / "valid.jsonl")]))
    return [(train, valid)]


def _build(args, train: list[CgepInstance], test: list[CgepInstance]):
    """`sedgpl` registers on import; the baselines take no arguments.

    Its `<a_i>` vocabulary must span train *and* test: a test candidate whose
    mention never occurs in training would otherwise have no token, and the
    instance could not be encoded at all. SeDGPL ships `to_add.json` built the
    same way. Only the token inventory crosses the split.
    """
    if args.predictor != "sedgpl":
        return successor_predictors.create(args.predictor)
    if not args.model_path:
        raise SystemExit("--predictor sedgpl needs --model-path")
    import finekg.succession.sedgpl  # noqa: F401 - registers "sedgpl"
    from finekg.succession.linearize import EventVocabulary

    return successor_predictors.create(
        "sedgpl", model_path=args.model_path,
        vocabulary=EventVocabulary.build([*train, *test]),
        epochs=args.epochs, sample_rate=args.sample_rate, device=args.device, lr=args.lr,
        edge_selector=args.edge_selector, max_edges=args.max_edges,
        enable_structure=args.structure_encoding,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="esc", choices=("esc", "maven"))
    parser.add_argument("--predictor", default="frequency",
                        choices=("random", "frequency", "sedgpl"))
    parser.add_argument("--folds", type=int, default=5, help="ESC topic folds")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-path", help="roberta-base checkpoint, for sedgpl")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sample-rate", type=float, default=0.8, help="SeDGPL's fewShot rate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-6, help="run.sh uses 1e-6 and 5e-6")
    parser.add_argument("--edge-selector", default="sedgpl", choices=("sedgpl", "distance"),
                        help="M1: 'distance' keeps the edges nearest the query; "
                             "'sedgpl' (default) slices the stored head")
    parser.add_argument("--max-edges", type=int, default=EDGE_BUDGET,
                        help="linearisation budget (SeDGPL uses 20)")
    parser.add_argument("--structure-encoding", action="store_true",
                        help="M2: add EeCE's fourth stream (reach-anchor bit); "
                             "off by default keeps the baseline byte-identical")
    parser.add_argument("--split-mode", default="topic", choices=("topic", "document"),
                        help="document folds leak a topic's story across the split")
    parser.add_argument("--max-folds", type=int, help="stop after N folds (smoke runs)")
    parser.add_argument("--limit-train", type=int, help="cap train instances (smoke runs)")
    parser.add_argument("--limit-test", type=int, help="cap test instances (smoke runs)")
    args = parser.parse_args()

    folds = (_esc_folds(args.folds, args.split_mode) if args.dataset == "esc"
             else _maven_folds())
    if args.max_folds:
        folds = folds[: args.max_folds]

    per_fold: list[dict[str, float]] = []
    for index, (train, test) in enumerate(folds):
        if args.limit_train:
            train = train[: args.limit_train]
        if args.limit_test:
            test = test[: args.limit_test]
        predictor = _build(args, train, test)
        predictor.fit(train)
        metrics = evaluate(predictor, test)
        per_fold.append(metrics)
        notes = ""
        if getattr(predictor, "skipped_train", 0):
            notes += f"  train_skipped={predictor.skipped_train}"
        if metrics["n_unscorable"]:
            notes += f"  UNSCORABLE={int(metrics['n_unscorable'])}"
        print(
            f"[cgep] fold {index}  n_train={len(train):5d} n_test={len(test):5d}  "
            + "  ".join(f"{k}={metrics[k]:.4f}" for k in ("mrr", "hits@1", "mrr_strict"))
            + notes
        )

    mode = f" / {args.split_mode}-split" if args.dataset == "esc" else ""
    print(f"\n{args.dataset} / {args.predictor}{mode} / {len(folds)} fold(s)")
    summary: dict[str, float] = {}
    for key in _HEADLINE:
        values = [m[key] for m in per_fold]
        summary[key] = mean(values)
        summary[f"{key}_sd"] = pstdev(values) if len(values) > 1 else 0.0
        spread = f" ± {summary[f'{key}_sd']:.4f}" if len(values) > 1 else ""
        print(f"  {key:16s} {summary[key]:.4f}{spread}")
    if args.dataset == "esc":
        print("\n  reference: SeDGPL reports ESC MRR 19.6 on an unstated split "
              "-- not a pass/fail threshold")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"dataset": args.dataset, "predictor": args.predictor,
                   "folds": per_fold, "summary": summary}
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[cgep] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
