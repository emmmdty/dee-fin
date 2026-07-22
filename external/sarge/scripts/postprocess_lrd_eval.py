"""Post-process candidates with LRD / rule / no planner; write canonical preds.

Inputs:
  - parsed_candidates.<split>.jsonl from a SARGE inference run
  - dataset + split (used to recover canonical doc_text via stage_dataset)
  - planner mode: ``none`` | ``rule`` | ``lrd``
  - for ``lrd``: LRD checkpoint + RoBERTa path

Output: a new run-style directory under ``--out`` with
``predictions/<dataset>/<split>.canonical.pred.jsonl`` so the existing
``scripts/eval_three_tracks.py`` can evaluate it.

Example:
    python scripts/postprocess_lrd_eval.py \\
        --candidates runs/sarge_infer_DuEE-Fin-dev500_dev_<ts>/intermediate/getm/parsed_candidates.dev.jsonl \\
        --dataset DuEE-Fin-dev500 --split dev \\
        --planner lrd \\
        --lrd-ckpt runs/lrd/dueefin_train_seed13/checkpoints/lrd_planner.pt \\
        --roberta models/chinese-roberta-wwm-ext_safetensors \\
        --out runs/sarge_postlrd_DuEE-Fin-dev500_dev_seed13/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from sarge.data.schema import load_schema  # noqa: E402
from sarge.data.staging import stage_dataset  # noqa: E402
from sarge.postprocess.lrd_planner import LRDConfig, LRDPlanner  # noqa: E402
from sarge.postprocess.rule_planner import EventRecord, apply_planner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--planner", choices=("none", "rule", "lrd"), required=True)
    parser.add_argument("--lrd-ckpt", default=None)
    parser.add_argument("--roberta", default=None,
                        help="RoBERTa path; required when --planner lrd")
    parser.add_argument("--rule-mode", default="conservative_assembler",
                        help="rule planner mode (pass_through / dedup_only / conservative_assembler)")
    parser.add_argument("--tau-override", type=float, default=None,
                        help="override LRD merge_thresholds with a constant value (e.g. 0.85) for tuning")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, help="output run directory")
    args = parser.parse_args()

    if args.planner == "lrd" and (not args.lrd_ckpt or not args.roberta):
        parser.error("--planner lrd requires --lrd-ckpt and --roberta")

    out_root = Path(args.out)
    pred_dir = out_root / "predictions" / args.dataset
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{args.split}.canonical.pred.jsonl"

    data_root = Path(args.data_root).resolve()
    schema = load_schema(args.dataset, data_root=data_root)

    # Stage canonical dataset to recover doc text per doc_id.
    staging = Path(tempfile.mkdtemp(prefix="sarge_postlrd_"))
    try:
        stage_dataset(
            dataset=args.dataset, processed_root=data_root,
            output_root=staging, splits=(args.split,),
        )
        doc_text_by_id = _load_doc_texts(staging / args.dataset / f"{args.split}.jsonl")
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    candidates_by_doc = _load_candidates(args.candidates)

    planner_runtime = None
    if args.planner == "lrd":
        planner_runtime = _load_lrd_planner(
            args.lrd_ckpt, args.roberta, schema, device=torch.device(args.device),
        )
        if args.tau_override is not None:
            with torch.no_grad():
                planner_runtime.merge_thresholds.fill_(float(args.tau_override))
            print(f"[lrd] merge_thresholds overridden to tau={args.tau_override}")

    n_in = 0
    n_out = 0
    n_docs = 0
    written = 0
    with pred_path.open("w", encoding="utf-8") as handle:
        for doc_id in sorted(doc_text_by_id.keys()):
            doc_text = doc_text_by_id[doc_id]
            rows = candidates_by_doc.get(doc_id) or []
            records = _flatten_to_event_records(rows)
            n_in += len(records)
            n_docs += 1
            if args.planner == "none":
                planned = records
            elif args.planner == "rule":
                planned, _ = apply_planner(records, mode=args.rule_mode, schema=schema)
            else:  # lrd
                assert planner_runtime is not None
                if len(records) <= 1:
                    planned = records
                else:
                    planned, _ = planner_runtime.disambiguate(records, doc_text=doc_text)
            n_out += len(planned)
            row = {
                "doc_id": doc_id,
                "events": [rec.to_canonical() for rec in planned],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"planner={args.planner} docs={written} "
        f"events_in={n_in} events_out={n_out} pred={pred_path}"
    )
    return 0


def _load_candidates(path: str) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row.get("doc_id") or ""
            if not doc_id:
                continue
            result.setdefault(doc_id, []).append(row)
    return result


def _load_doc_texts(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row.get("doc_id") or ""
            if doc_id:
                result[doc_id] = str(row.get("content") or "")
    return result


def _flatten_to_event_records(rows: list[dict]) -> list[EventRecord]:
    records: list[EventRecord] = []
    for row in rows:
        for ev in row.get("events") or []:
            if not isinstance(ev, dict):
                continue
            event_type = str(ev.get("event_type") or "").strip()
            if not event_type:
                continue
            args = ev.get("arguments") or {}
            normalised: dict[str, list[dict[str, str]]] = {}
            if isinstance(args, dict):
                for role, values in args.items():
                    for v in values or []:
                        text = v.get("text") if isinstance(v, dict) else str(v)
                        if text:
                            normalised.setdefault(str(role), []).append({"text": str(text)})
            records.append(EventRecord(event_type=event_type, arguments=normalised))
    return records


def _load_lrd_planner(
    ckpt_path: str, roberta_path: str, schema, *, device: torch.device,
) -> LRDPlanner:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg: LRDConfig = ckpt["config"]
    # Replace the frozen encoder_config so RoBERTa loads from the real path.
    real_encoder_cfg = dataclasses.replace(
        saved_cfg.encoder_config,
        model_path=roberta_path,
    )
    cfg = dataclasses.replace(saved_cfg, encoder_config=real_encoder_cfg)
    planner = LRDPlanner(cfg, schema)
    planner.scorer.load_state_dict(ckpt["scorer"])
    planner.encoder.projection.load_state_dict(ckpt["encoder_projection"])
    planner.encoder.role_embed.load_state_dict(ckpt["encoder_role_embed"])
    with torch.no_grad():
        planner.merge_thresholds.copy_(ckpt["merge_thresholds"])
    planner.to(device)
    planner.eval()
    planner.encoder._ensure_encoder()  # force RoBERTa load on `device`
    return planner


if __name__ == "__main__":
    raise SystemExit(main())
