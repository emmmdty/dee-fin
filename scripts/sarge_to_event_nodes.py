#!/usr/bin/env python
"""Adapt SARGE financial-application predictions into Fin-EKG event-node records.

Under v4, SARGE belongs to Phase G and does not replace Ch1. The flat records
produced here are consumed through ``finekg.core.io.event_nodes_from_sarge`` and
the application graph pipeline; historical raw graph artifacts remain readable.

SARGE's frozen canonical format is nested and surface-only:

    {"doc_id": ..., "events": [{"event_type": ..., "arguments": {role: [{"text": v}]}}]}

The event-graph builder needs one flat record per event with a `subject` (the
company/actor the event hangs on, for cross-doc timelines) and a `time_anchor`
(an ISO date, for temporal ordering). Neither is an explicit SARGE field, so we
derive them from the arguments: the subject from company/actor-typed roles, the
time anchor by parsing a date-typed role's value. This is pure CPU — it reuses
the LLM extraction SARGE already produced, no new inference.

    uv run python scripts/sarge_to_event_nodes.py \
        --pred <sarge_run>/predictions/DuEE-Fin-dev500/dev.canonical.pred.jsonl \
        --source-docs <sarge_data>/DuEE-Fin-dev500/dev.jsonl \
        --output data/raw/event_graph_zh/event_nodes.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Role-name cues. A role counts as the subject if its name contains an actor cue,
# as the time anchor if it contains a date cue. Ordered by preference.
_SUBJECT_CUES = ("回购方", "增持方", "减持方", "收购方", "发行", "上市公司", "公司", "企业",
                 "股东", "方", "机构", "主体", "被约谈", "破产")
_DATE_CUES = ("时间", "日期", "日")
# Counterparty roles: the *other* entity an event binds to, which turns a
# (company, HAS_EVENT, event_type) timeline fact into an entity-level
# (subject, event_type, object) fact. Ordered by preference.
_OBJECT_CUES = ("被收购方", "收购标的", "标的公司", "被投资方", "中标标的", "中标公司",
                "招标方", "交易对手", "质押方", "被质押方", "高管姓名", "公司名称", "股票简称")


def _iso_date(value: str | None) -> str | None:
    """Parse a Chinese/ISO date string to 'YYYY-MM-DD' (month-only -> day 01)."""
    if not value:
        return None
    m = re.search(r"(\d{4})\s*[年\-/.]\s*(\d{1,2})\s*[月\-/.]\s*(\d{1,2})", value)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    m = re.search(r"(\d{4})\s*[年\-/.]\s*(\d{1,2})\s*月", value)
    if m:
        y, mo = (int(x) for x in m.groups())
        if 1 <= mo <= 12:
            return f"{y:04d}-{mo:02d}-01"
    return None


def _first_text(values: object) -> str:
    if isinstance(values, list) and values and isinstance(values[0], dict):
        return str(values[0].get("text", "")).strip()
    return ""


def _record_doc_id(record: dict) -> str:
    for key in ("doc_id", "id", "document_id", "text_id"):
        value = record.get(key)
        if value:
            return str(value)
    return ""


def _load_meta(path: Path | None) -> dict[str, dict]:
    """doc_id -> sidecar record (date/stock) from the news exporter's JSONL."""
    if path is None:
        return {}
    meta: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        doc_id = _record_doc_id(record)
        if doc_id:
            meta[doc_id] = record
    return meta


def _load_source_docs(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    docs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        doc_id = _record_doc_id(record)
        text = str(record.get("text", ""))
        if doc_id and text:
            docs[doc_id] = text
    return docs


def _span_from_item(item: dict, doc_id: str, fallback_text: str = "") -> dict | None:
    text = str(item.get("text", fallback_text)).strip()
    if "span" in item and isinstance(item["span"], list) and len(item["span"]) >= 2:
        start, end = item["span"][:2]
    elif "char_start" in item:
        start = item["char_start"]
        end = item.get("char_end", item.get("end", int(start) + len(text)))
    elif "start" in item:
        start = item["start"]
        end = item.get("end", item.get("char_end", int(start) + len(text)))
    else:
        return None

    start_i, end_i = int(start), int(end)
    if end_i <= start_i:
        return None
    return {
        "doc_id": str(item.get("doc_id", doc_id) or doc_id),
        "char_start": start_i,
        "char_end": end_i,
        "sent_id": item.get("sent_id"),
        "text": text,
    }


def _locate_span(value: str, doc_id: str, source_text: str) -> dict | None:
    if not value or not source_text:
        return None
    start = source_text.find(value)
    if start < 0:
        return None
    return {
        "doc_id": doc_id,
        "char_start": start,
        "char_end": start + len(value),
        "sent_id": None,
        "text": value,
    }


def _trigger_evidence(event: dict, doc_id: str) -> list[dict]:
    spans: list[dict] = []
    for item in event.get("trigger_evidence") or []:
        if not isinstance(item, dict):
            continue
        span = _span_from_item(item, doc_id, str(event.get("trigger", "")))
        if span:
            spans.append(span)
    return spans


def _confidence(event: dict) -> float:
    """Extractor score for this event, or 1.0 when the extractor emits none.

    SARGE's canonical format carries no score, so every node lands on 1.0 and a
    confidence-thresholding admission policy (`relations.admission`) degenerates
    to admit-all. That is a property of the upstream extractor, not something to
    paper over with a synthetic score -- `main` reports whether any signal
    survived so the degeneracy is visible before calibration consumes it.
    """
    for key in ("confidence", "score"):
        raw = event.get(key)
        if isinstance(raw, int | float) and 0.0 <= float(raw) <= 1.0:
            return float(raw)
    return 1.0


def _argument_evidence(
    raw_arguments: dict[str, object], doc_id: str, source_text: str
) -> dict[str, list[dict]]:
    evidence: dict[str, list[dict]] = {}
    for role, values in raw_arguments.items():
        role_spans: list[dict] = []
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            value = str(item.get("text", "")).strip()
            span = _span_from_item(item, doc_id, value) or _locate_span(value, doc_id, source_text)
            if span:
                role_spans.append(span)
        if role_spans:
            evidence[role] = role_spans
    return evidence


def _derive_subject(arguments: dict[str, str]) -> str | None:
    for cue in _SUBJECT_CUES:
        for role, value in arguments.items():
            if cue in role and value:
                return value
    # fallback: first non-date argument value
    for role, value in arguments.items():
        if value and not any(c in role for c in _DATE_CUES):
            return value
    return None


def _derive_object(arguments: dict[str, str], subject: str | None = None) -> str | None:
    """The counterparty entity of the event, or None for unary events.

    Exact role match comes first because the cue strings nest: "收购方" is a
    substring of "被收购方", so a plain `cue in role` scan would let the acquirer
    masquerade as the acquiree whenever the argument dict happens to iterate that
    way. Values equal to `subject` are rejected for the same reason.
    """
    for cue in _OBJECT_CUES:
        value = arguments.get(cue)
        if value and value != subject:
            return value
    for cue in sorted(_OBJECT_CUES, key=len, reverse=True):
        for role, value in arguments.items():
            if cue in role and value and value != subject:
                return value
    return None


def _derive_time_anchor(arguments: dict[str, str]) -> str | None:
    for role, value in arguments.items():
        if any(c in role for c in _DATE_CUES):
            iso = _iso_date(value)
            if iso:
                return iso
    return None


# Domain lexicon: event type -> surface cues that must appear in the source
# text for the label to count as grounded (--require-type-cue). Derived from
# the DuEE-Fin schema's type semantics; unknown types fail open (kept).
_TYPE_CUES: dict[str, tuple[str, ...]] = {
    "股份回购": ("回购",),
    "股东增持": ("增持",),
    "股东减持": ("减持",),
    "企业收购": ("收购", "并购", "受让", "竞购", "要约"),
    "企业融资": ("融资", "募资", "募集", "定增", "增发"),
    "质押": ("质押",),
    "解除质押": ("解除质押", "解押", "解除了质押"),
    "中标": ("中标", "中选"),
    "公司上市": ("上市", "IPO", "挂牌", "招股"),
    "亏损": ("亏损", "预亏", "由盈转亏", "净亏"),
    "高管变动": ("辞职", "离职", "辞任", "任命", "聘任", "换届", "变动"),
    "被约谈": ("约谈",),
    "企业破产": ("破产", "清算", "重整"),
}


def _type_cue_grounded(event_type: str, source_text: str) -> bool:
    cues = _TYPE_CUES.get(event_type)
    if not cues or not source_text:
        return True  # unknown type / no source text: fail open
    return any(cue in source_text for cue in cues)


def _is_self_loop(arguments: dict[str, str]) -> bool:
    """Two distinct actor roles ("...方") sharing one value — the domain-shift
    mislabel signature (e.g. 收购方 == 被收购方 on an earnings news)."""
    actors = [(role, value) for role, value in arguments.items() if role.endswith("方") and value]
    for i, (role_a, value_a) in enumerate(actors):
        for role_b, value_b in actors[i + 1 :]:
            if role_a != role_b and value_a == value_b:
                return True
    return False


def _n_content_args(arguments: dict[str, str]) -> int:
    """Arguments that carry content (non-date values) — the completeness gate."""
    return sum(1 for value in arguments.values() if value and not _iso_date(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path, help="SARGE canonical pred JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/event_graph_zh/event_nodes.jsonl"),
        help="flat event-node records JSONL",
    )
    parser.add_argument(
        "--source-docs",
        type=Path,
        help="source document JSONL used to locate surface argument evidence",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        help="doc_id -> {date, stock} sidecar JSONL (e.g. the news exporter's "
        "sarge_input.jsonl): publication date and stock code override the "
        "role-cue derivation, anchoring news-dense graphs in time",
    )
    parser.add_argument(
        "--drop-self-loops",
        action="store_true",
        help="drop events where two distinct actor roles share one value "
        "(domain-shift mislabels, e.g. 收购方 == 被收购方)",
    )
    parser.add_argument(
        "--min-args",
        type=int,
        default=0,
        help="drop events with fewer than this many non-date arguments",
    )
    parser.add_argument(
        "--require-type-cue",
        action="store_true",
        help="drop events whose type's surface cues never appear in the source "
        "text (from --meta text or --source-docs); unknown types are kept",
    )
    args = parser.parse_args()

    records: list[dict] = []
    source_docs = _load_source_docs(args.source_docs)
    meta = _load_meta(args.meta)
    n_docs = n_dated = n_evidenced = n_meta = n_objects = 0
    n_self_loops = n_thin = n_uncued = 0
    for line in args.pred.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        doc = json.loads(line)
        doc_id = _record_doc_id(doc)
        n_docs += 1
        for i, event in enumerate(doc.get("events", [])):
            raw_arguments = event.get("arguments", {}) or {}
            arguments = {
                role: _first_text(vals) for role, vals in raw_arguments.items()
            }
            arguments = {r: v for r, v in arguments.items() if v}
            if args.drop_self_loops and _is_self_loop(arguments):
                n_self_loops += 1
                continue
            if args.min_args and _n_content_args(arguments) < args.min_args:
                n_thin += 1
                continue
            if args.require_type_cue:
                source_text = str(
                    meta.get(doc_id, {}).get("text") or source_docs.get(doc_id, "")
                )
                if not _type_cue_grounded(str(event.get("event_type", "")), source_text):
                    n_uncued += 1
                    continue
            argument_evidence = _argument_evidence(
                raw_arguments, doc_id, source_docs.get(doc_id, "")
            )
            meta_record = meta.get(doc_id, {})
            time_anchor = _iso_date(str(meta_record.get("date") or "")) or _derive_time_anchor(
                arguments
            )
            subject = (
                str(meta_record.get("stock") or meta_record.get("stock_name") or "").strip()
                or _derive_subject(arguments)
            )
            obj = _derive_object(arguments, subject)
            n_meta += int(bool(meta_record))
            n_dated += int(time_anchor is not None)
            n_evidenced += int(bool(argument_evidence))
            n_objects += int(obj is not None)
            records.append(
                {
                    "event_id": f"{doc_id}::evt{i}",
                    "event_type": str(event.get("event_type", "Unknown")),
                    "doc_id": doc_id,
                    "trigger": str(event.get("trigger", event.get("trigger_text", ""))).strip(),
                    "trigger_evidence": _trigger_evidence(event, doc_id),
                    "arguments": arguments,
                    "argument_evidence": argument_evidence,
                    "subject": subject,
                    "time_anchor": time_anchor,
                    "confidence": _confidence(event),
                    # Counterparty entity for entity-level quads. Lives in the
                    # annotation bag, not a new EventNode field (schema is frozen).
                    **({"metadata": {"object": obj}} if obj else {}),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    dated_pct = (n_dated / len(records) * 100) if records else 0.0
    print(
        f"wrote {len(records)} event nodes from {n_docs} docs -> {args.output} | "
        f"with time_anchor: {n_dated} ({dated_pct:.0f}%) | "
        f"with argument_evidence: {n_evidenced} | with object: {n_objects} | "
        f"meta-anchored: {n_meta} | "
        f"gated: self_loops={n_self_loops} thin={n_thin} uncued={n_uncued}"
    )
    if records and len({r["confidence"] for r in records}) == 1:
        print(
            f"warning: no confidence signal (all {records[0]['confidence']}); "
            "confidence-thresholding edge admission will degenerate to admit-all"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
