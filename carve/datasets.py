from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from carve.allocation import CandidateMention
from evaluator.canonical.loaders import adapt_document
from evaluator.canonical.normalize import normalize_optional_text, normalize_text
from evaluator.canonical.types import CanonicalEventRecord


@dataclass(frozen=True)
class DueeDocument:
    document_id: str
    text: str
    title: str
    records: list[CanonicalEventRecord]


CandidateLexicon = dict[str, dict[str, dict[str, int]]]


def load_duee_documents(path: str | Path, *, dataset: str = "DuEE-Fin-dev500") -> list[DueeDocument]:
    documents = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            canonical = adapt_document(row, dataset=dataset, index=index)
            text = str(row.get("text") or row.get("content") or "")
            title = str(row.get("title") or "")
            documents.append(
                DueeDocument(
                    document_id=canonical.document_id,
                    text=text,
                    title=title,
                    records=canonical.records,
                )
            )
    return documents


def multi_event_subset(documents: Iterable[DueeDocument]) -> list[DueeDocument]:
    selected = [document for document in documents if len(document.records) >= 2]
    return selected or [document for document in documents if document.records]


def build_candidate_lexicon(documents: Iterable[DueeDocument], *, min_count: int = 1) -> CandidateLexicon:
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for document in documents:
        for record in document.records:
            event_type = normalize_text(record.event_type)
            for role, values in record.arguments.items():
                normalized_role = normalize_text(role)
                for value in values:
                    normalized_value = normalize_optional_text(value)
                    if normalized_value:
                        counts[event_type][normalized_role][normalized_value] += 1
    return {
        event_type: {
            role: {value: count for value, count in sorted(values.items()) if count >= min_count}
            for role, values in roles.items()
        }
        for event_type, roles in counts.items()
    }


def generate_inference_candidates(
    document: DueeDocument,
    lexicon: CandidateLexicon,
    *,
    event_type: str,
    role: str,
) -> list[CandidateMention]:
    normalized_event_type = normalize_text(event_type)
    normalized_role = normalize_text(role)
    haystack = f"{document.title}\n{document.text}"
    candidates: dict[tuple[str, int, int], CandidateMention] = {}
    for value in lexicon.get(normalized_event_type, {}).get(normalized_role, {}):
        for match in re.finditer(re.escape(value), haystack):
            candidates[(value, match.start(), match.end())] = CandidateMention(
                event_type=normalized_event_type,
                role=normalized_role,
                value=value,
                start=match.start(),
                end=match.end(),
                source="train_lexicon_text_match",
            )
    for value, start, end in _regex_candidates(haystack, normalized_role):
        candidates.setdefault(
            (value, start, end),
            CandidateMention(
                event_type=normalized_event_type,
                role=normalized_role,
                value=value,
                start=start,
                end=end,
                source="role_regex_text_match",
            ),
        )
    return sorted(candidates.values(), key=lambda candidate: (candidate.start, candidate.end, candidate.value))


def write_canonical_jsonl(path: str | Path, documents: Iterable[DueeDocument | dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            if isinstance(document, DueeDocument):
                row = {
                    "document_id": document.document_id,
                    "events": [
                        {
                            "event_type": record.event_type,
                            "record_id": record.record_id,
                            "arguments": record.arguments,
                        }
                        for record in document.records
                    ],
                }
            else:
                row = document
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _regex_candidates(text: str, role: str) -> list[tuple[str, int, int]]:
    patterns = []
    if any(token in role for token in ("数量", "金额", "价格", "比例", "股比", "净亏损", "债务")):
        patterns.append(r"\d+(?:\.\d+)?(?:万|亿)?(?:元|股|万股|亿股|%|％)?")
    if any(token in role for token in ("时间", "日期", "完成")):
        patterns.append(r"\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日|截至[^，。；\s]{1,12}")
    results: list[tuple[str, int, int]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = normalize_optional_text(match.group(0))
            if value:
                results.append((value, match.start(), match.end()))
    return results
