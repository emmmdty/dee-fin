from __future__ import annotations

from typing import Protocol

from carve.allocation import CandidateMention
from carve.datasets import CandidateLexicon, DueeDocument, generate_inference_candidates as generate_lexicon_candidates
from evaluator.canonical.normalize import normalize_optional_text, normalize_text
from evaluator.canonical.types import CanonicalEventRecord


class MentionExtractor(Protocol):
    def extract(self, document: DueeDocument, event_type: str, role: str) -> list[CandidateMention]:
        ...


def generate_candidates(
    document: DueeDocument,
    evidence_logits: object | None,
    mention_extractor: MentionExtractor | None,
    event_type: str,
    role: str,
    *,
    lexicon: CandidateLexicon | None = None,
    oracle_inject: bool = False,
    records: list[CanonicalEventRecord] | None = None,
) -> list[CandidateMention]:
    normalized_event_type = normalize_text(event_type)
    normalized_role = normalize_text(role)
    candidates: dict[tuple[str, int, int, str], CandidateMention] = {}
    if mention_extractor is not None:
        for candidate in mention_extractor.extract(document, normalized_event_type, normalized_role):
            normalized_value = normalize_optional_text(candidate.value)
            if not normalized_value:
                continue
            normalized = CandidateMention(
                event_type=normalized_event_type,
                role=normalized_role,
                value=normalized_value,
                start=candidate.start,
                end=candidate.end,
                source=candidate.source,
                oracle_injected=candidate.oracle_injected,
                raw_span=candidate.raw_span if candidate.raw_span is not None else candidate.value,
            )
            candidates[(normalized.value, normalized.start, normalized.end, normalized.source)] = normalized
    if lexicon is not None:
        for candidate in generate_lexicon_candidates(
            document,
            lexicon,
            event_type=normalized_event_type,
            role=normalized_role,
        ):
            candidates.setdefault((candidate.value, candidate.start, candidate.end, candidate.source), candidate)
    if oracle_inject and records is not None:
        for record in records:
            if normalize_text(record.event_type) != normalized_event_type:
                continue
            for value in record.arguments.get(normalized_role, []):
                normalized_value = normalize_optional_text(value)
                if normalized_value:
                    candidates.setdefault(
                        (normalized_value, -1, -1, "oracle_gold"),
                        CandidateMention(
                            event_type=normalized_event_type,
                            role=normalized_role,
                            value=normalized_value,
                            source="oracle_gold",
                            oracle_injected=True,
                            raw_span=str(value),
                        ),
                    )
    return sorted(candidates.values(), key=lambda candidate: (candidate.start, candidate.end, candidate.value, candidate.source))
