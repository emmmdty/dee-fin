from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from re import Pattern

from sarge.surface_memory.types import SurfaceCandidate, SurfaceCandidateDict, SurfaceMemory, SurfaceMemoryDict
from sarge.data.loader import V2DocumentInput

DEFAULT_CONTEXT_WINDOW = 36
DEFAULT_CHUNK_SIZE = 512


@dataclass(frozen=True)
class SurfaceRule:
    name: str
    pattern: Pattern[str]
    group: str | int = 0


SURFACE_RULES: tuple[SurfaceRule, ...] = (
    SurfaceRule(
        "company_fragment",
        re.compile(
            r"(?<![\u4e00-\u9fffA-Za-z0-9])"
            r"[\u4e00-\u9fffA-Za-z0-9（）()·&.\-]{2,45}?"
            r"(?:股份有限公司|有限责任公司|集团有限公司|控股有限公司|有限公司|证券股份有限公司|银行股份有限公司|公司|法院|交易所)"
        ),
    ),
    SurfaceRule(
        "money",
        re.compile(
            r"(?:不低于|不超过|不少于|不多于|约|合计|共计|支付|作价|人民币)?"
            r"\s*[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:万|亿)?\s*(?:人民币|亿元|万元|元|美元|万美元)"
        ),
    ),
    SurfaceRule("percentage", re.compile(r"[0-9]+(?:\.[0-9]+)?\s*[%％]")),
    SurfaceRule(
        "arabic_date",
        re.compile(r"[0-9]{4}\s*年\s*[0-9]{1,2}\s*月\s*[0-9]{1,2}\s*日|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}"),
    ),
    SurfaceRule(
        "chinese_date",
        re.compile(r"[〇零一二三四五六七八九十]{4}\s*年\s*[〇零一二三四五六七八九十]{1,3}\s*月\s*[〇零一二三四五六七八九十]{1,3}\s*日"),
    ),
    SurfaceRule(
        "share_quantity",
        re.compile(r"[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:万|亿)?\s*(?:股|股份|股权)"),
    ),
    SurfaceRule(
        "person_after_title",
        re.compile(r"(?:董事长|监事长|法定代表人|高管|董事会秘书|董事|监事|总经理|经理|秘书)(?P<surface>[\u4e00-\u9fff]{2,4})"),
        "surface",
    ),
    SurfaceRule(
        "person_with_honorific",
        re.compile(r"(?P<surface>[\u4e00-\u9fff]{2,4})(?:先生|女士)"),
        "surface",
    ),
    SurfaceRule(
        "quoted_entity",
        re.compile(r"[“\"《](?P<surface>[^”\"》]{2,40})[”\"》]"),
        "surface",
    ),
    SurfaceRule(
        "stock_code",
        re.compile(r"(?:证券代码[:：]?\s*)?(?:SH|SZ)?[0-9]{6}"),
    ),
)


def build_surface_memory(
    document: V2DocumentInput,
    *,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> SurfaceMemory:
    candidates = list(
        iter_surface_candidates(
            document,
            context_window=context_window,
            chunk_size=chunk_size,
        )
    )
    return SurfaceMemory(doc_id=document.doc_id, candidates=candidates)


def iter_surface_candidates(
    document: V2DocumentInput,
    *,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterable[SurfaceCandidate]:
    merged: OrderedDict[tuple[str, int, int, str], dict[str, object]] = OrderedDict()
    for text_source, text in _document_text_sources(document):
        for rule in SURFACE_RULES:
            for surface, start, end in _rule_matches(rule, text):
                normalized_surface = _clean_surface(surface)
                if not normalized_surface:
                    continue
                key = (text_source, start, end, normalized_surface)
                if key not in merged:
                    merged[key] = {
                        "text_source": text_source,
                        "surface": normalized_surface,
                        "context": _context(text, start, end, context_window),
                        "chunk_id": f"chunk_{start // chunk_size:04d}",
                        "char_start": start,
                        "char_end": end,
                        "rule_names": [],
                    }
                rule_names = merged[key]["rule_names"]
                assert isinstance(rule_names, list)
                if rule.name not in rule_names:
                    rule_names.append(rule.name)

    for row in merged.values():
        rule_names = tuple(str(name) for name in row["rule_names"])
        surface = str(row["surface"])
        text_source = str(row["text_source"])
        char_start = int(row["char_start"])
        char_end = int(row["char_end"])
        yield SurfaceCandidate(
            candidate_id=_candidate_id(document.doc_id, text_source, char_start, char_end, surface, rule_names),
            doc_id=document.doc_id,
            surface=surface,
            context=str(row["context"]),
            chunk_id=str(row["chunk_id"]),
            source="rule",
            char_start=char_start,
            char_end=char_end,
            metadata={"rule_names": list(rule_names), "text_source": text_source},
        )


def surface_candidate_to_dict(candidate: SurfaceCandidate) -> SurfaceCandidateDict:
    payload: SurfaceCandidateDict = {
        "candidate_id": candidate.candidate_id,
        "doc_id": candidate.doc_id,
        "surface": candidate.surface,
        "context": candidate.context,
        "chunk_id": candidate.chunk_id,
        "source": candidate.source,
    }
    if candidate.char_start is not None:
        payload["char_start"] = candidate.char_start
    if candidate.char_end is not None:
        payload["char_end"] = candidate.char_end
    if candidate.role_score is not None:
        payload["role_score"] = candidate.role_score
    if candidate.metadata:
        payload["metadata"] = dict(candidate.metadata)
    return payload


def surface_memory_to_dict(memory: SurfaceMemory) -> SurfaceMemoryDict:
    return {
        "doc_id": memory.doc_id,
        "source": memory.source,
        "candidates": [surface_candidate_to_dict(candidate) for candidate in memory.candidates],
    }


def _rule_matches(rule: SurfaceRule, text: str) -> Iterable[tuple[str, int, int]]:
    for match in rule.pattern.finditer(text):
        if isinstance(rule.group, str):
            surface = match.group(rule.group)
            start, end = match.span(rule.group)
        else:
            surface = match.group(rule.group)
            start, end = match.span(rule.group)
        yield surface, start, end


def _candidate_id(
    doc_id: str,
    text_source: str,
    char_start: int,
    char_end: int,
    surface: str,
    rule_names: tuple[str, ...],
) -> str:
    raw_key = "\t".join((doc_id, text_source, str(char_start), str(char_end), surface, "|".join(rule_names)))
    digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}:csg:{digest}"


def _document_text_sources(document: V2DocumentInput) -> Iterable[tuple[str, str]]:
    seen: set[str] = set()
    for source_name, value in (("content", document.content), ("content_raw", document.content_raw)):
        if value is None:
            continue
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        yield source_name, text


def _context(text: str, start: int, end: int, window: int) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def _clean_surface(surface: str) -> str:
    text = re.sub(r"\s+", "", surface).strip()
    return text.strip("，,。；;：:（）()[]【】")
