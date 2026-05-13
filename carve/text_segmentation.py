from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sentence:
    text: str
    char_start: int
    char_end: int


_CLOSING_QUOTES = {"”", "’", '"', "'", "》", "」", "』"}
_SENTENCE_END = {"。", "！", "？", "!", "?"}


def split_sentences(text: str) -> list[Sentence]:
    """Deterministic newline-first Chinese sentence segmentation."""
    sentences: list[Sentence] = []
    offset = 0
    for paragraph in text.splitlines(keepends=True):
        content = paragraph.rstrip("\r\n")
        newline_len = len(paragraph) - len(content)
        _append_paragraph_sentences(content, offset, sentences)
        offset += len(content) + newline_len
    if not text:
        return []
    if text and text[-1] in "\r\n":
        return sentences
    return sentences


def _append_paragraph_sentences(paragraph: str, base_offset: int, sentences: list[Sentence]) -> None:
    if not paragraph.strip():
        return
    start = 0
    index = 0
    while index < len(paragraph):
        char = paragraph[index]
        if char in _SENTENCE_END:
            end = index + 1
            while end < len(paragraph) and paragraph[end] in _CLOSING_QUOTES:
                end += 1
            _append_sentence(paragraph, base_offset, start, end, sentences)
            start = end
            index = end
            continue
        index += 1
    if start < len(paragraph):
        _append_sentence(paragraph, base_offset, start, len(paragraph), sentences)


def _append_sentence(paragraph: str, base_offset: int, start: int, end: int, sentences: list[Sentence]) -> None:
    while start < end and paragraph[start].isspace():
        start += 1
    while end > start and paragraph[end - 1].isspace():
        end -= 1
    if start < end:
        sentences.append(Sentence(text=paragraph[start:end], char_start=base_offset + start, char_end=base_offset + end))
