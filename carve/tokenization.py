from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenOffset:
    text: str
    char_start: int
    char_end: int


def tokenize_with_offsets(text: str, *, base_offset: int = 0) -> list[TokenOffset]:
    """A deterministic character tokenizer used by P3 BIO construction."""
    tokens: list[TokenOffset] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        tokens.append(TokenOffset(text=char, char_start=base_offset + index, char_end=base_offset + index + 1))
    return tokens


def token_span_to_char_span(tokens: list[TokenOffset], token_start: int, token_end: int) -> tuple[int, int]:
    if token_start < 0 or token_end > len(tokens) or token_start >= token_end:
        raise ValueError(f"invalid token span: start={token_start} end={token_end} tokens={len(tokens)}")
    return tokens[token_start].char_start, tokens[token_end - 1].char_end
