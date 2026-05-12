from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProcNetTokenSlotDecoder:
    tokenizer: Any
    strip_decode_spaces: bool = True
    total_slot_count: int = 0
    token_id_slot_count: int = 0
    decoded_slot_count: int = 0
    passthrough_slot_count: int = 0
    decode_failure_count: int = 0
    decode_failures: list[str] = field(default_factory=list)

    def __call__(self, value: Any) -> Any:
        self.total_slot_count += 1
        token_ids = coerce_token_id_sequence(value)
        if token_ids is None:
            self.passthrough_slot_count += 1
            return value

        self.token_id_slot_count += 1
        try:
            decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        except Exception as exc:  # pragma: no cover - tokenizer-specific failure details
            self.decode_failure_count += 1
            if len(self.decode_failures) < 10:
                self.decode_failures.append(f"{value!r}: {exc}")
            return value

        if self.strip_decode_spaces:
            decoded = decoded.replace(" ", "")
        self.decoded_slot_count += 1
        return decoded

    def report(self) -> dict[str, Any]:
        return {
            "strip_decode_spaces": self.strip_decode_spaces,
            "total_slot_count": self.total_slot_count,
            "token_id_slot_count": self.token_id_slot_count,
            "decoded_slot_count": self.decoded_slot_count,
            "passthrough_slot_count": self.passthrough_slot_count,
            "decode_failure_count": self.decode_failure_count,
            "decode_failures": self.decode_failures,
        }


def coerce_token_id_sequence(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)):
        return _coerce_sequence(value)
    if not isinstance(value, str):
        return None

    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, (list, tuple)):
        return None
    return _coerce_sequence(parsed)


def _coerce_sequence(value: list[Any] | tuple[Any, ...]) -> list[int] | None:
    if not value:
        return None
    if not all(isinstance(item, int) for item in value):
        return None
    return list(value)


def load_procnet_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
