"""Schema-Aware Constrained Decoding (SACD).

Builds a JSON schema from a SARGE :class:`DatasetSchema` that vLLM's
``guided_decoding`` (or any compatible JSON-schema enforcement backend)
can use to constrain the GETM output during decoding.

The constraint enforces:

1. Root object has an ``events`` array.
2. Each event object has ``event_type`` (constrained to the union of
   schema event types) and ``arguments`` (object).
3. In the *strict* mode each event's ``arguments`` keys are constrained
   to the roles declared for its ``event_type`` (oneOf branch per
   event type, ``additionalProperties: False``).
4. Each argument value is an array of ``{"text": str}`` items, matching
   SARGE's canonical event row layout.

The *lax* mode keeps ``arguments`` as a free object (with the value
shape still constrained) and event_type as a string ``enum``. It is
useful when the grammar engine has trouble compiling the wide ``oneOf``
union (e.g. very large schemas).
"""

from __future__ import annotations

from typing import Any

from sarge.data.schema import DatasetSchema


_ARG_VALUE_SCHEMA: dict[str, Any] = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {"text": {"type": "string", "minLength": 1}},
        "required": ["text"],
    },
}


def build_dataset_json_schema(schema: DatasetSchema, *, strict: bool = True) -> dict[str, Any]:
    """Build a JSON schema for the GETM output of one dataset.

    Args:
        schema: SARGE :class:`DatasetSchema`.
        strict: if True, build a ``oneOf`` of per-event-type schemas with
            ``additionalProperties: False`` so role names are constrained
            to the valid set for each event type. If False, build a single
            event schema with ``event_type`` as an ``enum`` and free
            ``arguments`` keys (still constrains JSON structure and value
            shape).

    Returns:
        A JSON schema dict suitable for vLLM ``GuidedDecodingParams(json=...)``.
    """
    event_types = sorted(schema.event_types)
    if not event_types:
        raise ValueError("DatasetSchema has no event_types — cannot build JSON schema")

    if strict:
        event_options: list[dict[str, Any]] = []
        for event_type in event_types:
            roles = list(schema.event_roles[event_type])
            role_props = {role: _ARG_VALUE_SCHEMA for role in roles}
            event_options.append({
                "type": "object",
                "properties": {
                    "event_type": {"const": event_type},
                    "arguments": {
                        "type": "object",
                        "properties": role_props,
                        "minProperties": 1,
                        "additionalProperties": False,
                    },
                },
                "required": ["event_type", "arguments"],
                "additionalProperties": False,
            })
        events_item: dict[str, Any] = {"oneOf": event_options}
    else:
        events_item = {
            "type": "object",
            "properties": {
                "event_type": {"type": "string", "enum": event_types},
                "arguments": {
                    "type": "object",
                    "minProperties": 1,
                    "additionalProperties": _ARG_VALUE_SCHEMA,
                },
            },
            "required": ["event_type", "arguments"],
        }

    return {
        "type": "object",
        "properties": {"events": {"type": "array", "items": events_item}},
        "required": ["events"],
        "additionalProperties": False,
    }
