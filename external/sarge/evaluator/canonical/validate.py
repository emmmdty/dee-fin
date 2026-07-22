from __future__ import annotations

from dataclasses import dataclass

from evaluator.canonical.schema import EventSchema
from evaluator.canonical.stats import event_type
from evaluator.canonical.types import CanonicalDocument


@dataclass
class ValidationDiagnostics:
    invalid_event_type_count: int = 0
    invalid_role_count: int = 0
    validated_record_count: int = 0
    valid_record_count: int = 0

    def to_report(self) -> dict[str, float | int]:
        rate = self.valid_record_count / self.validated_record_count if self.validated_record_count else 1.0
        return {
            "invalid_event_type_count": self.invalid_event_type_count,
            "invalid_role_count": self.invalid_role_count,
            "validated_record_count": self.validated_record_count,
            "schema_valid_rate": rate,
        }


def validate_documents(documents: list[CanonicalDocument], schema: EventSchema | None) -> ValidationDiagnostics:
    diagnostics = ValidationDiagnostics()
    if schema is None:
        return diagnostics

    for document in documents:
        for record in document.records:
            diagnostics.validated_record_count += 1
            record_valid = True
            normalized_event_type = event_type(record)
            if not schema.has_event_type(normalized_event_type):
                diagnostics.invalid_event_type_count += 1
                record_valid = False
            for role in record.arguments:
                if not schema.has_role(normalized_event_type, str(role)):
                    diagnostics.invalid_role_count += 1
                    record_valid = False
            if record_valid:
                diagnostics.valid_record_count += 1
    return diagnostics
