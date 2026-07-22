from __future__ import annotations

import json
from pathlib import Path

from sarge.evaluation.evaluator_adapter import convert_sarge_predictions_to_evaluator


def test_convert_sarge_predictions_to_evaluator_value_lists(tmp_path: Path) -> None:
    source = tmp_path / "pred.jsonl"
    target = tmp_path / "eval.jsonl"
    source.write_text(
        json.dumps(
            {
                "doc_id": "doc-1",
                "events": [
                    {
                        "event_type": "质押",
                        "arguments": {
                            "质押方": [{"text": "甲公司"}, {"text": ""}],
                            "质押物": [{"text": "A股"}],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = convert_sarge_predictions_to_evaluator(source, target)

    assert report.document_count == 1
    assert report.event_count == 1
    assert report.value_count == 2
    assert report.skipped_value_count == 1
    row = json.loads(target.read_text(encoding="utf-8"))
    assert row == {
        "document_id": "doc-1",
        "predictions": [
            {
                "event_type": "质押",
                "arguments": {
                    "质押方": ["甲公司"],
                    "质押物": ["A股"],
                },
            }
        ],
    }
