import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from carve.allocation import (
    CandidateMention,
    build_allocation_targets,
    l_alloc,
    p4_toy_validation_summary,
    p5a_toy_comparison,
    sinkhorn,
)
from evaluator.canonical.types import CanonicalEventRecord
from scripts.carve.run_p4_p5a_toy_validation import main as run_toy_validation


class AllocationTargetTests(unittest.TestCase):
    def test_multi_positive_null_and_oracle_injection_targets(self) -> None:
        records = [
            CanonicalEventRecord(
                document_id="doc1",
                event_type="质押",
                record_id="r2",
                arguments={"质押方": ["甲公司"], "质权方": ["乙银行"]},
            ),
            CanonicalEventRecord(
                document_id="doc1",
                event_type="质押",
                record_id="r1",
                arguments={"质押方": ["甲公司"], "质权方": ["丙银行"]},
            ),
        ]
        candidates = [
            CandidateMention(event_type="质押", role="质押方", value="甲公司", start=3, end=6),
            CandidateMention(event_type="质押", role="质押方", value="无关方", start=9, end=12),
        ]

        batch = build_allocation_targets(
            records=records,
            event_type="质押",
            role="质押方",
            candidates=candidates,
            oracle_inject=True,
        )

        self.assertEqual([record.record_id for record in batch.records], ["r1", "r2"])
        self.assertEqual(batch.target.tolist()[0], [1.0, 1.0, 0.0])
        self.assertEqual(batch.target.tolist()[1], [0.0, 0.0, 1.0])
        self.assertTrue(batch.share_labels[0])
        self.assertFalse(batch.share_labels[1])

        injected = build_allocation_targets(
            records=records,
            event_type="质押",
            role="质权方",
            candidates=[],
            oracle_inject=True,
        )
        self.assertEqual([candidate.value for candidate in injected.candidates], ["丙银行", "乙银行"])
        self.assertTrue(all(candidate.oracle_injected for candidate in injected.candidates))

        inference = build_allocation_targets(
            records=records,
            event_type="质押",
            role="质权方",
            candidates=[],
            oracle_inject=False,
        )
        self.assertEqual(inference.candidates, [])
        self.assertEqual(tuple(inference.target.shape), (0, 3))

    def test_sinkhorn_and_l_alloc_handle_multi_positive_targets(self) -> None:
        logits = torch.tensor([[2.0, 2.0, -2.0], [-1.0, -1.0, 3.0]], dtype=torch.float32)
        probs = sinkhorn(logits, iterations=30)

        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-4))
        self.assertTrue(torch.all(probs > 0))

        target = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        loss_without_coverage = l_alloc(probs, target, positive_coverage_mu=0.0)
        loss_with_coverage = l_alloc(probs, target, positive_coverage_mu=0.5)
        self.assertTrue(math.isfinite(float(loss_without_coverage)))
        self.assertGreaterEqual(float(loss_with_coverage), float(loss_without_coverage))

    def test_p5a_toy_comparison_changes_preference(self) -> None:
        comparison = p5a_toy_comparison()

        self.assertEqual(comparison["baseline_choice"], "wrong-record")
        self.assertEqual(comparison["allocation_aware_choice"], "correct-record")
        self.assertGreater(comparison["allocation_margin"], 0)

    def test_p4_toy_validation_summary_covers_acceptance_checks(self) -> None:
        summary = p4_toy_validation_summary()

        self.assertEqual(summary["status"], "toy_behavior_only")
        self.assertEqual(summary["record_order"], ["r1", "r2"])
        self.assertEqual(
            summary["acceptance_checks"],
            {
                "repeated_surface_shared_by_two_records": True,
                "non_shared_value_single_positive_column": True,
                "unmatched_candidate_assigned_to_null": True,
                "deterministic_repeated_execution": True,
                "oracle_injected_candidate_train_only": True,
                "mu_zero_disables_positive_coverage": True,
                "share_gate_surface_only": True,
            },
        )
        self.assertEqual(summary["target_cases"]["shared_surface"]["target_row"], [1.0, 1.0, 0.0])
        self.assertEqual(summary["target_cases"]["single_positive"]["target_row"], [1.0, 0.0, 0.0])
        self.assertEqual(summary["target_cases"]["null_candidate"]["target_row"], [0.0, 0.0, 1.0])
        self.assertTrue(summary["target_cases"]["shared_surface"]["share_label"])
        self.assertFalse(summary["target_cases"]["single_positive"]["share_label"])
        self.assertEqual(summary["oracle_injection"]["inference_candidates"], [])
        self.assertEqual(summary["oracle_injection"]["training_oracle_values"], ["丙银行", "乙银行"])
        self.assertGreater(summary["loss"]["with_positive_coverage"], summary["loss"]["mu_zero"])
        self.assertTrue(summary["sinkhorn"]["row_sums_close_to_one"])

    def test_toy_validation_script_writes_p4_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "summary.json"

            exit_code = run_toy_validation(["--out", str(out)])

            self.assertEqual(exit_code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(payload["p4"]["accepted"])
            self.assertEqual(payload["p4"]["status"], "toy_behavior_only")


if __name__ == "__main__":
    unittest.main()
