import unittest

import torch

from carve.p3_planner import RecordPlanner, planner_loss
from evaluator.canonical.types import CanonicalEventRecord


class P3PlannerTests(unittest.TestCase):
    def test_planner_gold_n_t_matches_record_count(self) -> None:
        records = [
            CanonicalEventRecord(document_id="doc1", event_type="质押", record_id="r1"),
            CanonicalEventRecord(document_id="doc1", event_type="质押", record_id="r2"),
            CanonicalEventRecord(document_id="doc1", event_type="收购", record_id="r3"),
        ]

        self.assertEqual(RecordPlanner.gold_n_t(records, "质押"), 2)
        self.assertEqual(RecordPlanner.gold_n_t(records, "收购"), 1)
        self.assertEqual(RecordPlanner.gold_n_t(records, "增持"), 0)

    def test_planner_truncation_caps_at_k_max(self) -> None:
        logits = torch.zeros((2, 3), dtype=torch.float32)
        targets = torch.tensor([1, 4])

        loss, skipped = planner_loss(logits, targets, k_max=2)

        self.assertEqual(skipped, 1)
        self.assertTrue(torch.isfinite(loss))

    def test_predict_n_t_uses_argmax(self) -> None:
        planner = RecordPlanner(hidden_size=2, num_event_types=2, k_max=3)
        with torch.no_grad():
            planner.proj.weight.zero_()
            planner.proj.bias.copy_(torch.tensor([0.0, 1.0, 3.0, 2.0]))

        predicted = planner.predict_n_t(torch.zeros((1, 2)), torch.tensor([0]))

        self.assertEqual(predicted, 2)


if __name__ == "__main__":
    unittest.main()
