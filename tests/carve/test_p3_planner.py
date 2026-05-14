import unittest

import torch

from carve.p3_planner import (
    CountPlanner,
    RecordPlanner,
    planner_loss,
    presence_loss,
    truncated_poisson_argmax,
    truncated_poisson_nll,
)
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
        self.assertEqual(RecordPlanner.gold_presence(records, "质押"), 1)
        self.assertEqual(RecordPlanner.gold_presence(records, "增持"), 0)

    def test_type_gate_pos_weight_balanced_loss(self) -> None:
        logits = torch.zeros((3,), dtype=torch.float32)
        targets = torch.tensor([1.0, 0.0, 0.0])

        weighted = presence_loss(logits, targets, pos_weight=torch.tensor(2.0))
        unweighted = presence_loss(logits, targets, pos_weight=torch.tensor(1.0))

        expected = torch.tensor((2.0 + 1.0 + 1.0) * 0.6931471805599453 / 3.0)
        self.assertTrue(torch.allclose(weighted, expected, atol=1e-6))
        self.assertGreater(float(weighted), float(unweighted))

    def test_truncated_poisson_nll_numerical_stable(self) -> None:
        log_lambda = torch.tensor([-50.0, 0.0, 4.0], dtype=torch.float32)
        targets = torch.tensor([1.0, 2.0, 16.0], dtype=torch.float32)

        loss = truncated_poisson_nll(log_lambda, targets)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_truncated_poisson_argmax_inference(self) -> None:
        low_rate = truncated_poisson_argmax(torch.tensor([torch.log(torch.tensor(0.2))]), k_clip=16)
        mid_rate = truncated_poisson_argmax(torch.tensor([torch.log(torch.tensor(3.2))]), k_clip=16)
        high_rate = truncated_poisson_argmax(torch.tensor([torch.log(torch.tensor(20.0))]), k_clip=16)

        self.assertEqual(int(low_rate.item()), 1)
        self.assertEqual(int(mid_rate.item()), 3)
        self.assertEqual(int(high_rate.item()), 16)

    def test_planner_truncation_caps_at_k_max(self) -> None:
        logits = torch.zeros((2, 3), dtype=torch.float32)
        targets = torch.tensor([1, 4])

        loss, skipped = planner_loss(logits, targets, k_max=2)

        self.assertEqual(skipped, 1)
        self.assertTrue(torch.isfinite(loss))

    def test_predict_n_t_uses_argmax(self) -> None:
        planner = RecordPlanner(hidden_size=2, num_event_types=2, k_max=16)
        with torch.no_grad():
            planner.type_gate.proj.weight.zero_()
            planner.type_gate.proj.bias.fill_(10.0)
            planner.count_planner.proj.weight.zero_()
            planner.count_planner.proj.bias.fill_(torch.log(torch.tensor(3.2)))

        predicted = planner.predict_n_t(torch.zeros((1, 2)), torch.tensor([0]))

        self.assertEqual(predicted, 3)

    def test_count_planner_predict_count_uses_truncated_support(self) -> None:
        planner = CountPlanner(hidden_size=2, num_event_types=2)
        with torch.no_grad():
            planner.proj.weight.zero_()
            planner.proj.bias.fill_(torch.log(torch.tensor(20.0)))

        predicted = planner.predict_count(torch.zeros((1, 2)), torch.tensor([0]), k_clip=16)

        self.assertEqual(predicted, 16)


if __name__ == "__main__":
    unittest.main()
