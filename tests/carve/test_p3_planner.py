import unittest

import torch

from carve.p3_planner import (
    CountPlanner,
    RecordPlanner,
    SentenceLevelCountPlanner,
    TypeGate,
    _planner_features,
    planner_loss,
    presence_loss,
    sentence_count_loss,
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

    def test_truncated_poisson_nll_respects_sample_weights(self) -> None:
        log_lambda = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        targets = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)

        unweighted = truncated_poisson_nll(log_lambda, targets)
        equal_weights = truncated_poisson_nll(
            log_lambda, targets, sample_weights=torch.ones_like(targets)
        )
        # Equal weights must yield the same loss as the unweighted mean.
        self.assertTrue(torch.allclose(unweighted, equal_weights, atol=1e-6))

        # Up-weighting the high-count sample should change the loss.
        skewed_weights = torch.tensor([0.1, 1.0, 5.0], dtype=torch.float32)
        skewed = truncated_poisson_nll(log_lambda, targets, sample_weights=skewed_weights)
        self.assertFalse(torch.allclose(unweighted, skewed, atol=1e-3))

    def test_count_planner_bias_zero_initialized(self) -> None:
        planner = CountPlanner(hidden_size=4, num_event_types=2)
        self.assertTrue(torch.allclose(planner.proj.bias, torch.zeros_like(planner.proj.bias)))

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

    def test_type_gate_uses_evidence_vec_when_sentence_repr_varies(self) -> None:
        torch.manual_seed(0)
        gate = TypeGate(hidden_size=4, num_event_types=2)
        global_repr = torch.zeros((1, 4))
        type_id = torch.tensor([0])
        sentence_a = torch.zeros((1, 3, 4))
        sentence_a[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        sentence_b = torch.zeros((1, 3, 4))
        sentence_b[0, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        mask = torch.ones((1, 3), dtype=torch.bool)

        logit_a = float(gate(global_repr, type_id, sentence_repr=sentence_a, sentence_mask=mask).item())
        logit_b = float(gate(global_repr, type_id, sentence_repr=sentence_b, sentence_mask=mask).item())

        self.assertNotAlmostEqual(logit_a, logit_b, places=5)

    def test_type_gate_lexical_hit_changes_logit(self) -> None:
        gate = TypeGate(hidden_size=2, num_event_types=1)
        with torch.no_grad():
            gate.proj.weight.zero_()
            gate.proj.weight[0, -1] = 5.0
            gate.proj.bias.zero_()
        global_repr = torch.zeros((1, 2))
        type_id = torch.tensor([0])

        logit_with_hit = float(gate(global_repr, type_id, lexical_hit=torch.tensor([1.0])).item())
        logit_without_hit = float(gate(global_repr, type_id, lexical_hit=torch.tensor([0.0])).item())

        self.assertAlmostEqual(logit_with_hit - logit_without_hit, 5.0, places=5)

    def test_planner_features_global_only_backward_compat(self) -> None:
        embedding = torch.nn.Embedding(2, 3)
        torch.nn.init.constant_(embedding.weight, 0.0)
        global_repr = torch.tensor([[1.0, 2.0, 3.0]])
        type_id = torch.tensor([0])

        features = _planner_features(global_repr, type_id, embedding)

        self.assertEqual(features.shape, (1, 3 * 3 + 1))
        self.assertTrue(torch.allclose(features[0, :3], torch.tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.allclose(features[0, 3:], torch.zeros((7,))))

    def test_padding_mask_excludes_zero_sentences(self) -> None:
        embedding = torch.nn.Embedding(1, 2)
        torch.nn.init.constant_(embedding.weight, 1.0)
        global_repr = torch.zeros((1, 2))
        type_id = torch.tensor([0])
        sentence_repr = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
        full_mask = torch.tensor([[True, True, True]])
        empty_mask = torch.tensor([[False, False, False]])

        full_features = _planner_features(global_repr, type_id, embedding, sentence_repr=sentence_repr, sentence_mask=full_mask)
        empty_features = _planner_features(global_repr, type_id, embedding, sentence_repr=sentence_repr, sentence_mask=empty_mask)

        evidence_slice = slice(4, 6)
        self.assertTrue(torch.allclose(empty_features[0, evidence_slice], torch.zeros((2,))))
        self.assertFalse(torch.allclose(full_features[0, evidence_slice], torch.zeros((2,))))


    def test_sentence_level_count_planner_forward_shapes(self) -> None:
        planner = SentenceLevelCountPlanner(hidden_size=4, num_event_types=2)
        sentence_repr = torch.randn(3, 5, 4)  # [B=3, S=5, H=4]
        type_id = torch.tensor([0, 1, 0])

        logits = planner.forward(type_id, sentence_repr)

        self.assertEqual(logits.shape, (3, 5))  # [B, S]

    def test_sentence_count_loss_masking(self) -> None:
        logits = torch.randn(3, 5)
        labels = torch.ones(3, 5)
        mask = torch.tensor([
            [True, True, False, False, False],
            [True, False, False, False, False],
            [False, False, False, False, False],
        ])
        loss = float(sentence_count_loss(logits, labels, mask, pos_weight=1.0).item())

        self.assertGreater(loss, 0.0)
        # Masked positions should not contribute: if only mask=True positions are
        # computed, loss should be identical to a version where masked positions are random.
        labels_offset = labels.clone()
        labels_offset[0, 2:] = 99.0
        labels_offset[1, 1:] = 99.0
        labels_offset[2, :] = 99.0
        loss_offset = float(sentence_count_loss(logits, labels_offset, mask, pos_weight=1.0).item())
        self.assertAlmostEqual(loss, loss_offset, places=4)

    def test_sentence_aggregated_count_matches_label_when_perfect(self) -> None:
        planner = SentenceLevelCountPlanner(hidden_size=4, num_event_types=2)
        # freeze the scorer so it outputs very confident predictions
        with torch.no_grad():
            planner.scorer[0].weight.zero_()
            planner.scorer[0].bias.zero_()
            planner.scorer[2].weight.zero_()
            # Set bias so logit ≈ +10 for all positions → sigmoid ≈ 1.0
            planner.scorer[2].bias.fill_(10.0)
            planner.type_embedding.weight.zero_()

        sentence_repr = torch.randn(2, 4, 4)
        mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        type_id = torch.tensor([0, 0])

        count = planner.expected_count(type_id, sentence_repr, mask)

        # With bias=10, sigmoid ≈ 1.0, so expected sum = masked sentence count
        self.assertEqual(int(count[0].item()), 3)  # 3 valid sentences
        self.assertEqual(int(count[1].item()), 2)  # 2 valid sentences

    def test_record_planner_sentence_mode_integration(self) -> None:
        planner = RecordPlanner(
            hidden_size=4, num_event_types=2, k_max=5, count_head_mode="sentence",
        )
        self.assertEqual(planner.count_head_mode, "sentence")
        self.assertIsInstance(planner.count_planner, SentenceLevelCountPlanner)

        # In sentence mode, count_log_lambda should raise
        with self.assertRaises(RuntimeError):
            planner.count_log_lambda(torch.zeros((1, 4)), torch.tensor([0]))

        # sentence_count_logits should work
        sent_repr = torch.randn(2, 3, 4)
        logits = planner.sentence_count_logits(torch.tensor([0, 1]), sent_repr)
        self.assertEqual(logits.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
