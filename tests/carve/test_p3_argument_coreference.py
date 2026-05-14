import unittest

import torch

from carve.p3_planner import (
    ArgumentCoreferenceHead,
    RecordPlanner,
    coref_pair_loss,
    predict_clusters,
)


class ArgumentCoreferenceHeadTests(unittest.TestCase):
    def test_forward_shape_and_symmetry(self) -> None:
        torch.manual_seed(0)
        head = ArgumentCoreferenceHead(hidden_size=4, num_event_types=2, num_roles=3)
        span_repr = torch.randn(2, 5, 4)
        span_role_id = torch.tensor([
            [1, 2, 3, 0, 0],
            [1, 1, 2, 0, 0],
        ])
        sent_pos = torch.tensor([
            [1, 1, 2, 0, 0],
            [1, 2, 3, 0, 0],
        ])
        type_id = torch.tensor([0, 1])
        span_mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, True, False, False],
        ])

        affinity = head(span_repr, span_role_id, sent_pos, type_id, span_mask=span_mask)

        self.assertEqual(affinity.shape, (2, 5, 5))
        # symmetry on the valid sub-matrix
        for b in range(2):
            for i in range(3):
                for j in range(3):
                    self.assertAlmostEqual(
                        float(affinity[b, i, j].item()),
                        float(affinity[b, j, i].item()),
                        places=4,
                    )

    def test_padding_positions_are_strongly_negative(self) -> None:
        head = ArgumentCoreferenceHead(hidden_size=4, num_event_types=2, num_roles=3)
        span_repr = torch.randn(1, 4, 4)
        span_role_id = torch.tensor([[1, 2, 0, 0]])
        sent_pos = torch.tensor([[1, 1, 0, 0]])
        type_id = torch.tensor([0])
        span_mask = torch.tensor([[True, True, False, False]])

        affinity = head(span_repr, span_role_id, sent_pos, type_id, span_mask=span_mask)

        # the padded (i,j) entries must be -inf-equivalent
        self.assertLess(float(affinity[0, 0, 2].item()), -1e8)
        self.assertLess(float(affinity[0, 2, 3].item()), -1e8)
        # the diagonal is also masked
        self.assertLess(float(affinity[0, 0, 0].item()), -1e8)
        # but valid (0,1) is finite
        self.assertGreater(float(affinity[0, 0, 1].item()), -1e8)

    def test_same_role_bias_changes_affinity(self) -> None:
        head = ArgumentCoreferenceHead(hidden_size=2, num_event_types=1, num_roles=2)
        with torch.no_grad():
            head.span_proj.weight.zero_()
            head.role_embedding.weight.zero_()
            head.sent_pos_embedding.weight.zero_()
            head.type_embedding.weight.zero_()
            head.feature_proj.weight.zero_()
            head.feature_proj.bias.zero_()
            head.same_role_bias.fill_(3.0)
            head.same_sentence_bias.zero_()
        span_repr = torch.zeros(1, 3, 2)
        # mentions 0 and 1 share role 1; mention 2 has role 2
        span_role_id = torch.tensor([[1, 1, 2]])
        sent_pos = torch.tensor([[1, 1, 1]])
        type_id = torch.tensor([0])
        mask = torch.tensor([[True, True, True]])

        affinity = head(span_repr, span_role_id, sent_pos, type_id, span_mask=mask)

        # (0,1) gets same_role_bias=3.0; (0,2) does not
        self.assertAlmostEqual(float(affinity[0, 0, 1].item()), 3.0, places=5)
        self.assertAlmostEqual(float(affinity[0, 0, 2].item()), 0.0, places=5)

    def test_backward_does_not_nan(self) -> None:
        torch.manual_seed(0)
        head = ArgumentCoreferenceHead(hidden_size=4, num_event_types=2, num_roles=3)
        span_repr = torch.randn(2, 4, 4, requires_grad=True)
        span_role_id = torch.tensor([
            [1, 2, 3, 0],
            [1, 1, 2, 0],
        ])
        sent_pos = torch.tensor([
            [1, 1, 2, 0],
            [1, 2, 3, 0],
        ])
        type_id = torch.tensor([0, 1])
        mask = torch.tensor([
            [True, True, True, False],
            [True, True, True, False],
        ])
        labels = torch.zeros(2, 4, 4)
        labels[0, 0, 1] = labels[0, 1, 0] = 1.0
        labels[1, 0, 2] = labels[1, 2, 0] = 1.0
        eligible = torch.zeros(2, 4, 4, dtype=torch.bool)
        eligible[0, :3, :3] = True
        eligible[1, :3, :3] = True

        affinity = head(span_repr, span_role_id, sent_pos, type_id, span_mask=mask)
        loss = coref_pair_loss(affinity, labels, eligible, pos_weight=1.0)
        loss.backward()

        for p in head.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all().item())
        self.assertTrue(torch.isfinite(span_repr.grad).all().item())


class PredictClustersTests(unittest.TestCase):
    def test_threshold_partitions_into_two_components(self) -> None:
        # mentions {0,1} strongly connected; {2,3} strongly connected; cross pairs weak
        m_max = 4
        affinity = torch.full((m_max, m_max), -5.0)
        affinity[0, 1] = affinity[1, 0] = 5.0
        affinity[2, 3] = affinity[3, 2] = 5.0
        mask = torch.tensor([True, True, True, True])

        clusters = predict_clusters(affinity, mask, threshold=0.5)

        self.assertEqual(len(clusters), 2)
        cluster_sets = sorted([tuple(sorted(c)) for c in clusters])
        self.assertEqual(cluster_sets, [(0, 1), (2, 3)])

    def test_transitive_closure_merges_chain(self) -> None:
        # 0--1, 1--2, but 0..2 weak -> connected components still merge them
        m_max = 4
        affinity = torch.full((m_max, m_max), -5.0)
        affinity[0, 1] = affinity[1, 0] = 5.0
        affinity[1, 2] = affinity[2, 1] = 5.0
        mask = torch.tensor([True, True, True, True])

        clusters = predict_clusters(affinity, mask, threshold=0.5)

        # {0,1,2} merged via transitive closure; 3 stays alone (singleton)
        sets = sorted([tuple(sorted(c)) for c in clusters])
        self.assertIn((0, 1, 2), sets)
        self.assertIn((3,), sets)
        self.assertEqual(len(clusters), 2)

    def test_padded_positions_ignored(self) -> None:
        m_max = 5
        affinity = torch.full((m_max, m_max), 5.0)  # all "strong" if not masked
        affinity.fill_diagonal_(-1e9)
        mask = torch.tensor([True, True, False, False, False])

        clusters = predict_clusters(affinity, mask, threshold=0.5)

        all_indices = sorted(idx for c in clusters for idx in c)
        self.assertEqual(all_indices, [0, 1])

    def test_empty_mask_returns_empty(self) -> None:
        affinity = torch.zeros(3, 3)
        mask = torch.tensor([False, False, False])

        clusters = predict_clusters(affinity, mask, threshold=0.5)

        self.assertEqual(clusters, [])

    def test_high_threshold_separates_into_singletons(self) -> None:
        affinity = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        affinity.fill_diagonal_(-1e9)
        mask = torch.tensor([True, True, True])

        # logit=1.0 -> prob ≈ 0.73; threshold 0.9 -> no edges
        clusters = predict_clusters(affinity, mask, threshold=0.9)

        # 3 singletons
        self.assertEqual(len(clusters), 3)


class CorefPairLossTests(unittest.TestCase):
    def test_ambiguous_pairs_excluded(self) -> None:
        # Use distinct logits so the masked-out pair contributes a different BCE term
        # and dropping it shifts the mean visibly.
        affinity = torch.tensor([[[0.0, 0.0, 8.0], [0.0, 0.0, 0.0], [8.0, 0.0, 0.0]]])
        labels = torch.zeros(1, 3, 3)
        labels[0, 0, 1] = labels[0, 1, 0] = 1.0
        eligible_full = torch.zeros(1, 3, 3, dtype=torch.bool)
        eligible_full[0, :, :] = True
        eligible_full[0, range(3), range(3)] = False  # exclude diagonal
        eligible_ambig = eligible_full.clone()
        eligible_ambig[0, 0, 2] = False  # mark (0,2) ambiguous
        eligible_ambig[0, 2, 0] = False

        loss_full = float(coref_pair_loss(affinity, labels, eligible_full, pos_weight=1.0).item())
        loss_ambig = float(coref_pair_loss(affinity, labels, eligible_ambig, pos_weight=1.0).item())

        # Dropping the high-logit (label=0) pair removes a large BCE term -> loss drops
        self.assertLess(loss_ambig, loss_full)

    def test_returns_zero_when_no_eligible(self) -> None:
        affinity = torch.randn(1, 3, 3)
        labels = torch.zeros(1, 3, 3)
        eligible = torch.zeros(1, 3, 3, dtype=torch.bool)

        loss = coref_pair_loss(affinity, labels, eligible, pos_weight=1.0)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_pos_weight_increases_loss_when_positives_present(self) -> None:
        affinity = torch.zeros(1, 3, 3)  # uniform logits => BCE = log(2) per cell
        labels = torch.zeros(1, 3, 3)
        labels[0, 0, 1] = labels[0, 1, 0] = 1.0
        eligible = torch.zeros(1, 3, 3, dtype=torch.bool)
        # only the (0,1) upper-tri eligible pair is counted
        eligible[0, 0, 1] = True
        eligible[0, 1, 0] = True  # lower-tri pair will be excluded by upper mask

        unweighted = float(coref_pair_loss(affinity, labels, eligible, pos_weight=1.0).item())
        weighted = float(coref_pair_loss(affinity, labels, eligible, pos_weight=5.0).item())

        self.assertGreater(weighted, unweighted)
        # at logit=0 with target=1 and pos_weight=w, BCE = w * log(2). With one eligible pair, ratio == w
        self.assertAlmostEqual(weighted / unweighted, 5.0, places=4)


class RecordPlannerCorefIntegrationTests(unittest.TestCase):
    def test_record_planner_coref_mode_builds_apcc(self) -> None:
        planner = RecordPlanner(
            hidden_size=4, num_event_types=2, k_max=5,
            count_head_mode="coref", num_roles=3,
        )
        self.assertEqual(planner.count_head_mode, "coref")
        self.assertIsInstance(planner.count_planner, ArgumentCoreferenceHead)

    def test_record_planner_coref_log_lambda_raises(self) -> None:
        planner = RecordPlanner(
            hidden_size=4, num_event_types=2, k_max=5,
            count_head_mode="coref", num_roles=3,
        )
        with self.assertRaises(RuntimeError):
            planner.count_log_lambda(torch.zeros((1, 4)), torch.tensor([0]))

    def test_record_planner_coref_affinity_routes_to_head(self) -> None:
        torch.manual_seed(0)
        planner = RecordPlanner(
            hidden_size=4, num_event_types=2, k_max=5,
            count_head_mode="coref", num_roles=3,
        )
        span_repr = torch.randn(2, 4, 4)
        span_role_id = torch.tensor([[1, 2, 3, 0], [1, 1, 2, 0]])
        sent_pos = torch.tensor([[1, 1, 2, 0], [1, 2, 3, 0]])
        type_id = torch.tensor([0, 1])
        mask = torch.tensor([
            [True, True, True, False],
            [True, True, True, False],
        ])

        affinity = planner.coref_affinity(span_repr, span_role_id, sent_pos, type_id, span_mask=mask)

        self.assertEqual(affinity.shape, (2, 4, 4))

    def test_predict_n_t_returns_cluster_count_when_typegate_fires(self) -> None:
        torch.manual_seed(0)
        planner = RecordPlanner(
            hidden_size=4, num_event_types=2, k_max=5,
            count_head_mode="coref", num_roles=3,
        )
        with torch.no_grad():
            # make TypeGate always fire
            planner.type_gate.proj.weight.zero_()
            planner.type_gate.proj.bias.fill_(10.0)
            # construct APCC weights so that mentions 0-1 cluster together, 2 alone
            head = planner.count_planner
            head.span_proj.weight.zero_()
            head.span_proj.bias.zero_()
            head.role_embedding.weight.zero_()
            head.sent_pos_embedding.weight.zero_()
            head.type_embedding.weight.zero_()
            head.feature_proj.weight.zero_()
            head.feature_proj.bias.zero_()
            head.same_role_bias.fill_(5.0)
            head.same_sentence_bias.zero_()

        global_repr = torch.zeros(1, 4)
        type_id = torch.tensor([0])
        # mentions 0,1 share role 1; mention 2 has role 2 -> two clusters
        span_repr = torch.zeros(1, 3, 4)
        span_role_id = torch.tensor([[1, 1, 2]])
        sent_pos = torch.tensor([[1, 2, 3]])
        mask = torch.tensor([[True, True, True]])

        # threshold > sigmoid(0) = 0.5 so the cross-role pairs (logit 0 -> prob 0.5) do not merge;
        # same-role pair (logit 5 -> prob ~0.99) does merge.
        n_t = planner.predict_n_t(
            global_repr, type_id,
            span_repr=span_repr,
            span_role_id=span_role_id,
            span_sent_pos=sent_pos,
            span_mask=mask,
            coref_threshold=0.9,
        )

        self.assertEqual(n_t, 2)

    def test_predict_n_t_returns_zero_when_typegate_off(self) -> None:
        planner = RecordPlanner(
            hidden_size=4, num_event_types=2, k_max=5,
            count_head_mode="coref", num_roles=3,
        )
        with torch.no_grad():
            planner.type_gate.proj.weight.zero_()
            planner.type_gate.proj.bias.fill_(-10.0)  # always off

        n_t = planner.predict_n_t(
            torch.zeros(1, 4), torch.tensor([0]),
            span_repr=torch.zeros(1, 3, 4),
            span_role_id=torch.tensor([[1, 1, 2]]),
            span_sent_pos=torch.tensor([[1, 1, 1]]),
            span_mask=torch.tensor([[True, True, True]]),
        )

        self.assertEqual(n_t, 0)

    def test_predict_n_t_minimum_one_when_typegate_fires_with_no_mentions(self) -> None:
        planner = RecordPlanner(
            hidden_size=4, num_event_types=2, k_max=5,
            count_head_mode="coref", num_roles=3,
        )
        with torch.no_grad():
            planner.type_gate.proj.weight.zero_()
            planner.type_gate.proj.bias.fill_(10.0)

        n_t = planner.predict_n_t(
            torch.zeros(1, 4), torch.tensor([0]),
            span_repr=torch.zeros(1, 3, 4),
            span_role_id=torch.tensor([[0, 0, 0]]),
            span_sent_pos=torch.tensor([[0, 0, 0]]),
            span_mask=torch.tensor([[False, False, False]]),
        )

        # TypeGate=1 but no candidate mentions -> default to 1 to avoid contradiction
        self.assertEqual(n_t, 1)


if __name__ == "__main__":
    unittest.main()
