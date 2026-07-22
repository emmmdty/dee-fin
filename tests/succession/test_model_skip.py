"""SeDGPL's modules import on CPU and fail loudly when built without torch."""

from __future__ import annotations

import pytest

from finekg.succession.model import TORCH_AVAILABLE, build_sedgpl


@pytest.mark.skipif(TORCH_AVAILABLE, reason="torch is installed")
def test_building_without_torch_names_the_extra_to_install():
    with pytest.raises(RuntimeError, match="llm` extra"):
        build_sedgpl("roberta-base", vocab_size=51273)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="needs torch")
def test_gated_fusion_interpolates_and_is_not_a_residual():
    import torch

    from finekg.succession.model import GatedFusion

    fusion = GatedFusion(4)
    x = torch.ones(1, 4)
    # With both projections zeroed the gate sits at sigmoid(0) = 0.5 exactly.
    torch.nn.init.zeros_(fusion.w_x.weight)
    torch.nn.init.zeros_(fusion.w_y.weight)
    out = fusion(x, torch.zeros(1, 4))
    assert torch.allclose(out, 0.5 * x)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="needs torch")
def test_gated_fusion_residual_is_identity_when_the_added_stream_is_zero():
    # The residual form `x + g*y` (M2's third gate) is a no-op at y=0 regardless
    # of the gate, which is what lets a zero-init structural stream start clean.
    import torch

    from finekg.succession.model import GatedFusion

    fusion = GatedFusion(4)
    x = torch.randn(2, 4)
    assert torch.allclose(fusion.residual(x, torch.zeros(2, 4)), x)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="needs torch")
def test_eece_stacks_two_gates_over_three_streams():
    import torch

    from finekg.succession.model import EeCE

    eece = EeCE(hidden_size=8)
    out = eece(torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 8))
    assert out.shape == (3, 8)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="needs torch")
def test_reach_embedding_is_zero_initialised():
    # The add-on must start as a no-op: a random N(0,1) embedding (norm ~28) would
    # swamp the fused event embedding (norm ~8) and corrupt training. Zero-init.
    import torch

    from finekg.succession.model import EeCE

    eece = EeCE(hidden_size=8, enable_structure=True)
    assert torch.count_nonzero(eece.reach_embed.weight) == 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="needs torch")
def test_eece_third_gate_is_identity_at_init():
    # Zero-init struct -> gated residual h2 + g*struct = h2 exactly, so the ON arm
    # starts identical to the two-gate baseline and can only learn to add signal.
    import torch

    from finekg.succession.model import EeCE

    eece = EeCE(hidden_size=8, enable_structure=True)
    inst, sent, typ = torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 8)
    out = eece(inst, sent, typ, reach_anchor=torch.tensor([1, 0, 1]))
    two_gate = eece.gate_type(eece.gate_sentence(inst, sent), typ)
    assert out.shape == (3, 8)
    assert torch.allclose(out, two_gate)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="needs torch")
def test_eece_third_gate_adds_structure_once_the_embedding_is_learned():
    # After the reach embedding moves off zero, the gate admits the structural
    # residual and the event representation changes.
    import torch

    from finekg.succession.model import EeCE

    eece = EeCE(hidden_size=8, enable_structure=True)
    eece.reach_embed.weight.data.normal_()
    inst, sent, typ = torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 8)
    out = eece(inst, sent, typ, reach_anchor=torch.tensor([1, 0, 1]))
    two_gate = eece.gate_type(eece.gate_sentence(inst, sent), typ)
    assert not torch.allclose(out, two_gate)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="needs torch")
def test_eece_without_structure_is_exactly_the_two_gate_baseline():
    # Byte-identical baseline: no third gate is built and none is applied.
    import torch

    from finekg.succession.model import EeCE

    eece = EeCE(hidden_size=8, enable_structure=False)
    inst, sent, typ = torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 8)
    out = eece(inst, sent, typ)
    baseline = eece.gate_type(eece.gate_sentence(inst, sent), typ)
    assert torch.allclose(out, baseline)
    assert not hasattr(eece, "reach_embed")
    assert not hasattr(eece, "gate_structure")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="needs torch")
def test_reach_embedding_is_binary():
    from finekg.succession.model import EeCE

    eece = EeCE(hidden_size=8, enable_structure=True)
    assert eece.reach_embed.num_embeddings == 2


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="needs torch")
def test_sedgpl_predictor_carries_the_structure_flag():
    # Selected by config like the edge selector; baseline stays the default (off).
    from finekg.succession.linearize import EventVocabulary
    from finekg.succession.sedgpl import SeDGPLPredictor

    vocab = EventVocabulary.build([])
    assert SeDGPLPredictor("m", vocabulary=vocab).enable_structure is False
    on = SeDGPLPredictor("m", vocabulary=vocab, enable_structure=True)
    assert on.enable_structure is True


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="needs torch")
def test_scep_similarity_loss_falls_as_the_anchor_approaches_gold():
    import torch

    from finekg.succession.model import ScEP

    scep = ScEP()
    candidates = torch.eye(4)
    aligned = scep.similarity_loss(candidates[0].unsqueeze(0), candidates, gold_index=0)
    opposed = scep.similarity_loss(-candidates[0].unsqueeze(0), candidates, gold_index=0)
    assert aligned < opposed


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="needs torch")
def test_sedgpl_predictor_resolves_the_named_edge_selector():
    # M1 is selected by config name, and the baseline stays the default.
    from finekg.succession.linearize import (
        EventVocabulary,
        select_nearest_edges,
        truncate_edges,
    )
    from finekg.succession.sedgpl import SeDGPLPredictor

    vocab = EventVocabulary.build([])
    assert SeDGPLPredictor("m", vocabulary=vocab)._select is truncate_edges
    distance = SeDGPLPredictor("m", vocabulary=vocab, edge_selector="distance")
    assert distance._select is select_nearest_edges
