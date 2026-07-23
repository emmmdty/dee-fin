"""CPU tests for the discriminative supervised relation extractor (Phase A).

Registry wiring, torch-free construction, document-level candidate enumeration
and edge building from injected scores all run without torch. The model itself
(`PairClassifier`, pair features) is torch-guarded and tested under
`pytest.importorskip`, so it exercises on the GPU server and skips on CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import finekg.relations.extractor.supervised as sup
from finekg.core.schema import EventNode, EvidenceSpan
from finekg.relations.extractor import relation_extractors


def _node(eid: str, sent: int, start: int, etype: str = "Attack") -> EventNode:
    return EventNode(
        event_id=eid,
        event_type=etype,
        doc_id="d1",
        trigger=eid,
        trigger_evidence=[
            EvidenceSpan(doc_id="d1", char_start=start, char_end=start + 1, sent_id=sent)
        ],
    )


def test_supervised_registered():
    assert "supervised" in relation_extractors


def test_extractor_instantiates_torch_free():
    # __init__ must not build the model (lazy on first extract), so the pipeline
    # constructs on a CPU box; the module follows the succession/model.py guard.
    ex = relation_extractors.create("supervised", checkpoint_path=None)
    assert ex._model is None
    assert hasattr(sup, "TORCH_AVAILABLE")


def test_ensure_model_fails_fast_without_torch_or_checkpoint():
    # No silent degradation to an untrained model: on a CPU box this names the
    # llm extra; with torch installed it refuses because checkpoint_path is None.
    ex = relation_extractors.create("supervised", checkpoint_path=None)
    with pytest.raises((RuntimeError, ValueError)) as excinfo:
        ex._ensure_model()
    assert "llm" in str(excinfo.value) or "checkpoint_path" in str(excinfo.value)


def test_candidate_pairs_document_level_all_ordered_pairs():
    ex = relation_extractors.create("supervised")
    nodes = [_node("a", 0, 0), _node("b", 0, 5), _node("c", 1, 0)]
    pairs = ex._candidate_pairs(nodes)
    # Document-level = all ordered mention pairs (both directions), no self-pairs.
    assert len(pairs) == 6
    assert ("a", "b") in pairs and ("b", "a") in pairs
    assert all(h != t for h, t in pairs)


def test_locate_trigger_token_and_fail_fast():
    # offsets mimic a tokenizer's offset_mapping: (char_start, char_end) per token,
    # specials carrying (0, 0).
    offsets = [(0, 0), (0, 3), (4, 8), (0, 0)]  # <s>, "The", "bomb", </s>
    assert sup.locate_trigger_token("The bomb", "bomb", offsets) == 2
    with pytest.raises(ValueError):  # unlocatable -> raise, never read a wrong token
        sup.locate_trigger_token("The bomb", "missile", offsets)


def test_locate_trigger_token_is_case_insensitive():
    # MAVEN-ERE's trigger_word is lower-cased while the sentence keeps its original
    # casing, so a sentence-initial or proper-noun trigger ("armed" in "Armed police
    # officers ...") only matches case-insensitively. 0.65% of mentions hit this.
    offsets = [(0, 0), (0, 5), (6, 12), (0, 0)]  # <s>, "Armed", "police", </s>
    assert sup.locate_trigger_token("Armed police", "armed", offsets) == 1


def test_locate_trigger_token_respects_word_boundaries():
    # "arm" must not match inside "armed" -- a substring hit would pool a wrong token.
    offsets = [(0, 0), (0, 5), (6, 12), (0, 0)]  # <s>, "armed", "police", </s>
    with pytest.raises(ValueError):
        sup.locate_trigger_token("armed police", "arm", offsets)


def test_locate_trigger_token_raises_when_truncated_away():
    # A trigger past the tokenised span (long sentence + truncation) must raise
    # rather than pool a wrong token. max_length=512 is what keeps this from
    # firing on MAVEN-ERE (longest sentence = 322 BPE tokens).
    offsets = [(0, 0), (0, 5), (0, 0)]  # truncation kept only the first word
    with pytest.raises(ValueError, match="truncated"):
        sup.locate_trigger_token("armed police officers", "officers", offsets)


def test_extract_builds_grounded_edges_from_scores(monkeypatch):
    ex = relation_extractors.create("supervised")
    nodes = [_node("a", 0, 0), _node("b", 1, 0)]

    def fake_scores(self, ns, pairs, context):
        return {("a", "b"): {"causal": ("CAUSE", 0.9)}}  # only a->b causal

    monkeypatch.setattr(sup.SupervisedRelationExtractor, "_score_pairs", fake_scores)
    edges = ex.extract(nodes)
    assert len(edges) == 1
    e = edges[0]
    assert (e.head_id, e.tail_id) == ("a", "b")
    assert e.relation_type.value == "causal" and e.subtype == "CAUSE"
    assert e.directed is True
    assert abs(e.confidence - 0.9) < 1e-6
    assert len(e.evidence) >= 1  # grounded in the endpoints' trigger spans


def test_extract_no_prediction_yields_no_edge(monkeypatch):
    ex = relation_extractors.create("supervised")
    monkeypatch.setattr(
        sup.SupervisedRelationExtractor,
        "_score_pairs",
        lambda self, ns, pairs, context: {},
    )
    assert ex.extract([_node("a", 0, 0), _node("b", 1, 0)]) == []


def test_pair_classifier_and_features_shapes():
    torch = pytest.importorskip("torch")
    from finekg.relations.extractor.supervised import PairClassifier, _pair_features

    h = torch.zeros(3, 8)
    feats = _pair_features(h, h)
    assert feats.shape == (3, 32)  # [h; h; h*h; |h-h|] = 4 * 8
    model = PairClassifier(8, {"temporal": 7, "causal": 3, "subevent": 2})
    out = model(feats)
    assert out["causal"].shape == (3, 3)
    assert out["temporal"].shape == (3, 7)
    assert out["subevent"].shape == (3, 2)

    # The batched path (what extract/training use) must equal per-pair construction.
    a, b = torch.randn(4, 8), torch.randn(4, 8)
    per_pair = torch.stack([_pair_features(a[i], b[i]) for i in range(4)])
    assert torch.allclose(_pair_features(a, b), per_pair)


def test_supervised_config_wires_the_pipeline_on_cpu():
    # Loads the real config: checks its syntax, that the pipeline selects the
    # supervised extractor, and that construction needs no torch (model is lazy).
    from finekg.core.config import load_config
    from finekg.relations import RelationPipeline, RelationPipelineConfig

    repo = Path(__file__).resolve().parents[2]
    config = RelationPipelineConfig.from_dict(
        load_config(repo / "configs" / "relations" / "supervised.yaml")
    )
    assert config.extractor == "supervised"
    pipeline = RelationPipeline(config)
    assert type(pipeline.extractor).__name__ == "SupervisedRelationExtractor"
