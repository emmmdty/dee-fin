"""SeDGPL's encoder and head: EeCE gated fusion + ScEP contrastive prediction.

Reimplemented from `model.py` of github.com/zhanchuanhong/SeDGPL (2 commits, no
license, MAVEN build unreleased), because M1/M2/M3 all hang off these modules and
patching someone else's untested script is not a base to build on.

**EeCE** enriches each event token's *input* embedding before the encoder runs,
by fusing three views of the event through two sigmoid gates::

    g1 = sigma(W1_1 @ inst + W1_2 @ sent);  h1 = g1 * inst + (1 - g1) * sent
    g2 = sigma(W2_1 @ h1   + W2_2 @ type);  h2 = g2 * h1   + (1 - g2) * type

`inst` is the event token's word embedding, `sent` its trigger position in the
sentence encoding, `type` its position in a parallel type-only template. `h2`
overwrites the token in `inputs_embeds`. M2 adds a third gate over a structural
stream; `GatedFusion` is factored out so that is a composition, not a rewrite.

**ScEP** reads the `<mask>` position, projects it through the MLM head, and keeps
only the candidate token ids. Training adds an InfoNCE term over candidate word
embeddings. M3's selective head (`succession.selective`) layers a conformal
prediction set over these candidate scores -- it wraps the predictor, so ScEP's
arg-max stays the point prediction and the set is added on top, not carved in.

torch is imported behind an availability guard so this module imports (and the
predictor registers) on a CPU box; building needs the `llm` extra and a GPU.
"""

from __future__ import annotations

__all__ = ["TORCH_AVAILABLE", "build_sedgpl"]

try:  # pragma: no cover - exercised on the GPU server
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import RobertaForMaskedLM

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - the local CPU path
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class GatedFusion(nn.Module):
        """`g = sigma(Wx @ x + Wy @ y)`, then `g * x + (1 - g) * y`.

        Both projections are bias-free, as in the original. `y` is the stream
        being mixed *in*, so `fuse(x, zeros)` reduces to `sigma(Wx @ x) * x`,
        not to `x` -- a gate is not a residual, and M2's ablations depend on
        knowing which.
        """

        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.w_x = nn.Linear(hidden_size, hidden_size, bias=False)
            self.w_y = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            gate = torch.sigmoid(self.w_x(x) + self.w_y(y))
            return gate * x + (1.0 - gate) * y

        def residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """`x + g * y`, an add-on that is a no-op when `y` is zero.

            M2's third gate uses this over a zero-initialised structural stream, so
            it starts as the exact identity on `x` (the two-gate output) and can
            only *add* structure as the embedding learns -- unlike `forward`, whose
            interpolation replaces half of `x` at init and, with a random-scale `y`,
            corrupts the event embedding the encoder reads.
            """
            gate = torch.sigmoid(self.w_x(x) + self.w_y(y))
            return x + gate * y

    class EeCE(nn.Module):
        """Gated fusion of the event views: instance, sentence, type, (+structure).

        With ``enable_structure`` the M2 stream is added: a per-token ``reach_anchor``
        bit is embedded and admitted through a third gate stacked after the type
        gate as a *residual*, ``h3 = h2 + g3 * struct``. The embedding is
        zero-initialised, so ``struct`` is zero at init and ``h3 = h2`` exactly --
        the ON arm starts identical to the two-gate baseline and can only learn to
        *add* structure. (An interpolation gate ``g3*h2 + (1-g3)*struct`` over a
        default N(0,1) embedding was tried first and halved MRR: at init that
        embedding's norm ~28 dwarfs the fused ~8, so it replaced the event
        representation with noise the lr=1e-6 schedule could not undo.)

        With ``enable_structure`` off, neither the embedding nor the gate is
        constructed and ``forward`` returns the two-gate result unchanged -- so the
        baseline is byte-identical (same submodules, same init RNG draws), and the
        existing seed-209 run doubles as the A/B off-arm.
        """

        def __init__(self, hidden_size: int = 768, enable_structure: bool = False) -> None:
            super().__init__()
            self.gate_sentence = GatedFusion(hidden_size)
            self.gate_type = GatedFusion(hidden_size)
            self.enable_structure = enable_structure
            if enable_structure:
                self.reach_embed = nn.Embedding(2, hidden_size)
                nn.init.zeros_(self.reach_embed.weight)
                self.gate_structure = GatedFusion(hidden_size)

        def forward(
            self,
            instance_emb: torch.Tensor,
            sentence_emb: torch.Tensor,
            type_emb: torch.Tensor,
            reach_anchor: torch.Tensor | None = None,
        ) -> torch.Tensor:
            fused = self.gate_sentence(instance_emb, sentence_emb)
            h2 = self.gate_type(fused, type_emb)
            if not self.enable_structure:
                return h2
            struct = self.reach_embed(reach_anchor)
            return self.gate_structure.residual(h2, struct)

    class ScEP(nn.Module):
        """Score candidates at the mask, with the contrastive term SeDGPL adds.

        `similarity_loss` is InfoNCE over the *word embeddings* of the candidate
        tokens, pulling the mask representation towards gold's embedding:

            -log( sum exp cos(anchor, pos) / (sum exp cos(anchor, pos) + sum exp cos(anchor, neg)) )
        """

        def forward(
            self,
            mask_emb: torch.Tensor,
            lm_head: nn.Module,
            candidate_ids: torch.Tensor,
        ) -> torch.Tensor:
            logits = lm_head(mask_emb)
            return logits.index_select(-1, candidate_ids)

        def similarity_loss(
            self,
            mask_emb: torch.Tensor,
            candidate_embeddings: torch.Tensor,
            gold_index: int,
        ) -> torch.Tensor:
            positive = candidate_embeddings[gold_index].unsqueeze(0)
            negatives = torch.cat(
                (candidate_embeddings[:gold_index], candidate_embeddings[gold_index + 1 :])
            )
            pos = torch.exp(F.cosine_similarity(mask_emb, positive, dim=-1)).sum()
            neg = torch.exp(F.cosine_similarity(mask_emb, negatives, dim=-1)).sum()
            return -torch.log(pos / (pos + neg))

    class SeDGPL(nn.Module):
        """RoBERTa-MLM over an EeCE-enriched template, read out by ScEP.

        Three encoders, as in the original: one for the masked template (whose
        input embeddings we overwrite), one for the sentences, one for the type
        template. They start as copies of the same checkpoint and are all trained.
        """

        def __init__(
            self,
            model_name: str,
            vocab_size: int,
            hidden_size: int = 768,
            enable_structure: bool = False,
        ) -> None:
            super().__init__()
            self.template_model = RobertaForMaskedLM.from_pretrained(model_name)
            self.template_model.resize_token_embeddings(vocab_size)
            self.sentence_model = RobertaForMaskedLM.from_pretrained(model_name)
            self.sentence_model.resize_token_embeddings(vocab_size)
            self.type_model = RobertaForMaskedLM.from_pretrained(model_name)
            self.type_model.resize_token_embeddings(vocab_size)
            self.eece = EeCE(hidden_size, enable_structure=enable_structure)
            self.scep = ScEP()

        def initialise_event_tokens(self, to_add: dict[int, list[int]]) -> None:
            """Mean-init each added `<a_i>` from the subwords of its surface form.

            `model.handler` in the original. Without it the new rows are random,
            and the type template -- which is nothing but added tokens -- is noise.
            """
            weights = self.template_model.roberta.embeddings.word_embeddings.weight
            with torch.no_grad():
                for token_id, subword_ids in to_add.items():
                    if subword_ids:
                        weights[token_id] = weights[torch.tensor(subword_ids)].mean(dim=0)

    def build_sedgpl(model_name: str, vocab_size: int, **kwargs: object) -> SeDGPL:
        return SeDGPL(model_name, vocab_size, **kwargs)  # type: ignore[arg-type]

else:

    def build_sedgpl(model_name: str, vocab_size: int, **kwargs: object):
        raise RuntimeError(
            "build_sedgpl needs torch + transformers: install the `llm` extra "
            "(uv sync --extra llm). This is a GPU-only component."
        )
