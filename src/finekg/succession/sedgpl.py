"""The SeDGPL predictor: train on CGEP instances, score a candidate set.

Wires `linearize` -> `encode` -> `model` into the `SuccessorPredictor` contract,
so it is directly comparable with the torch-free `random` and `frequency`
baselines through the same `evaluate`.

Faithful to the original where it matters and explicit where it deviates:

* `batch_size` is 1 and gradients accumulate over `accumulate` instances. The
  original hardcodes 1 with no accumulation; its contrastive term is defined per
  instance, so batching it changes the loss, not just the schedule.
* `sample_rate` subsamples the training set once per fit, as `fewShot` does.
  `run.sh` passes 0.8, so SeDGPL's published numbers are on 80% of train.
* Instances whose mentions cannot be located are skipped in training, counted,
  and raised as `UnscorableInstance` at test time -- never scored flat. A flat
  score wins every optimistic tie-break, so silently failing on every test
  instance would report MRR 1.0.

**The event vocabulary is transductive.** Every distinct mention needs an `<a_i>`
token, and a test instance's candidates are mentions the training split may never
contain; SeDGPL ships a precomputed `data/to_add.json` covering all splits for
exactly this reason. So `vocabulary` must be built over train *and* test, and the
predictor refuses to guess: build it with `EventVocabulary.build(train + test)`
and pass it in. Only the token inventory crosses the split -- no labels, no
graphs, no gradients.

torch lives behind `model.TORCH_AVAILABLE`; the predictor registers on CPU and
raises on `fit`/`score`.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence

from finekg.succession.data.cgep import CgepInstance
from finekg.succession.encode import EncodedInstance, build_tokenizer, encode_instance
from finekg.succession.linearize import EDGE_BUDGET, EventVocabulary, edge_selectors
from finekg.succession.model import TORCH_AVAILABLE, build_sedgpl
from finekg.succession.predictor import (
    SuccessorPredictor,
    UnscorableInstance,
    successor_predictors,
)

__all__ = ["SeDGPLPredictor"]


@successor_predictors.register("sedgpl")
class SeDGPLPredictor(SuccessorPredictor):
    def __init__(
        self,
        model_path: str,
        *,
        vocabulary: EventVocabulary,
        epochs: int = 10,
        lr: float = 1e-6,
        weight_decay: float = 1e-2,
        sim_ratio: float = 0.5,
        sample_rate: float = 0.8,
        accumulate: int = 1,
        max_length: int = 200,
        max_edges: int = EDGE_BUDGET,
        edge_selector: str = "sedgpl",
        enable_structure: bool = False,
        seed: int = 209,
        device: str = "cuda",
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "SeDGPLPredictor needs torch + transformers: install the `llm` extra."
            )
        self.model_path = model_path
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.sim_ratio = sim_ratio
        self.sample_rate = sample_rate
        self.accumulate = accumulate
        self.max_length = max_length
        self.max_edges = max_edges
        self.edge_selector = edge_selector
        self.enable_structure = enable_structure
        self._select = edge_selectors.create(edge_selector)
        self.seed = seed
        self.device = device
        self.skipped_train = 0
        self._vocab = vocabulary
        self._tokenizer = None
        self._model = None

    def _encode(self, instance: CgepInstance) -> EncodedInstance | None:
        try:
            return encode_instance(
                instance, self._tokenizer, self._vocab,
                max_length=self.max_length, max_edges=self.max_edges,
                selector=self._select,
            )
        except ValueError:
            return None

    def _tensors(self, encoded: EncodedInstance):
        import torch

        def to(values) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long, device=self.device)

        return (
            to([encoded.template_ids]), to([encoded.template_mask]),
            to([encoded.type_ids]), to([encoded.type_mask]),
            to(encoded.sentence_ids), to(encoded.sentence_mask),
            to(encoded.candidate_token_ids),
        )

    def _forward(self, encoded: EncodedInstance):
        """Logits over `encoded.candidate_token_ids`, and the mask representation."""
        import torch

        model = self._model
        template, t_mask, types, ty_mask, sents, s_mask, candidates = self._tensors(encoded)

        sentence_out = model.sentence_model.roberta(sents, attention_mask=s_mask)[0]
        type_out = model.type_model.roberta(types, attention_mask=ty_mask)[0]
        word_emb = model.template_model.roberta.embeddings.word_embeddings(template).clone()

        positions = torch.tensor(encoded.event_positions, device=self.device)
        rows = torch.tensor(encoded.event_rows, device=self.device)
        offsets = torch.tensor(encoded.sentence_positions, device=self.device)

        instance_emb = word_emb[0, positions]
        sentence_emb = sentence_out[rows, offsets]
        type_emb = type_out[0, positions]
        reach = (
            torch.tensor(encoded.reach_anchor, dtype=torch.long, device=self.device)
            if self.enable_structure
            else None
        )
        word_emb[0, positions] = model.eece(
            instance_emb, sentence_emb, type_emb, reach_anchor=reach
        )

        hidden = model.template_model.roberta(attention_mask=t_mask, inputs_embeds=word_emb)[0]
        mask_emb = hidden[:, encoded.mask_index]
        logits = model.scep(mask_emb, model.template_model.lm_head, candidates)
        return logits, mask_emb, candidates

    def fit(self, instances: Sequence[CgepInstance]) -> None:
        import torch
        from torch.nn.functional import cross_entropy

        from finekg.succession.encode import subword_initialisers

        rng = random.Random(self.seed)
        torch.manual_seed(self.seed)

        self._tokenizer = build_tokenizer(self.model_path, self._vocab)
        self._model = build_sedgpl(
            self.model_path, len(self._tokenizer), enable_structure=self.enable_structure
        ).to(self.device)
        self._model.initialise_event_tokens(subword_initialisers(self._tokenizer, self._vocab))

        pool = list(instances)
        if 0.0 < self.sample_rate < 1.0:
            pool = rng.sample(pool, int(len(pool) * self.sample_rate))

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        embeddings = self._model.template_model.roberta.embeddings.word_embeddings

        self._model.train()
        start = time.perf_counter()
        for epoch in range(1, self.epochs + 1):
            rng.shuffle(pool)
            optimizer.zero_grad(set_to_none=True)
            running = torch.zeros((), device=self.device)
            seen = 0
            for step, instance in enumerate(pool, start=1):
                encoded = self._encode(instance)
                if encoded is None:
                    self.skipped_train += 1
                    continue
                logits, mask_emb, candidates = self._forward(encoded)
                gold = torch.tensor([encoded.label], device=self.device)
                loss = cross_entropy(logits, gold)
                loss = loss + self.sim_ratio * self._model.scep.similarity_loss(
                    mask_emb, embeddings(candidates), encoded.label
                )
                (loss / self.accumulate).backward()
                if step % self.accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                running += loss.detach()
                seen += 1
                if step % 1000 == 0:
                    print(f"[sedgpl] epoch {epoch}/{self.epochs} step {step}/{len(pool)} "
                          f"elapsed={time.perf_counter() - start:.0f}s", flush=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            mean = (running / max(seen, 1)).item()
            print(f"[sedgpl] epoch {epoch}/{self.epochs} done  loss={mean:.4f}  "
                  f"skipped={self.skipped_train}  elapsed={time.perf_counter() - start:.0f}s",
                  flush=True)

    def score(self, instance: CgepInstance) -> list[float]:
        import torch

        if self._model is None:
            raise RuntimeError("SeDGPLPredictor.score called before fit")
        encoded = self._encode(instance)
        if encoded is None:
            # Never return flat scores: they win every optimistic tie-break, so a
            # wholly broken predictor would report MRR 1.0.
            raise UnscorableInstance(instance.instance_id)
        self._model.eval()
        with torch.no_grad():
            logits, _, _ = self._forward(encoded)
        return logits.squeeze(0).tolist()
