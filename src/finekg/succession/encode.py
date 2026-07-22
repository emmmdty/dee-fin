"""Turn a linearised ECG into the tensors SeDGPL's encoder expects.

Reimplements `tools.get_batch` / `getSentence` / `getposHandler`. The whole file
exists to keep one invariant true: **the i-th event token in the template must be
paired with the i-th sentence encoding**. The original enforces this by walking
`relation` in order and asserting `len(sentId) - 1 == len(ePosition)`, which is a
count check, not an identity check -- it passes even if the pairing is shifted.
Here the node index travels alongside the position, so a shift is impossible.

The event-token order in a template is fixed by how it is rendered:

    <a_h0> rel <a_t0> , <a_h1> rel <a_t1> , ... , <a_anchor> rel <mask> .

so flattening the ordered edges and dropping the final tail (which is `<mask>`,
not a token) gives exactly the node behind each event token, in order.

Each event's sentence is encoded with its *own* mention swapped for its `<a_i>`
token (`util.doReplace`), so the sentence encoder and the template encoder refer
to the event by the same symbol.

transformers is imported behind an availability guard so the module imports on a
CPU box; encoding needs the `llm` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import punctuation

from finekg.succession.data.cgep import CgepInstance
from finekg.succession.linearize import (
    EdgeSelector,
    EventVocabulary,
    Linearization,
    linearize,
)
from finekg.succession.structure import event_reach_anchor

__all__ = [
    "TRANSFORMERS_AVAILABLE",
    "EncodedInstance",
    "build_tokenizer",
    "encode_instance",
    "event_token_nodes",
    "replace_mention",
]

try:  # pragma: no cover - exercised on the GPU server
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - the local CPU path
    TRANSFORMERS_AVAILABLE = False


def event_token_nodes(linearization: Linearization) -> list[int]:
    """Node index behind each event token of the template, in template order.

    Both endpoints of every template edge, then the anchor. The query edge's tail
    is the `<mask>`, so it contributes no token -- which is also why it must not
    appear anywhere else in the graph.
    """
    nodes: list[int] = []
    for head, _, tail in linearization.edges[:-1]:
        nodes.extend((head, tail))
    nodes.append(linearization.query_edge[0])
    return nodes


def replace_mention(sentence: str, span: tuple[int, int] | None, token: str) -> str:
    """Swap the trigger's tokens for `token` (`util.doReplace`).

    Punctuation glued to the last trigger token ("died.") survives as its own
    word, matching ESC's pre-tokenised sentences. An unknown or out-of-range span
    leaves the sentence alone; `encode_instance` then refuses to encode it rather
    than silently reading the event off position 0.
    """
    words = sentence.split()
    if span is None or not (0 <= span[0] < span[1] <= len(words)):
        return sentence
    last = words[span[1] - 1]
    suffix = last[len(last.rstrip(punctuation)) :]
    replacement = [token, suffix] if suffix else [token]
    return " ".join([*words[: span[0]], *replacement, *words[span[1] :]])


@dataclass(frozen=True)
class EncodedInstance:
    """Token ids and the positions that bind template events to their sentences."""

    template_ids: list[int]
    template_mask: list[int]
    type_ids: list[int]
    type_mask: list[int]
    mask_index: int
    event_positions: list[int]  # positions of `<a_i>` tokens in the template
    event_rows: list[int]  # row of `sentence_ids` holding that event's sentence
    sentence_positions: list[int]  # position of `<a_i>` inside that row
    reach_anchor: list[int]  # 1 if that token's event reaches the anchor (M2)
    sentence_ids: list[list[int]]
    sentence_mask: list[list[int]]
    candidate_token_ids: list[int]
    label: int


def build_tokenizer(model_path: str, vocab: EventVocabulary):
    """RoBERTa tokenizer with one added token per distinct mention and type.

    Tokens are added as `<a_0> ... <a_n>` in the vocabulary's own order, so their
    ids are contiguous and `id >= id_of('<a_0>')` identifies an event token. The
    `<mask>` id stays below that boundary.
    """
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "build_tokenizer needs transformers: install the `llm` extra "
            "(uv sync --extra llm)."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_tokens([f"<a_{i}>" for i in range(len(vocab))])
    return tokenizer


def subword_initialisers(tokenizer, vocab: EventVocabulary) -> dict[int, list[int]]:
    """`<a_i>` id -> the subword ids of its surface form (`data/to_add.json`).

    `SeDGPL.initialise_event_tokens` averages these into the new embedding rows.
    Without it the type template -- which is nothing but added tokens -- is noise.
    """
    to_add: dict[int, list[int]] = {}
    for token, surface in vocab.to_add.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        subwords = tokenizer(surface.strip(), add_special_tokens=False)["input_ids"]
        to_add[token_id] = subwords
    return to_add


def encode_instance(
    instance: CgepInstance,
    tokenizer,
    vocab: EventVocabulary,
    *,
    max_length: int = 200,
    max_edges: int = 20,
    selector: EdgeSelector | None = None,
) -> EncodedInstance:
    """Encode one instance, keeping every event token bound to its sentence."""
    linearization = linearize(instance, vocab, max_edges=max_edges, selector=selector)
    first_event_id = tokenizer.convert_tokens_to_ids("<a_0>")

    def encode(text: str) -> tuple[list[int], list[int]]:
        out = tokenizer(
            text, add_special_tokens=True, padding="max_length",
            max_length=max_length, truncation=True,
        )
        return out["input_ids"], out["attention_mask"]

    template_ids, template_mask = encode(linearization.template)
    type_ids, type_mask = encode(linearization.type_template)

    positions = [i for i, tid in enumerate(template_ids) if tid >= first_event_id]
    nodes = event_token_nodes(linearization)
    if len(positions) != len(nodes):
        # Truncation ate an event token; the pairing would silently shift.
        raise ValueError(
            f"{instance.instance_id}: {len(positions)} event tokens in a template "
            f"of {len(nodes)} events -- raise max_length or lower max_edges"
        )

    # One sentence row per distinct event, encoded with its own mention swapped
    # for the event token so both encoders name the event identically.
    rows: dict[int, int] = {}
    sentence_ids: list[list[int]] = []
    sentence_mask: list[list[int]] = []
    sentence_positions: list[int] = []
    for node_index in nodes:
        if node_index not in rows:
            node = instance.nodes[node_index]
            token = vocab.token(node.trigger)
            ids, mask = encode(replace_mention(node.sentence, node.token_span, token))
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id not in ids:
                # An unlocatable or truncated-away mention would otherwise read
                # the event's representation off `<s>` at position 0.
                raise ValueError(
                    f"{instance.instance_id}: event token {token} missing from the "
                    f"sentence of node {node.node_id} (span={node.token_span}) -- "
                    "the mention was not located, or truncation dropped it"
                )
            rows[node_index] = len(sentence_ids)
            sentence_ids.append(ids)
            sentence_mask.append(mask)
            sentence_positions.append(ids.index(token_id))

    mask_id = tokenizer.mask_token_id
    if mask_id not in template_ids:
        raise ValueError(f"{instance.instance_id}: template lost its <mask> to truncation")

    return EncodedInstance(
        template_ids=template_ids,
        template_mask=template_mask,
        type_ids=type_ids,
        type_mask=type_mask,
        mask_index=template_ids.index(mask_id),
        event_positions=positions,
        event_rows=[rows[n] for n in nodes],
        sentence_positions=[sentence_positions[rows[n]] for n in nodes],
        reach_anchor=event_reach_anchor(
            linearization.edges[:-1], nodes, linearization.query_edge[0]
        ),
        sentence_ids=sentence_ids,
        sentence_mask=sentence_mask,
        candidate_token_ids=[
            tokenizer.convert_tokens_to_ids(vocab.token(c.trigger)) for c in instance.candidates
        ],
        label=instance.label,
    )
