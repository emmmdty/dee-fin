"""Load SeDGPL's released CGEP-ESC build (`ESCSubWoRe.npy`).

This is the only CGEP corpus SeDGPL published, so it is our reproduction anchor:
if our linearisation, encoder and head cannot recover its reported ESC numbers on
its own data, the fault is ours and not the MAVEN rebuild's.

Two facts about the release shape the loader.

**It is a pickle from a third-party repo.** `numpy.load(allow_pickle=True)` will
execute whatever the stream says, so `load_npy_object` unpickles behind a
whitelist that admits only the three names a pickled ndarray needs. Static
disassembly (`pickletools.genops`) of the released file shows exactly those three
and no `STACK_GLOBAL`/`INST`/`REDUCE`-into-builtins opcodes; the whitelist keeps
that true for any future copy. sha256
``8ec791fb609cadf2ba1c8589d3f18ce1fac95b50c57203f1b25472d8438e5026``.

**The released data does not fit the released code.** `load_data.py` reads
``data['train'] / ['valid'] / ['test']``, but the file is keyed by EventStoryLine
*topic*: ``data[topic][doc_id] -> [instance, ...]`` over 22 topics, 244 documents
and 1192 instances (the paper says 243 / 1191). There is no train/valid/test
split to load, so ESC has to be evaluated by cross-validation over topics, as
EventStoryLine conventionally is. Which split produced the paper's ESC MRR of
19.6 is not stated, so any comparison against it must name its own protocol.

Node tuples are already flattened past `processNode` (9 fields), edges carry the
single relation string ``'cause'``, and candidate sets hold 256 distinct event
nodes.
"""

from __future__ import annotations

import io
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from finekg.succession.data.cgep import CgepInstance, CgepNode

__all__ = [
    "ESC_CANDIDATE_SET_SIZE",
    "load_cgep_esc",
    "load_npy_object",
    "topic_folds",
]

ESC_CANDIDATE_SET_SIZE = 256

# The three names a pickled numpy array reconstructs through. Nothing else.
_ALLOWED_GLOBALS = frozenset(
    {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),  # numpy >= 2.0 spelling
    }
)

# Field offsets inside an ESC node tuple, post-`processNode`.
_TOPIC, _DOC, _INDEX, _TYPE, _MENTION, _SENTENCE, _SENT_ID, _PLACE = 1, 2, 3, 4, 5, 6, 7, 8


def _token_span(place: object) -> tuple[int, int] | None:
    """Parse ESC's `'_2'` / `'_1_2'` event-place field into a token span.

    `util.doCorrect` has already made these indices contiguous, so the span is
    just their extent.
    """
    parts = [p for p in str(place).split("_") if p]
    if not parts or not all(p.isdigit() for p in parts):
        return None
    indices = [int(p) for p in parts]
    return (min(indices), max(indices) + 1)


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Refuse to import anything but numpy's array-reconstruction helpers."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked pickle import: {module}.{name}")


def load_npy_object(path: str | Path) -> Any:
    """Unpickle the object stored in an ``allow_pickle`` .npy, behind a whitelist.

    Reads the header itself rather than calling `numpy.load`, so the restricted
    unpickler is the only thing that ever touches the payload.
    """
    raw = Path(path).read_bytes()
    if raw[:6] != b"\x93NUMPY":
        raise ValueError(f"{path} is not a .npy file")
    major = raw[6]
    if major == 1:
        header_len = int.from_bytes(raw[8:10], "little")
        offset = 10 + header_len
    elif major in (2, 3):
        header_len = int.from_bytes(raw[8:12], "little")
        offset = 12 + header_len
    else:
        raise ValueError(f"unsupported .npy version {major}")
    array = _NumpyOnlyUnpickler(io.BytesIO(raw[offset:])).load()
    return array.item() if getattr(array, "shape", None) == () else array


def _node(raw: tuple) -> CgepNode:
    sentence = str(raw[_SENTENCE])
    trigger = str(raw[_MENTION])
    span = _token_span(raw[_PLACE]) if len(raw) > _PLACE else None

    # `util.doCorrect`: 27 of ESC's 2824 mentions skip interior words ("keep a
    # hold on" over "keep a comfortable hold on"). SeDGPL widens the mention to
    # the contiguous span rather than the other way round, so the added `<a_i>`
    # token stands for text that actually occurs in the sentence.
    if span is not None:
        words = sentence.split()
        contiguous = " ".join(words[span[0] : span[1]])
        if contiguous and contiguous != trigger.strip():
            trigger = contiguous + " "

    sent_id = raw[_SENT_ID]
    return CgepNode(
        node_id=f"{raw[_TOPIC]}::{raw[_DOC]}::{raw[_INDEX]}",
        event_type=str(raw[_TYPE]),
        # Mentions keep their trailing space: SeDGPL keys its added vocabulary by
        # the exact string, so stripping here would merge distinct answer tokens.
        trigger=trigger,
        sentence=sentence,
        sent_id=int(sent_id) if isinstance(sent_id, int) else None,
        token_span=span,
    )


def load_cgep_esc(path: str | Path) -> dict[str, list[CgepInstance]]:
    """Topic -> CGEP instances, in the same shape as the MAVEN rebuild."""
    data = load_npy_object(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a topic-keyed dict")

    by_topic: dict[str, list[CgepInstance]] = {}
    for topic in sorted(data, key=lambda t: int(t) if str(t).isdigit() else 0):
        instances: list[CgepInstance] = []
        for doc_id, raw_instances in data[topic].items():
            for position, raw in enumerate(raw_instances):
                nodes = tuple(_node(n) for n in raw["node"])
                candidates = tuple(_node(c) for c in raw["candiSet"])
                edges = tuple((int(h), str(rel), int(t)) for h, rel, t in raw["edge"])
                instances.append(
                    CgepInstance(
                        instance_id=f"{doc_id}::{position}",
                        doc_id=str(doc_id),
                        nodes=nodes,
                        edges=edges,
                        candidates=candidates,
                        label=int(raw["label"]),
                    )
                )
        by_topic[str(topic)] = instances
    return by_topic


def topic_folds(topics: list[str], n_folds: int = 5) -> Iterator[tuple[list[str], list[str]]]:
    """Deterministic round-robin topic folds, yielding (train, test) topic ids.

    EventStoryLine is evaluated by cross-validation over topics, never by a
    document split: documents inside one topic narrate the same story, so a
    document-level split leaks the causal chain across train and test.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    ordered = sorted(topics, key=lambda t: int(t) if str(t).isdigit() else 0)
    if len(ordered) < n_folds:
        raise ValueError(f"{len(ordered)} topics cannot fill {n_folds} folds")
    for fold in range(n_folds):
        test = ordered[fold::n_folds]
        yield [t for t in ordered if t not in set(test)], test
