"""LoRA-fine-tuned LLM relation extractor (server / CUDA).

Implements the evidence-grounded, "extract all relations for the event set at
once" formulation (cf. LLMERE, COLING 2025). This is the retained v3 generative
baseline; v4 Phase A adds a separate discriminative extractor. Differences from
a vanilla LLM extractor:

- the prompt requires an `evidence_quote` for every predicted relation, and
  predictions whose quote is not found in the source text are dropped by the
  grounding stage (anti-fabrication);
- relations are emitted jointly so the downstream consistency solver can
  enforce global structure.

Heavy imports (torch/transformers/peft) are lazy so this module imports on a
CPU-only box; instantiation requires the `llm` extra and a GPU.
"""

from __future__ import annotations

import json
import re

from finekg.core.schema import EventNode, EvidenceSpan, RelationEdge, RelationType
from finekg.relations.extractor.base import (
    ExtractionContext,
    RelationExtractor,
    relation_extractors,
)

__all__ = ["LLMRelationExtractor", "build_relation_prompt", "parse_relation_json"]

_SUBTYPE_BY_TYPE = {
    RelationType.COREFERENCE: "",
    RelationType.TEMPORAL: "BEFORE",
    RelationType.CAUSAL: "CAUSE",
    RelationType.SUBEVENT: "SUBEVENT_OF",
}

_PROMPT_TEMPLATE = """你是金融事件关系抽取器。给定同一/相关文档中的事件列表，判断 \
所有事件两两之间的关系，并为每条关系给出原文证据。只输出 JSON。

{document_block}事件列表：
{event_block}

关系类型取值：coreference(同一事件)、temporal(时序，subtype=BEFORE/OVERLAP)、\
causal(因果，subtype=CAUSE/PRECONDITION)、subevent(子事件)。
每条关系必须给出 evidence_quote（从上面原文中逐字摘取的短片段，用于核验）。
输出格式：{{"relations": [{{"head": <事件号>, "tail": <事件号>, "type": <类型>, \
"subtype": <子类型或空>, "evidence_quote": <原文片段>, "rationale": <简短理由>}}]}}
"""


def _doc_excerpt(nodes: list[EventNode], doc_text: str, max_chars: int) -> str:
    """The evidence-bearing slice of the document for the prompt.

    Grounding verifies quotes by substring search over the full `doc_text`, so
    the excerpt is assembled from verbatim lines of it (sentence ids first,
    trigger-string match as fallback) — quotes copied from the excerpt remain
    locatable in the original. Quotes spanning a join of non-adjacent lines
    will not ground, which is exactly the anti-fabrication contract.
    """
    if len(doc_text) <= max_chars:
        return doc_text
    lines = doc_text.split("\n")
    wanted = sorted(
        {
            span.sent_id
            for n in nodes
            for span in n.trigger_evidence
            if span.sent_id is not None and 0 <= span.sent_id < len(lines)
        }
    )
    if not wanted:
        triggers = [n.trigger for n in nodes if n.trigger]
        wanted = [i for i, line in enumerate(lines) if any(t in line for t in triggers)]
    excerpt = "\n".join(lines[i] for i in wanted) if wanted else doc_text
    return excerpt[:max_chars]


def build_relation_prompt(
    nodes: list[EventNode], doc_text: str = "", max_doc_chars: int = 1800
) -> str:
    """Render the one-shot, all-pairs extraction prompt.

    `doc_text` is the source document; without it the model has nothing to
    quote from and the evidence-grounding check (and the grounding reward in
    GRPO) is unsatisfiable, so every caller that has the text must pass it.
    """
    lines = []
    for i, n in enumerate(nodes):
        trigger = f"，触发词={n.trigger}" if n.trigger else ""
        args = "，".join(f"{k}={v}" for k, v in n.arguments.items())
        date = f"，时间={n.time_anchor}" if n.time_anchor else ""
        lines.append(f"[{i}] 类型={n.event_type}{trigger}{date}，论元: {args or '（无）'}")
    excerpt = _doc_excerpt(nodes, doc_text, max_doc_chars) if doc_text else ""
    document_block = f"原文（evidence_quote 必须逐字摘自其中）：\n{excerpt}\n\n" if excerpt else ""
    return _PROMPT_TEMPLATE.format(
        document_block=document_block, event_block="\n".join(lines)
    )


def parse_relation_json(
    text: str, nodes: list[EventNode], doc_id: str = ""
) -> list[RelationEdge]:
    """Parse the model's JSON into grounded `RelationEdge`s.

    Tolerant to code-fence wrapping and trailing prose; silently skips malformed
    items and out-of-range event indices.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    edges: list[RelationEdge] = []
    for item in payload.get("relations", []):
        try:
            head, tail = int(item["head"]), int(item["tail"])
            rel = RelationType(str(item["type"]).strip().lower())
            # Inside the guard and clamped: sampled completions emit things
            # like "confidence": "high" or 1.5, and an uncaught ValueError /
            # pydantic bound violation here would crash the GRPO training loop.
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.8))))
        except (KeyError, ValueError, TypeError):
            continue
        if not (0 <= head < len(nodes)) or not (0 <= tail < len(nodes)) or head == tail:
            continue
        quote = str(item.get("evidence_quote", ""))
        edges.append(
            RelationEdge(
                head_id=nodes[head].event_id,
                tail_id=nodes[tail].event_id,
                relation_type=rel,
                subtype=str(item.get("subtype", "") or _SUBTYPE_BY_TYPE.get(rel, "")),
                directed=rel != RelationType.COREFERENCE,
                confidence=confidence,
                evidence=[EvidenceSpan(doc_id=doc_id, char_start=0, char_end=0, text=quote)]
                if quote
                else [],
                rationale=str(item.get("rationale", "")),
            )
        )
    return edges


@relation_extractors.register("llm")
class LLMRelationExtractor(RelationExtractor):
    """Generation-based extractor backed by a (optionally LoRA-adapted) LLM."""

    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        max_new_tokens: int = 1024,
        device: str = "cuda",
    ) -> None:
        # Lazy, GPU-only imports.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(device).eval()

    def extract(
        self, nodes: list[EventNode], context: ExtractionContext | None = None
    ) -> list[RelationEdge]:
        if len(nodes) < 2:
            return []
        import torch

        doc_id = nodes[0].doc_id
        doc_text = context.doc_text.get(doc_id, "") if context else ""
        prompt = build_relation_prompt(nodes, doc_text=doc_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        text = self.tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return parse_relation_json(text, nodes, doc_id=doc_id)
