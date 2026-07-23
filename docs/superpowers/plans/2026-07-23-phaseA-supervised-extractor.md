# Phase A: 判别式监督关系抽取器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) 或 subagent-driven-development 逐 task 实现。步骤用 `- [ ]` 勾选。

**Goal:** 注册判别式 `@register("supervised")` 关系抽取器，在金标 MAVEN-ERE 节点上做事件对 pair-classification，解生成式 causal 召回 0.4%(3/810) 瓶颈。

**Architecture:** encode-once RoBERTa 编码整篇 → 触发词表示 `h_i` → 事件对特征 `[h_i;h_j;h_i⊙h_j;|h_i−h_j|]` → 三组独立头(temporal/causal/subevent，各 `NONE`+subtypes)。候选/标签/评测复用 `relations/pairs.py`；训练新建(现有 `train_relation_extractor.py` 是 v3 生成式)。

**Tech Stack:** Python 3.10, PyTorch 2.6(lazy, GPU/`llm` extra), RoBERTa(transformers), pydantic 契约, pytest。

## Global Constraints (verbatim)

- 包/函数名**不得含 `ch1/ch2/ch3`**；新组件走 registry + lazy import；GPU 组件配 CPU 缓存回放/测试 skip。
- **`EventNode` schema 零新增字段**(扩展用 `metadata`)；`tests/core/test_propagation.py` 是测试锁。
- CPU 可导入整包；神经代码 **lazy import torch**(模块级不 import torch，方法内才 import)。
- `uv run pytest`(只增不改绿) + `uv run ruff check src tests scripts`(0 error, ≤100 列) + `uv run finekg-smoke`(绿)。
- 只在**金标节点**上评测(解耦 Ch1)；候选须**文档级**(非仅相邻句)。
- 类不平衡三选一(加权CE/focal/负采样)并**消融**。
- **代码简洁去冗余、fail-fast、无掩盖问题的 fallback/默认值**(缺 gold/数据显式报错或计数跳过)。
- 报数**如实**(降/未达也照报)。

---

### Task 1: 抽取器骨架 — registry 注册 + torch-lazy 导入 + 文档级候选枚举

**Files:**
- Create: `src/finekg/relations/extractor/supervised.py`
- Test: `tests/relations/test_supervised.py`

**Interfaces:**
- Produces: `SupervisedRelationExtractor(checkpoint_path: str | None = None, max_distance: int | None = None)`,
  registered as `"supervised"`; `extract(nodes, context) -> list[RelationEdge]`(Task 2 补全);
  helper `_candidate_pairs(nodes: list[EventNode]) -> list[tuple[str, str]]`.
- Consumes: `relations.pairs.candidate_pairs`, `relations.data.maven_ere.RelationDocument`, `relations.extractor.base`.

- [ ] **Step 1: 写失败测试**

```python
# tests/relations/test_supervised.py
from __future__ import annotations
import sys
from finekg.core.schema import EventNode, EvidenceSpan
from finekg.relations.extractor import relation_extractors
import finekg.relations.extractor.supervised as sup


def _node(eid: str, sent: int, start: int, etype: str = "Attack") -> EventNode:
    return EventNode(
        event_id=eid, event_type=etype, doc_id="d1", trigger=eid,
        trigger_evidence=[EvidenceSpan(doc_id="d1", char_start=start, char_end=start + 1, sent_id=sent)],
    )


def test_supervised_registered():
    assert "supervised" in relation_extractors


def test_module_imports_without_torch():
    # lazy import: importing the extractor module must not pull torch
    assert "torch" not in sys.modules or sys.modules["torch"] is not None
    # the module object itself has no module-level torch symbol
    assert not hasattr(sup, "torch")


def test_candidate_pairs_document_level_all_ordered_pairs():
    ex = relation_extractors.create("supervised")
    nodes = [_node("a", 0, 0), _node("b", 0, 5), _node("c", 1, 0)]
    pairs = ex._candidate_pairs(nodes)
    # document-level = all ordered pairs, both directions, no self-pairs
    assert len(pairs) == 6
    assert ("a", "b") in pairs and ("b", "a") in pairs
    assert all(h != t for h, t in pairs)
```

- [ ] **Step 2: 跑测试确认失败** — `uv run pytest tests/relations/test_supervised.py -q` → FAIL(module 不存在)

- [ ] **Step 3: 最小实现** — `supervised.py`(模块级**只** import 轻量；torch 不出现在模块级):

```python
"""Discriminative supervised relation extractor (RoBERTa pair-classification).

Reproduces the official MAVEN-ERE strong baseline: gold event mentions given,
every candidate mention pair labelled. Encode-once RoBERTa + per-family heads.
torch is imported lazily inside methods so the package imports on CPU without it.
"""
from __future__ import annotations

from finekg.core.schema import EventNode, RelationEdge
from finekg.relations.data.maven_ere import RelationDocument
from finekg.relations.extractor.base import ExtractionContext, RelationExtractor, relation_extractors
from finekg.relations.pairs import candidate_pairs

__all__ = ["SupervisedRelationExtractor"]


@relation_extractors.register("supervised")
class SupervisedRelationExtractor(RelationExtractor):
    def __init__(self, checkpoint_path: str | None = None, max_distance: int | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.max_distance = max_distance
        self._model = None  # lazy-loaded on first extract (needs torch)

    def _candidate_pairs(self, nodes: list[EventNode]) -> list[tuple[str, str]]:
        if not nodes:
            return []
        doc = RelationDocument(doc_id=nodes[0].doc_id, nodes=nodes, gold_edges=[])
        return candidate_pairs(doc, self.max_distance)

    def extract(self, nodes, context: ExtractionContext | None = None) -> list[RelationEdge]:
        raise NotImplementedError  # Task 2
```

- [ ] **Step 4: 确保 supervised 被 import(注册生效)** — 在 `src/finekg/relations/extractor/__init__.py` 加 `from . import supervised  # noqa: F401`(照 heuristic/llm 现有范式)。

- [ ] **Step 5: 跑测试确认通过**(除 extract 外) — `uv run pytest tests/relations/test_supervised.py -q -k "registered or imports or candidate"` → PASS

- [ ] **Step 6: 提交** — `git add -A && git commit -m "feat(relations): supervised extractor skeleton + doc-level candidate pairs"`

---

### Task 2: `extract()` 边生成(CPU 可测，monkeypatch 打分)

**Files:** Modify `src/finekg/relations/extractor/supervised.py`; Test `tests/relations/test_supervised.py`

**Interfaces:**
- Produces: `extract(nodes, context) -> list[RelationEdge]`; internal
  `_score_pairs(nodes, pairs, context) -> dict[tuple[str,str], dict[str, tuple[str, float]]]`
  (family value → (subtype, prob)，只含非 `NONE` 预测；Task 3 用模型实现)。
- Directed 规则: temporal/causal/subevent 全 `directed=True`；evidence = 两端 `trigger_evidence`。

- [ ] **Step 1: 写失败测试**

```python
def test_extract_builds_edges_from_scores(monkeypatch):
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
    assert len(e.evidence) >= 1  # grounded in endpoints' trigger spans


def test_extract_no_prediction_yields_no_edge(monkeypatch):
    ex = relation_extractors.create("supervised")
    monkeypatch.setattr(sup.SupervisedRelationExtractor, "_score_pairs",
                        lambda self, ns, pairs, context: {})
    assert ex.extract([_node("a", 0, 0), _node("b", 1, 0)]) == []
```

- [ ] **Step 2: 跑测试确认失败** — `... -k extract` → FAIL(NotImplementedError)

- [ ] **Step 3: 实现 extract**(替换 Task 1 的 raise；`_score_pairs` 暂 raise，Task 3 填):

```python
    _FAMILY_TYPE = {  # family value -> RelationType
        "temporal": RelationType.TEMPORAL,
        "causal": RelationType.CAUSAL,
        "subevent": RelationType.SUBEVENT,
    }

    def extract(self, nodes, context=None):
        pairs = self._candidate_pairs(nodes)
        if not pairs:
            return []
        by_id = {n.event_id: n for n in nodes}
        scored = self._score_pairs(nodes, pairs, context)
        edges: list[RelationEdge] = []
        for (head, tail), families in scored.items():
            for family, (subtype, prob) in families.items():
                edges.append(RelationEdge(
                    head_id=head, tail_id=tail,
                    relation_type=self._FAMILY_TYPE[family], subtype=subtype,
                    directed=True, confidence=prob,
                    evidence=list(by_id[head].trigger_evidence) + list(by_id[tail].trigger_evidence),
                ))
        return edges

    def _score_pairs(self, nodes, pairs, context):
        raise NotImplementedError  # Task 3: model-backed scoring
```

(加 `RelationType` 到模块级 import。)

- [ ] **Step 4: 跑测试确认通过** — `... -k extract` → PASS

- [ ] **Step 5: 提交** — `git commit -am "feat(relations): supervised.extract builds grounded edges from pair scores"`

---

### Task 3: `PairClassifier` 模型 + 模型化打分(torch，CPU skip)

**Files:** Modify `supervised.py`(加 `PairClassifier` + `_load_model` + `_score_pairs` 实现); Test `test_supervised.py`(torch-skip 块，照 `tests/succession/test_model_skip.py` 范式)。

**Interfaces:**
- Produces: `PairClassifier(nn.Module)` — `forward(pair_feats) -> dict[str, Tensor(logits)]`；三头 `temporal/causal/subevent`。
- `_load_model()` lazy import torch+transformers，load `checkpoint_path`(缺失则 **fail-fast 报错**，不静默用随机权重)。

- [ ] **Step 1: 写 torch-skip 测试**

```python
import pytest
torch = pytest.importorskip("torch")  # 整块 GPU 相关测试；本地无 torch 自动 skip

def test_pair_classifier_forward_shapes():
    from finekg.relations.extractor.supervised import PairClassifier
    model = PairClassifier(hidden_size=16, subtype_counts={"temporal": 7, "causal": 3, "subevent": 2})
    feats = torch.zeros(4, 16 * 4)  # [h_i; h_j; h_i*h_j; |h_i-h_j|]
    out = model(feats)
    assert out["causal"].shape == (4, 3)  # NONE + CAUSE + PRECONDITION
    assert out["temporal"].shape == (4, 7)
```

- [ ] **Step 2: 跑** — 本地无 torch → SKIP(如实，不算通过也不算失败)；服务器有 torch → 先 FAIL。

- [ ] **Step 3: 实现** `PairClassifier`(lazy import 在文件内用 `if TYPE_CHECKING` + 方法内 import；`nn.Module` 定义放工厂函数内或用 torch-guard 惯例——照 `succession/model.py` 现有范式)，`_load_model`(fail-fast: checkpoint 不存在直接 `raise FileNotFoundError`)，`_score_pairs`(编码整篇→取触发词表示→pair 特征→forward→argmax，非 NONE 收进结果)。

- [ ] **Step 4: 跑**(服务器/有 torch) → PASS；`uv run pytest -q`(本地) 该块 SKIP、其余绿。

- [ ] **Step 5: 提交** — `git commit -am "feat(relations): PairClassifier + model-backed pair scoring (torch-guarded)"`

---

### Task 4: 判别式训练脚本 + 数据准备(CPU 测负采样/权重)

**Files:** Create `scripts/train_supervised_relations.py`; Test `test_supervised.py`(负采样确定性，CPU)。

**Interfaces:**
- Produces(importable，CPU 可测): `build_training_rows(docs, max_distance) -> list[Row]`(复用 `pairs.pair_examples`),
  `downsample_negatives(rows, ratio, seed) -> list[Row]`(确定性), `class_weights(rows) -> dict`。

- [ ] **Step 1: 写失败测试**(CPU)

```python
def test_downsample_negatives_is_deterministic():
    from scripts.train_supervised_relations import downsample_negatives
    rows = [{"labels": {}} for _ in range(100)] + [{"labels": {"causal": "CAUSE"}} for _ in range(4)]
    a = downsample_negatives(rows, ratio=3.0, seed=13)
    b = downsample_negatives(rows, ratio=3.0, seed=13)
    assert a == b                       # same seed -> same subset
    n_pos = sum(1 for r in a if r["labels"])
    n_neg = sum(1 for r in a if not r["labels"])
    assert n_pos == 4 and n_neg == 12   # ratio 3:1 kept
```

- [ ] **Step 2: 跑确认失败**

- [ ] **Step 3: 实现** `downsample_negatives`(seed 化 `random.Random(seed)`，正例全留、负例按 ratio 采样；正例为 0 时**报错**不静默返回空)、`class_weights`、训练 `main()`(GPU：`pair_examples`→索引化→负采样→加权 CE→循环→存 checkpoint)。GPU 部分不写 CPU 测试。

- [ ] **Step 4: 跑确认通过**(CPU 测负采样)

- [ ] **Step 5: 提交** — `git commit -am "feat(scripts): supervised relation training + deterministic negative sampling"`

---

### Task 5: config + 评测接线 + 本地整体验收

**Files:** Create `configs/relations/supervised.yaml`; 本地跑三校验。

- [ ] **Step 1: 写 config**

```yaml
# configs/relations/supervised.yaml — Phase A discriminative extractor
data:
  loader: maven_ere
  path: data/raw/maven_ere/valid.jsonl
relations:
  extractor: supervised
  extractor_kwargs:
    checkpoint_path: runs/relations/supervised_maven
    max_distance: null          # document-level (no window)
  require_evidence: true        # edges carry endpoints' trigger spans -> kept
  consistency: greedy
```

- [ ] **Step 2: 测 config 装载选中 supervised**(CPU；`__init__` lazy 不需 torch)

```python
def test_pipeline_selects_supervised():
    from finekg.relations import RelationPipeline, RelationPipelineConfig
    cfg = RelationPipelineConfig.from_dict({"relations": {"extractor": "supervised",
        "extractor_kwargs": {"checkpoint_path": None}}})
    pipe = RelationPipeline(cfg)
    assert type(pipe.extractor).__name__ == "SupervisedRelationExtractor"
```

- [ ] **Step 3: 跑全量本地校验** — `uv run pytest && uv run ruff check src tests scripts && uv run finekg-smoke` → 全绿(torch 块 SKIP)。

- [ ] **Step 4: 提交 + 推送** — `git add -A && git commit -m "feat(relations): supervised eval config + pipeline wiring" && git push origin main`

---

### Task 6 (GPU，服务器 · handoff): 训练 + pair-setting 评测

> 本轮本地止步 Task 5；Task 6 待代码 push 后在服务器执行(card 1 空闲)。

- [ ] 服务器同步: `ssh gpu-4090; cd /data/TJK/Fin-EKG; git fetch origin && git reset --hard origin/main`
- [ ] 训练(card 1，screen/nohup)：`CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 /home/TJK/.local/bin/uv run --extra llm python -u scripts/train_supervised_relations.py --train data/raw/maven_ere/train.jsonl --model <roberta-base> --output runs/relations/supervised_maven > logs/train_supervised.log 2>&1 &`
- [ ] 预测 dump：`uv run --extra llm python scripts/evaluate_relations.py --config configs/relations/supervised.yaml --dump-predictions runs/pred_supervised.jsonl`
- [ ] pair 评测(CPU)：`uv run python scripts/evaluate_relation_pairs.py --predictions runs/pred_supervised.jsonl --gold-path data/raw/maven_ere/valid.jsonl --output runs/pair_eval_supervised.json`
- [ ] 回传 + **如实**写 `docs/TODO.md`：causal/subevent P/R/F1 对照 0.4%(降/未达也照报)。达不到走 PHASE_A 止损。

## Self-Review

- **Spec 覆盖**：PHASE_A Steps ①候选→Task1、②判别式实现+类不平衡→Task2/3/4、③evaluate 接线→Task5、④GPU 训练→Task6；设计稿 6 决策全落 task。✅
- **Placeholder**：无 TBD；每 code step 有代码或明确复用点(`pair_examples`/`candidate_pairs`)。Task3/4 的 GPU 实现给骨架+关键约束(fail-fast checkpoint、确定性采样)，执行时补全。
- **类型一致**：`_score_pairs` 返回 `dict[pair, dict[family, (subtype, prob)]]` 在 Task2 消费、Task3 产出，签名一致；`_candidate_pairs` Task1 定义、Task2 使用一致。✅
