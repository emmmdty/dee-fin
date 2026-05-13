import math
import tempfile
import unittest
from pathlib import Path

import torch

from carve.datasets import DueeDocument
from carve.p2_heads import build_evidence_labels, pointer_mi_loss
from carve.p2_runner import build_arg_parser, run_p2
from carve.text_segmentation import split_sentences
from evaluator.canonical.schema import EventSchema
from evaluator.canonical.types import CanonicalEventRecord


class P2EvidencePointerTests(unittest.TestCase):
    def test_split_sentences_deterministic(self) -> None:
        text = "标题\n甲公司质押100万股。乙银行为质权方！“丙公司解除质押。”"

        first = split_sentences(text)
        second = split_sentences(text)

        self.assertEqual(first, second)
        self.assertEqual([sentence.text for sentence in first], ["标题", "甲公司质押100万股。", "乙银行为质权方！", "“丙公司解除质押。”"])

    def test_evidence_labels_role_mask(self) -> None:
        document = DueeDocument(
            document_id="doc1",
            title="公告",
            text="甲公司质押100万股。",
            records=[
                CanonicalEventRecord(
                    document_id="doc1",
                    event_type="质押",
                    arguments={"质押方": ["甲公司"], "不存在角色": ["100万股"]},
                )
            ],
        )
        schema = EventSchema.from_mapping({"质押": ["质押方", "质权方"]})

        labels = build_evidence_labels(document, split_sentences(document.text), schema)

        self.assertEqual(labels.y_ev_role.shape, (1, 1, 2))
        self.assertEqual(labels.y_ev_role[0, 0].tolist(), [1.0, 0.0])

    def test_evidence_labels_unalignable_recorded(self) -> None:
        document = DueeDocument(
            document_id="doc1",
            title="公告",
            text="甲公司发布公告。",
            records=[
                CanonicalEventRecord(
                    document_id="doc1",
                    event_type="质押",
                    arguments={"质押方": ["乙公司"]},
                )
            ],
        )
        schema = EventSchema.from_mapping({"质押": ["质押方"]})

        labels = build_evidence_labels(document, split_sentences(document.text), schema)

        self.assertEqual(labels.unalignable, [("质押", "质押方", "乙公司")])
        self.assertEqual(labels.pos_sent, {})
        self.assertEqual(labels.y_ev_type.tolist(), [[0.0]])

    def test_evidence_labels_or_for_type_label(self) -> None:
        document = DueeDocument(
            document_id="doc1",
            title="公告",
            text="甲公司质押100万股。乙银行接受质押。",
            records=[
                CanonicalEventRecord(
                    document_id="doc1",
                    event_type="质押",
                    arguments={"质押方": ["甲公司"], "质权方": ["乙银行"]},
                )
            ],
        )
        schema = EventSchema.from_mapping({"质押": ["质押方", "质权方"]})

        labels = build_evidence_labels(document, split_sentences(document.text), schema)

        self.assertEqual(labels.y_ev_role[:, 0, :].tolist(), [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(labels.y_ev_type[:, 0].tolist(), [1.0, 1.0])

    def test_pointer_mi_log_sum_exp(self) -> None:
        log_p = torch.log_softmax(torch.tensor([[1.0, 0.0, 2.0]], dtype=torch.float32), dim=-1)

        loss = pointer_mi_loss(log_p, [[0, 2]])

        expected = -torch.logsumexp(log_p[0, torch.tensor([0, 2])], dim=0)
        self.assertTrue(torch.allclose(loss, expected))

    def test_pointer_mi_mask_empty(self) -> None:
        log_p = torch.log_softmax(torch.tensor([[1.0, 0.0, 2.0], [4.0, 0.0, 0.0]], dtype=torch.float32), dim=-1)

        loss = pointer_mi_loss(log_p, [[0, 2], []])

        expected = -torch.logsumexp(log_p[0, torch.tensor([0, 2])], dim=0)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(math.isfinite(float(loss)))

    def test_smoke_run_evidence_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "schema.json").write_text(
                '[{"event_type":"质押","role_list":[{"role":"质押方"},{"role":"质押股票/股份数量"}]}]\n',
                encoding="utf-8",
            )
            line = (
                '{"id":"doc1","title":"公告","text":"甲公司质押100万股。",'
                '"event_list":[{"event_type":"质押","arguments":['
                '{"role":"质押方","argument":"甲公司"},'
                '{"role":"质押股票/股份数量","argument":"100万股"}]}]}\n'
            )
            (data_root / "train.jsonl").write_text(line * 3, encoding="utf-8")
            (data_root / "dev.jsonl").write_text(line, encoding="utf-8")
            run_dir = root / "run"

            args = build_arg_parser().parse_args(
                [
                    "--dataset",
                    "DuEE-Fin-dev500",
                    "--data-root",
                    str(data_root),
                    "--schema",
                    str(data_root / "schema.json"),
                    "--run-dir",
                    str(run_dir),
                    "--model-path",
                    "__toy__",
                    "--smoke",
                    "--max-epochs",
                    "2",
                ]
            )
            report = run_p2(args)

            self.assertEqual(report["status"], "p2_smoke")
            self.assertTrue((run_dir / "diagnostics" / "p2_train_history.json").exists())
            self.assertTrue((run_dir / "diagnostics" / "evidence_metrics.json").exists())
            self.assertTrue((run_dir / "checkpoints" / "p2.pt").exists())


if __name__ == "__main__":
    unittest.main()
