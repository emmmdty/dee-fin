import tempfile
import unittest
from pathlib import Path

import torch

from carve.datasets import DueeDocument, build_candidate_lexicon, generate_inference_candidates
from carve.p3_planner import RecordPlanner
from carve.p5b_runner import PlannerGate, build_arg_parser, run_p5b
from evaluator.canonical.types import CanonicalEventRecord


def _save_planner_checkpoint(
    path: Path,
    *,
    event_types: list[str],
    hidden_size: int = 4,
    k_clip: int = 3,
    presence_bias: float = 10.0,
    count_bias_log_rate: float = 0.0,
    encoder_feature_mode: str = "global_only",
    presence_threshold: float = 0.5,
) -> None:
    planner = RecordPlanner(hidden_size=hidden_size, num_event_types=len(event_types), k_max=k_clip)
    with torch.no_grad():
        planner.type_gate.proj.weight.zero_()
        planner.type_gate.proj.bias.fill_(presence_bias)
        planner.count_planner.proj.weight.zero_()
        planner.count_planner.proj.bias.fill_(count_bias_log_rate)
    payload = {
        "planner": planner.state_dict(),
        "planner_metadata": {
            "event_types": event_types,
            "hidden_size": hidden_size,
            "k_clip": k_clip,
            "presence_pos_weight": 1.0,
            "presence_threshold_multi_event_dev": presence_threshold,
            "presence_threshold_all_dev": presence_threshold,
            "acceptance_population": ["multi_event_dev", "all_dev"],
            "train_population": "all_train",
            "encoder_feature_mode": encoder_feature_mode,
            "max_sentences": 256,
        },
    }
    torch.save(payload, path)


class P5BDiagnosticRunnerTests(unittest.TestCase):
    def test_inference_candidates_are_generated_without_dev_gold_values(self) -> None:
        train_doc = DueeDocument(
            document_id="train1",
            text="甲公司发布公告，甲公司质押100万股。",
            title="训练公告",
            records=[
                CanonicalEventRecord(
                    document_id="train1",
                    event_type="质押",
                    arguments={"质押方": ["甲公司"], "质押股票/股份数量": ["100万股"]},
                )
            ],
        )
        dev_doc = DueeDocument(
            document_id="dev1",
            text="乙公司发布公告，乙公司质押200万股。",
            title="开发公告",
            records=[
                CanonicalEventRecord(
                    document_id="dev1",
                    event_type="质押",
                    arguments={"质押方": ["乙公司"], "质押股票/股份数量": ["200万股"]},
                )
            ],
        )

        lexicon = build_candidate_lexicon([train_doc], min_count=1)
        candidates = generate_inference_candidates(dev_doc, lexicon, event_type="质押", role="质押方")

        self.assertEqual([candidate.value for candidate in candidates], [])

    def test_smoke_run_writes_dev_predictions_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "schema.json").write_text(
                '[{"event_type":"质押","role_list":[{"role":"质押方"},{"role":"质押股票/股份数量"}]}]\n',
                encoding="utf-8",
            )
            train_line = (
                '{"id":"train1","title":"训练公告","text":"甲公司质押100万股。",'
                '"event_list":[{"event_type":"质押","arguments":['
                '{"role":"质押方","argument":"甲公司"},'
                '{"role":"质押股票/股份数量","argument":"100万股"}]}]}\n'
            )
            dev_line = (
                '{"id":"dev1","title":"开发公告","text":"甲公司质押100万股。",'
                '"event_list":[{"event_type":"质押","arguments":['
                '{"role":"质押方","argument":"甲公司"},'
                '{"role":"质押股票/股份数量","argument":"100万股"}]}]}\n'
            )
            (data_root / "train.jsonl").write_text(train_line, encoding="utf-8")
            (data_root / "dev.jsonl").write_text(dev_line, encoding="utf-8")
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
                    "--seed",
                    "7",
                    "--max-epochs",
                    "2",
                    "--batch-size",
                    "2",
                    "--routes",
                    "baseline,carve",
                    "--smoke",
                ]
            )
            report = run_p5b(args)

            self.assertEqual(report["dataset"], "DuEE-Fin-dev500")
            self.assertTrue((run_dir / "canonical" / "dev.baseline.pred.jsonl").exists())
            self.assertTrue((run_dir / "canonical" / "dev.carve.pred.jsonl").exists())
            self.assertTrue((run_dir / "eval" / "dev.baseline.unified_strict.json").exists())
            self.assertTrue((run_dir / "eval" / "dev.carve.unified_strict.json").exists())
            self.assertTrue((run_dir / "diagnostics" / "p5b_duee_fin_decision_row.json").exists())

    def test_planner_gate_loads_with_valid_global_only_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "r3_planner.pt"
            _save_planner_checkpoint(checkpoint_path, event_types=["质押", "收购"])

            gate = PlannerGate(
                checkpoint_path=checkpoint_path,
                encoder=None,
                feature_mode="global_only",
                presence_threshold=0.5,
                device=torch.device("cpu"),
            )
            self.assertEqual(gate.event_types, ["质押", "收购"])
            self.assertEqual(gate.feature_mode, "global_only")
            self.assertEqual(gate.k_clip, 3)

    def test_planner_gate_feature_mode_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "r3_planner.pt"
            _save_planner_checkpoint(
                checkpoint_path,
                event_types=["质押"],
                encoder_feature_mode="evidence_lexical",
            )
            with self.assertRaisesRegex(ValueError, "feature mode"):
                PlannerGate(
                    checkpoint_path=checkpoint_path,
                    encoder=None,
                    feature_mode="global_only",
                    presence_threshold=0.5,
                    device=torch.device("cpu"),
                )

    def test_planner_gate_predict_uses_trained_biases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "r3_planner.pt"
            _save_planner_checkpoint(
                checkpoint_path,
                event_types=["质押", "收购"],
                hidden_size=4,
                k_clip=4,
                presence_bias=20.0,
                count_bias_log_rate=float(torch.log(torch.tensor(2.5))),
            )
            gate = PlannerGate(
                checkpoint_path=checkpoint_path,
                encoder=None,
                feature_mode="global_only",
                presence_threshold=0.5,
                device=torch.device("cpu"),
            )
            doc = DueeDocument(
                document_id="d1",
                text="无关字符串",
                title="无关",
                records=[],
            )

            present_known, count_known = gate.predict(doc, "质押")
            present_unknown, count_unknown = gate.predict(doc, "未登录事件类型")

            self.assertTrue(present_known)
            self.assertGreaterEqual(count_known, 1)
            self.assertFalse(present_unknown)
            self.assertEqual(count_unknown, 0)

    def test_smoke_run_with_planner_gate_overrides_type_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "schema.json").write_text(
                '[{"event_type":"质押","role_list":[{"role":"质押方"}]}]\n',
                encoding="utf-8",
            )
            (data_root / "train.jsonl").write_text(
                '{"id":"train1","title":"质押公告","text":"甲公司质押100万股。",'
                '"event_list":[{"event_type":"质押","arguments":[{"role":"质押方","argument":"甲公司"}]}]}\n',
                encoding="utf-8",
            )
            (data_root / "dev.jsonl").write_text(
                '{"id":"dev1","title":"质押公告","text":"甲公司质押100万股。",'
                '"event_list":[{"event_type":"质押","arguments":[{"role":"质押方","argument":"甲公司"}]}]}\n',
                encoding="utf-8",
            )
            checkpoint_path = root / "r3_planner.pt"
            _save_planner_checkpoint(
                checkpoint_path,
                event_types=["质押"],
                hidden_size=4,
                k_clip=2,
                presence_bias=15.0,
                count_bias_log_rate=0.0,
            )
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
                    "--seed",
                    "7",
                    "--max-epochs",
                    "2",
                    "--batch-size",
                    "2",
                    "--routes",
                    "baseline",
                    "--smoke",
                    "--planner-checkpoint",
                    str(checkpoint_path),
                    "--planner-feature-mode",
                    "global_only",
                ]
            )
            report = run_p5b(args)

            self.assertTrue((run_dir / "canonical" / "dev.baseline.pred.jsonl").exists())
            self.assertIn("baseline", report["routes"])


    def test_planner_gate_loads_sentence_mode_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "r3_planner.pt"
            event_types = ["质押", "收购"]
            # Build a sentence-mode checkpoint
            planner = RecordPlanner(
                hidden_size=4, num_event_types=len(event_types), k_max=3, count_head_mode="sentence",
            )
            with torch.no_grad():
                planner.type_gate.proj.weight.zero_()
                planner.type_gate.proj.bias.fill_(10.0)  # always present
                planner.count_planner.scorer[0].weight.zero_()
                planner.count_planner.scorer[0].bias.zero_()
                planner.count_planner.scorer[2].weight.zero_()
                planner.count_planner.scorer[2].bias.fill_(10.0)  # always predict count=1
                planner.count_planner.type_embedding.weight.zero_()
            payload = {
                "planner": planner.state_dict(),
                "planner_metadata": {
                    "event_types": event_types,
                    "hidden_size": 4,
                    "k_clip": 3,
                    "count_head_mode": "sentence",
                    "presence_threshold_all_dev": 0.5,
                    "encoder_feature_mode": "global_only",
                    "max_sentences": 256,
                },
            }
            torch.save(payload, checkpoint_path)

            gate = PlannerGate(
                checkpoint_path=checkpoint_path,
                encoder=None,
                feature_mode="global_only",
                presence_threshold=None,
                device=torch.device("cpu"),
            )
            self.assertEqual(gate.count_head_mode, "sentence")

            # predict on a toy document (no encoder → zeros repr = constant)
            doc = DueeDocument(
                document_id="d1",
                title="质押",
                text="公司进行了质押。",
                records=[],
            )
            present, count = gate.predict(doc, "质押")
            # With presence_bias=10, always present; with sentence_scorer_bias=10,
            # sigmoid≈1.0 for every sentence → expected count clamped to ≥1
            self.assertTrue(present)
            self.assertGreaterEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
