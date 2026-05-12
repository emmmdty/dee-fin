import tempfile
import unittest
from pathlib import Path

from carve.datasets import DueeDocument, build_candidate_lexicon, generate_inference_candidates
from carve.p5b_runner import build_arg_parser, run_p5b
from evaluator.canonical.types import CanonicalEventRecord


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


if __name__ == "__main__":
    unittest.main()
