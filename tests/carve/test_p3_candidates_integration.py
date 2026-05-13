import tempfile
import unittest
from pathlib import Path

from carve.allocation import CandidateMention
from carve.candidates import generate_candidates
from carve.datasets import DueeDocument, build_candidate_lexicon
from carve.p3_runner import build_arg_parser, run_p3
from evaluator.canonical.types import CanonicalEventRecord


class DummyMentionExtractor:
    def extract(self, document: DueeDocument, event_type: str, role: str) -> list[CandidateMention]:
        return [
            CandidateMention(
                event_type=event_type,
                role=role,
                value="甲公司",
                start=0,
                end=3,
                source="dummy_crf",
                raw_span="甲公司",
            )
        ]


class P3CandidatesIntegrationTests(unittest.TestCase):
    def test_generate_candidates_oracle_inference(self) -> None:
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

        candidates = generate_candidates(
            document,
            evidence_logits=None,
            mention_extractor=None,
            event_type="质押",
            role="质押方",
            oracle_inject=False,
            records=document.records,
        )

        self.assertEqual(candidates, [])

    def test_generate_candidates_lexicon_fallback(self) -> None:
        document = DueeDocument(
            document_id="doc1",
            title="公告",
            text="甲公司发布公告。",
            records=[],
        )
        lexicon = build_candidate_lexicon(
            [
                DueeDocument(
                    document_id="train",
                    title="公告",
                    text="甲公司发布公告。",
                    records=[
                        CanonicalEventRecord(
                            document_id="train",
                            event_type="质押",
                            arguments={"质押方": ["甲公司"]},
                        )
                    ],
                )
            ]
        )

        candidates = generate_candidates(
            document,
            evidence_logits=None,
            mention_extractor=None,
            event_type="质押",
            role="质押方",
            lexicon=lexicon,
        )

        self.assertEqual([candidate.value for candidate in candidates], ["甲公司"])
        self.assertEqual(candidates[0].source, "train_lexicon_text_match")

    def test_generate_candidates_normalizes_crf_spans(self) -> None:
        document = DueeDocument(document_id="doc1", title="", text="甲公司发布公告。", records=[])

        candidates = generate_candidates(
            document,
            evidence_logits=None,
            mention_extractor=DummyMentionExtractor(),
            event_type="质押",
            role="质押方",
        )

        self.assertEqual(candidates[0].value, "甲公司")
        self.assertEqual(candidates[0].raw_span, "甲公司")

    def test_smoke_run_p3(self) -> None:
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
            report = run_p3(args)

            self.assertEqual(report["status"], "p3_smoke")
            self.assertTrue((run_dir / "diagnostics" / "p3_mention_metrics.json").exists())
            self.assertTrue((run_dir / "diagnostics" / "p3_planner_metrics.json").exists())
            self.assertTrue((run_dir / "checkpoints" / "p3.pt").exists())


if __name__ == "__main__":
    unittest.main()
