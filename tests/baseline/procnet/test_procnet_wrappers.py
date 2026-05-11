from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "baseline" / "procnet"


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ProcNetWrapperTest(unittest.TestCase):
    def test_native_event_table_serializer_preserves_gold_pred_and_none_slots(self) -> None:
        from scripts.baseline.procnet.procnet_event_table_dump import write_native_event_table

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "native.json"
            write_native_event_table(
                out,
                dataset="TinySet",
                split="dev",
                seed=42,
                epoch=3,
                event_types=["EventA"],
                event_type_fields={"EventA": ["RoleA", "RoleB"]},
                documents=[
                    {
                        "document_id": "doc-1",
                        "gold": [[[None, "gold-value"]]],
                        "pred": [[["pred-value", None]]],
                    }
                ],
                metadata={"source": "unit-test"},
            )

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], "procnet_native_event_table_v1")
            self.assertIsNone(payload["documents"][0]["gold"][0][0][0])
            self.assertEqual(payload["documents"][0]["gold"][0][0][1], "gold-value")
            self.assertEqual(payload["documents"][0]["pred"][0][0][0], "pred-value")
            self.assertIsNone(payload["documents"][0]["pred"][0][0][1])

    def test_native_event_table_is_accepted_by_evaluator_if_available(self) -> None:
        from scripts.baseline.procnet.procnet_event_table_dump import write_native_event_table

        with tempfile.TemporaryDirectory() as tmpdir:
            native = Path(tmpdir) / "native.json"
            report = Path(tmpdir) / "report.json"
            write_native_event_table(
                native,
                dataset="TinySet",
                split="test",
                seed=42,
                epoch=1,
                event_types=["EventA"],
                event_type_fields={"EventA": ["RoleA"]},
                documents=[
                    {
                        "document_id": "doc-1",
                        "gold": [[["same"]]],
                        "pred": [[["same"]]],
                    }
                ],
                metadata={},
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evaluator",
                    "legacy-doc2edag",
                    "--input-format",
                    "native-event-table",
                    "--native-table",
                    str(native),
                    "--out",
                    str(report),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode != 0 and "No module named evaluator" in result.stderr:
                self.skipTest("evaluator package is not importable in this environment")
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(data["input_format"], "native-event-table")

    def test_canonical_exports_write_gold_and_pred_for_dev_and_test(self) -> None:
        from scripts.baseline.procnet.procnet_data_adapters import export_canonical_split

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "canonical"
            for split in ("dev", "test"):
                export_canonical_split(
                    output_root,
                    split=split,
                    gold_documents=[
                        {
                            "document_id": f"{split}-doc",
                            "events": [
                                {
                                    "event_type": "EventA",
                                    "arguments": {"RoleA": "gold"},
                                }
                            ],
                        }
                    ],
                    pred_documents=[
                        {
                            "document_id": f"{split}-doc",
                            "predictions": [
                                {
                                    "event_type": "EventA",
                                    "arguments": {"RoleA": "pred"},
                                }
                            ],
                        }
                    ],
                )

            for split in ("dev", "test"):
                gold = output_root / f"{split}.canonical.gold.jsonl"
                pred = output_root / f"{split}.canonical.pred.jsonl"
                self.assertTrue(gold.exists())
                self.assertTrue(pred.exists())
                self.assertIn('"events"', gold.read_text(encoding="utf-8"))
                self.assertIn('"predictions"', pred.read_text(encoding="utf-8"))

    def test_evaluator_commands_use_run_local_canonical_gold(self) -> None:
        from scripts.baseline.procnet.procnet_eval_runner import build_evaluator_commands

        run_dir = Path("runs/baseline/procnet/exp1")
        commands = build_evaluator_commands(
            project_root=Path("."),
            run_dir=run_dir,
            dataset="DuEE-Fin-dev500",
            schema_path=Path("runs/baseline/procnet/exp1/staged_data/schema.json"),
            splits=["dev", "test"],
            python_bin="python",
        )
        joined = "\n".join(" ".join(cmd.argv) for cmd in commands)

        self.assertIn("canonical/dev.canonical.gold.jsonl", joined)
        self.assertIn("canonical/test.canonical.gold.jsonl", joined)
        self.assertNotIn("data/processed/DuEE-Fin-dev500/dev.jsonl", joined)
        self.assertIn("native-event-table", joined)
        self.assertIn("canonical-jsonl", joined)
        self.assertIn("unified-strict", joined)

    def test_two_gpu_dry_run_uses_two_separate_single_gpu_commands(self) -> None:
        result = subprocess.run(
            ["bash", str(SCRIPT_ROOT / "run_two_gpu_seed42.sh"), "--dry-run"],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", result.stdout)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", result.stdout)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=0,1", result.stdout)
        self.assertIn("--dataset ChFinAnn-Doc2EDAG", result.stdout)
        self.assertIn("--dataset DuEE-Fin-dev500", result.stdout)

    def test_run_procnet_dry_run_does_not_create_run_directory_or_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "runs"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_ROOT / "run_procnet_repro.py"),
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--dataset",
                    "ChFinAnn-Doc2EDAG",
                    "--experiment-name",
                    "dryrun_unit",
                    "--seed",
                    "42",
                    "--max-epochs",
                    "100",
                    "--patience",
                    "8",
                    "--gpu",
                    "0",
                    "--output-root",
                    str(output_root),
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("DRY RUN", result.stdout)
            self.assertIn("--max-epochs 100", result.stdout)
            self.assertIn("--patience 8", result.stdout)
            self.assertFalse((output_root / "dryrun_unit").exists())

    def test_duee_adapter_requires_distinct_train_dev_test(self) -> None:
        from scripts.baseline.procnet.procnet_data_adapters import validate_duee_split_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            train.write_text("[]\n", encoding="utf-8")
            dev.write_text("[]\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                validate_duee_split_paths(train=train, dev=dev, test=dev)

    def test_seed_metadata_and_output_layout_summary_paths(self) -> None:
        from scripts.baseline.procnet.procnet_wrapper import build_run_config, expected_artifacts_for_split

        config = build_run_config(
            project_root=Path("."),
            dataset="ChFinAnn-Doc2EDAG",
            experiment_name="exp1",
            seed=42,
            max_epochs=100,
            patience=8,
            gpu="0",
            python_bin="python",
            output_root=Path("runs/baseline/procnet"),
            server_mode=False,
        )
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.max_epochs, 100)
        self.assertEqual(config.patience, 8)
        self.assertEqual(config.run_dir, Path("runs/baseline/procnet/exp1"))

        artifacts = expected_artifacts_for_split(config.run_dir, "test")
        self.assertEqual(
            artifacts["native_gold_path"],
            "native_event_tables/test.procnet_native_event_table_v1.json#documents[].gold",
        )
        self.assertEqual(artifacts["canonical_gold_path"], "canonical/test.canonical.gold.jsonl")

    def test_summary_lists_four_gold_pred_artifacts_for_dev_and_test(self) -> None:
        from scripts.baseline.procnet.collect_metrics import write_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "exp"
            write_summary(
                run_dir=run_dir,
                dataset="ChFinAnn-Doc2EDAG",
                seed=42,
                splits=["dev", "test"],
                early_stopping={
                    "best_epoch": None,
                    "best_dev_f1": 0.0,
                    "final_epoch": 0,
                    "stopped_by_early_stopping": False,
                    "patience": 8,
                    "max_epochs": 100,
                },
                warnings=[],
                evaluator_commands=[],
            )
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            for split in ("dev", "test"):
                artifacts = summary["artifacts"][split]
                self.assertEqual(
                    artifacts["native_gold_path"],
                    f"native_event_tables/{split}.procnet_native_event_table_v1.json#documents[].gold",
                )
                self.assertEqual(
                    artifacts["native_pred_path"],
                    f"native_event_tables/{split}.procnet_native_event_table_v1.json#documents[].pred",
                )
                self.assertEqual(artifacts["canonical_gold_path"], f"canonical/{split}.canonical.gold.jsonl")
                self.assertEqual(artifacts["canonical_pred_path"], f"canonical/{split}.canonical.pred.jsonl")

    def test_baseline_procnet_source_files_are_unmodified(self) -> None:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--", "baseline/procnet"],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
