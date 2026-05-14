import tempfile
import unittest
from pathlib import Path

import torch

from carve.p3_planner_only_runner import build_arg_parser, run_r3_planner_only


class R3PlannerOnlyRunnerTests(unittest.TestCase):
    def test_smoke_run_writes_artifacts_and_dual_population_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = self._write_toy_dataset(root / "data")
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
                    "--max-epochs",
                    "2",
                    "--batch-size",
                    "2",
                    "--smoke",
                    "--encoder-feature-mode",
                    "evidence_lexical",
                ]
            )
            report = run_r3_planner_only(args)

            self.assertEqual(report["status"], "r3_planner_only_smoke")
            self.assertEqual(report["acceptance_population"], ["multi_event_dev", "all_dev"])
            self.assertEqual(report["encoder_feature_mode"], "evidence_lexical")
            self.assertEqual(report["train_population"]["name"], "all_train")
            self.assertEqual(report["dev_populations"]["multi_event_dev"]["documents"], 1)
            self.assertEqual(report["dev_populations"]["all_dev"]["documents"], 2)
            self.assertEqual(report["diagnostic_populations"], [])
            self.assertTrue((run_dir / "diagnostics" / "r3_planner_train_history.json").exists())
            self.assertTrue((run_dir / "diagnostics" / "r3_planner_metrics.json").exists())
            self.assertTrue((run_dir / "diagnostics" / "r3_planner_baselines.json").exists())
            self.assertTrue((run_dir / "checkpoints" / "r3_planner.pt").exists())

    def test_baselines_are_reported_for_multi_and_all_dev(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = self._write_toy_dataset(root / "data")
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
                    "--max-epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--smoke",
                ]
            )
            report = run_r3_planner_only(args)

            baselines = report["baselines"]
            self.assertAlmostEqual(baselines["multi_event_dev"]["predict_one"]["count_mae_positive"], 0.5)
            self.assertAlmostEqual(baselines["all_dev"]["predict_one"]["count_mae_positive"], 1 / 3, places=6)
            self.assertEqual(
                baselines["multi_event_dev"]["p5b_lexical_trigger"]["type_gate_recall"],
                1.0,
            )
            self.assertEqual(
                baselines["multi_event_dev"]["p5b_lexical_trigger"]["count_mae_positive"],
                0.5,
            )
            self.assertEqual(baselines["multi_event_dev"]["legacy_single_softmax"]["diagnostic_only"], True)

    def test_dual_population_acceptance_checks_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = self._write_toy_dataset(root / "data")
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
                    "--max-epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--smoke",
                ]
            )
            report = run_r3_planner_only(args)
            checks = report["acceptance_checks"]

            expected_keys = {
                "multi_event_dev/type_gate_auc",
                "multi_event_dev/type_gate_f1_youden",
                "multi_event_dev/count_mae_positive",
                "all_dev/type_gate_auc",
                "all_dev/type_gate_f1_youden",
                "all_dev/count_mae_positive",
                "training/presence_loss_trend",
                "training/count_loss_trend",
            }
            self.assertTrue(expected_keys.issubset(set(checks.keys())))
            for key in (
                "multi_event_dev/type_gate_auc",
                "all_dev/type_gate_auc",
                "multi_event_dev/count_mae_positive",
                "all_dev/count_mae_positive",
            ):
                entry = checks[key]
                self.assertIn("best_baseline", entry)
                self.assertIn("baseline_relative_threshold", entry)
                self.assertIn("passed_absolute", entry)
                self.assertIn("passed_baseline_relative", entry)

    def test_acceptance_gate_fails_when_lexical_dominates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = self._write_toy_dataset(root / "data")
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
                    "--max-epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--smoke",
                    "--encoder-feature-mode",
                    "global_only",
                ]
            )
            report = run_r3_planner_only(args)
            self.assertFalse(report["accepted"])
            self.assertEqual(report["encoder_feature_mode"], "global_only")

    def test_train_population_flag_switches_cache_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = self._write_toy_dataset(root / "data")
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
                    "--max-epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--smoke",
                    "--train-population",
                    "multi_event_train",
                ]
            )
            report = run_r3_planner_only(args)
            self.assertEqual(report["train_population"]["name"], "multi_event_train")

    def test_smoke_run_sentence_mode_writes_artifacts_and_noise_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = self._write_toy_dataset(root / "data")
            run_dir = root / "run"

            args = build_arg_parser().parse_args(
                [
                    "--dataset", "DuEE-Fin-dev500",
                    "--data-root", str(data_root),
                    "--schema", str(data_root / "schema.json"),
                    "--run-dir", str(run_dir),
                    "--model-path", "__toy__",
                    "--max-epochs", "2",
                    "--batch-size", "2",
                    "--smoke",
                    "--encoder-feature-mode", "evidence_lexical",
                    "--count-head-mode", "sentence",
                ]
            )
            report = run_r3_planner_only(args)

            self.assertEqual(report["status"], "r3_planner_only_smoke")
            self.assertEqual(report["count_head_mode"], "sentence")
            # Noise diagnostics must be present for sentence mode
            noise = report["noise_diagnostics"]
            self.assertIn("train", noise)
            self.assertIn("multi_event_dev", noise)
            self.assertIn("all_dev", noise)
            for key in ("train", "multi_event_dev", "all_dev"):
                self.assertIn("gold_record_sentence_recall", noise[key])
                self.assertIn("mean_sentence_label_count_over_gold", noise[key])
            # Checkpoint must record count_head_mode
            checkpoint_path = run_dir / "checkpoints" / "r3_planner.pt"
            self.assertTrue(checkpoint_path.exists())
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(
                ckpt.get("planner_metadata", {}).get("count_head_mode"), "sentence",
            )
            # Acceptance checks must include sentence_mode keys
            checks = report["acceptance_checks"]
            self.assertIn("multi_event_dev/sentence_score_auc", checks)
            self.assertIn("all_dev/sentence_score_auc", checks)
            self.assertIn("training/sentence_count_loss_trend", checks)

    def test_acceptance_count_uses_baseline_relative_only_in_sentence_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = self._write_toy_dataset(root / "data")
            run_dir = root / "run"

            args = build_arg_parser().parse_args(
                [
                    "--dataset", "DuEE-Fin-dev500",
                    "--data-root", str(data_root),
                    "--schema", str(data_root / "schema.json"),
                    "--run-dir", str(run_dir),
                    "--model-path", "__toy__",
                    "--max-epochs", "1",
                    "--batch-size", "4",
                    "--smoke",
                    "--count-head-mode", "sentence",
                ]
            )
            report = run_r3_planner_only(args)
            checks = report["acceptance_checks"]

            # In sentence mode, count_mae_positive absolute_threshold should reflect
            # the dynamic baseline-relative gate (predict_one - margin), not hardcoded 0.5
            for pop in ("multi_event_dev", "all_dev"):
                count_entry = checks[f"{pop}/count_mae_positive"]
                self.assertIn("best_baseline", count_entry)
                self.assertIn("baseline_relative_threshold", count_entry)
                self.assertIn("passed_baseline_relative", count_entry)
                # The absolute threshold should match the baseline-relative threshold
                # (they converge in sentence mode)
                self.assertEqual(
                    count_entry["absolute_threshold"],
                    count_entry["baseline_relative_threshold"],
                )

    @staticmethod
    def _write_toy_dataset(data_root: Path) -> Path:
        data_root.mkdir()
        (data_root / "schema.json").write_text(
            '[{"event_type":"质押","role_list":[{"role":"质押方"}]},'
            '{"event_type":"收购","role_list":[{"role":"收购方"}]}]\n',
            encoding="utf-8",
        )
        train_rows = [
            (
                '{"id":"train1","title":"质押 收购","text":"甲公司质押。乙公司收购。",'
                '"event_list":['
                '{"event_type":"质押","arguments":[{"role":"质押方","argument":"甲公司"}]},'
                '{"event_type":"质押","arguments":[{"role":"质押方","argument":"丙公司"}]},'
                '{"event_type":"收购","arguments":[{"role":"收购方","argument":"乙公司"}]}'
                "]}\n"
            ),
            (
                '{"id":"train2","title":"质押","text":"丁公司质押。丙公司质押。",'
                '"event_list":['
                '{"event_type":"质押","arguments":[{"role":"质押方","argument":"丁公司"}]},'
                '{"event_type":"质押","arguments":[{"role":"质押方","argument":"丙公司"}]}'
                "]}\n"
            ),
        ]
        dev_rows = [
            (
                '{"id":"dev_multi","title":"质押 收购","text":"甲公司质押。乙公司收购。",'
                '"event_list":['
                '{"event_type":"质押","arguments":[{"role":"质押方","argument":"甲公司"}]},'
                '{"event_type":"质押","arguments":[{"role":"质押方","argument":"丙公司"}]},'
                '{"event_type":"收购","arguments":[{"role":"收购方","argument":"乙公司"}]}'
                "]}\n"
            ),
            (
                '{"id":"dev_single","title":"收购","text":"乙公司收购。",'
                '"event_list":['
                '{"event_type":"收购","arguments":[{"role":"收购方","argument":"乙公司"}]}'
                "]}\n"
            ),
        ]
        (data_root / "train.jsonl").write_text("".join(train_rows), encoding="utf-8")
        (data_root / "dev.jsonl").write_text("".join(dev_rows), encoding="utf-8")
        return data_root


if __name__ == "__main__":
    unittest.main()
