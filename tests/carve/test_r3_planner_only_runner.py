import tempfile
import unittest
from pathlib import Path

from carve.p3_planner_only_runner import build_arg_parser, run_r3_planner_only


class R3PlannerOnlyRunnerTests(unittest.TestCase):
    def test_smoke_run_writes_artifacts_and_separate_populations(self) -> None:
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
                ]
            )
            report = run_r3_planner_only(args)

            self.assertEqual(report["status"], "r3_planner_only_smoke")
            self.assertEqual(report["acceptance_population"], "multi_event_dev")
            self.assertEqual(report["train_population"]["name"], "multi_event_train")
            self.assertEqual(report["train_population"]["documents"], 2)
            self.assertEqual(report["dev_populations"]["multi_event_dev"]["documents"], 1)
            self.assertEqual(report["dev_populations"]["all_dev"]["documents"], 2)
            self.assertIn("all_dev", report["diagnostic_populations"])
            self.assertNotIn("all_dev", report["acceptance_checks"])
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
