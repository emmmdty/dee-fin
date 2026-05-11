import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluator.canonical.loaders import adapt_document
from evaluator.canonical.schema import EventSchema
from evaluator.canonical.types import CanonicalDocument, CanonicalEventRecord
from evaluator.docfee_official.metric import evaluate_docfee_official
from evaluator.legacy_doc2edag.metric import evaluate_legacy_doc2edag, evaluate_legacy_doc2edag_native_table
from evaluator.legacy_doc2edag.native_table import parse_native_event_table
from evaluator.unified_strict.metric import evaluate_unified_strict


def doc(document_id, *records):
    return CanonicalDocument(document_id=document_id, records=list(records))


def record(document_id, event_type, arguments):
    return CanonicalEventRecord(
        document_id=document_id,
        event_type=event_type,
        arguments={role: [value] if isinstance(value, str) else value for role, value in arguments.items()},
    )


def native_table_payload(
    *,
    event_types=("E",),
    event_type_fields=None,
    gold=None,
    pred=None,
):
    fields = event_type_fields or {event_type: ["A", "B"] for event_type in event_types}
    gold_matrix = gold if gold is not None else [[] for _ in event_types]
    pred_matrix = pred if pred is not None else [[] for _ in event_types]
    return {
        "format": "procnet_native_event_table_v1",
        "dataset": "toy",
        "seed": 44,
        "split": "test",
        "event_types": list(event_types),
        "event_type_fields": fields,
        "documents": [
            {
                "document_id": "doc1",
                "gold": gold_matrix,
                "pred": pred_matrix,
            }
        ],
    }


def evaluate_native_table_payload(payload):
    return evaluate_legacy_doc2edag_native_table(parse_native_event_table(payload))


class UnifiedStrictMetricTests(unittest.TestCase):
    def test_exact_match_gives_perfect_scores(self):
        gold = [doc("d1", record("d1", "E", {"A": "x", "B": "y"}))]
        pred = [doc("d1", record("d1", "E", {"A": "x", "B": "y"}))]

        report = evaluate_unified_strict(gold, pred, dataset="toy")

        self.assertEqual(report["overall"]["tp"], 2)
        self.assertEqual(report["overall"]["fp"], 0)
        self.assertEqual(report["overall"]["fn"], 0)
        self.assertEqual(report["overall"]["precision"], 1.0)
        self.assertEqual(report["overall"]["recall"], 1.0)
        self.assertEqual(report["overall"]["f1"], 1.0)

    def test_nfkc_normalization_matches_full_width_and_half_width(self):
        gold = [doc("d1", record("d1", "E", {"Company": "ＡＢＣ有限公司"}))]
        pred = [doc("d1", record("d1", "E", {"Company": "ABC有限公司"}))]

        report = evaluate_unified_strict(gold, pred, dataset="toy")

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 0)
        self.assertEqual(report["overall"]["fn"], 0)
        self.assertEqual(report["overall"]["f1"], 1.0)

    def test_empty_values_are_excluded_from_scoring(self):
        gold = [doc("d1", record("d1", "E", {"A": "x", "Empty": ["", "   ", None]}))]
        pred = [doc("d1", record("d1", "E", {"A": "x", "Empty": ["", "   ", None], "Extra": ""}))]

        report = evaluate_unified_strict(gold, pred, dataset="toy")

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 0)
        self.assertEqual(report["overall"]["fn"], 0)

    def test_duplicate_role_values_do_not_inflate_tp(self):
        gold = [doc("d1", record("d1", "E", {"公司": ["甲公司"]}))]
        pred = [doc("d1", record("d1", "E", {"公司": ["甲公司", "甲公司"]}))]

        report = evaluate_unified_strict(gold, pred, dataset="toy")

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 0)
        self.assertEqual(report["overall"]["fn"], 0)

    def test_record_id_and_event_id_are_not_scored(self):
        gold = [
            adapt_document(
                {
                    "document_id": "d1",
                    "events": [
                        {
                            "event_type": "E",
                            "record_id": "gold-record",
                            "event_id": "gold-event",
                            "arguments": {"A": ["x"]},
                        }
                    ],
                },
                dataset="toy",
            )
        ]
        pred = [
            adapt_document(
                {
                    "document_id": "d1",
                    "events": [
                        {
                            "event_type": "E",
                            "record_id": "pred-record",
                            "event_id": "pred-event",
                            "arguments": {"A": ["x"]},
                        }
                    ],
                },
                dataset="toy",
            )
        ]

        report = evaluate_unified_strict(gold, pred, dataset="toy")

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 0)
        self.assertEqual(report["overall"]["fn"], 0)
        self.assertEqual(report["overall"]["precision"], 1.0)
        self.assertEqual(report["overall"]["recall"], 1.0)
        self.assertEqual(report["overall"]["f1"], 1.0)

    def test_missing_role_creates_fn(self):
        gold = [doc("d1", record("d1", "E", {"A": "x", "B": "y"}))]
        pred = [doc("d1", record("d1", "E", {"A": "x"}))]

        report = evaluate_unified_strict(gold, pred, dataset="toy")

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 0)
        self.assertEqual(report["overall"]["fn"], 1)

    def test_extra_role_creates_fp(self):
        gold = [doc("d1", record("d1", "E", {"A": "x"}))]
        pred = [doc("d1", record("d1", "E", {"A": "x", "B": "extra"}))]

        report = evaluate_unified_strict(gold, pred, dataset="toy")

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 1)
        self.assertEqual(report["overall"]["fn"], 0)

    def test_wrong_event_type_does_not_match_correct_event_type(self):
        gold = [doc("d1", record("d1", "Correct", {"A": "x"}))]
        pred = [doc("d1", record("d1", "Wrong", {"A": "x"}))]

        report = evaluate_unified_strict(gold, pred, dataset="toy")

        self.assertEqual(report["overall"]["tp"], 0)
        self.assertEqual(report["overall"]["fp"], 1)
        self.assertEqual(report["overall"]["fn"], 1)

    def test_duplicate_predicted_record_creates_fp(self):
        gold = [doc("d1", record("d1", "E", {"A": "x"}))]
        pred = [
            doc(
                "d1",
                record("d1", "E", {"A": "x"}),
                record("d1", "E", {"A": "x"}),
            )
        ]

        report = evaluate_unified_strict(gold, pred, dataset="toy")

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 1)
        self.assertEqual(report["overall"]["fn"], 0)

    def test_multi_record_same_event_type_uses_global_record_alignment(self):
        gold = [
            doc(
                "d1",
                record("d1", "E", {"Name": "A", "Amount": "1"}),
                record("d1", "E", {"Name": "B", "Amount": "2"}),
            )
        ]
        pred = [
            doc(
                "d1",
                record("d1", "E", {"Name": "B", "Amount": "2"}),
                record("d1", "E", {"Name": "A", "Amount": "1"}),
            )
        ]

        report = evaluate_unified_strict(gold, pred, dataset="toy")

        self.assertEqual(report["overall"]["tp"], 4)
        self.assertEqual(report["overall"]["fp"], 0)
        self.assertEqual(report["overall"]["fn"], 0)

    def test_unified_strict_matching_is_event_type_constrained(self):
        gold = [doc("d1", record("d1", "A", {"Role": "same"}))]
        pred = [doc("d1", record("d1", "B", {"Role": "same"}))]

        report = evaluate_unified_strict(gold, pred, dataset="toy")

        self.assertEqual(report["matching_policy"]["event_type_constrained"], True)
        self.assertEqual(report["overall"]["tp"], 0)

    def test_invalid_schema_role_is_diagnosed_and_not_repaired(self):
        schema = EventSchema.from_mapping({"E": ["Allowed"]})
        gold = [doc("d1", record("d1", "E", {"Allowed": "x"}))]
        pred = [doc("d1", record("d1", "E", {"Allowed": "x", "BadRole": "x"}))]

        report = evaluate_unified_strict(gold, pred, dataset="toy", schema=schema)

        self.assertEqual(report["diagnostics"]["invalid_role_count"], 1)
        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 1)

    def test_single_event_and_multi_event_subset_metrics_are_computed(self):
        gold = [
            doc("single", record("single", "E", {"A": "x"})),
            doc("multi", record("multi", "E", {"A": "x"}), record("multi", "E", {"A": "y"})),
        ]
        pred = [
            doc("single", record("single", "E", {"A": "x"})),
            doc("multi", record("multi", "E", {"A": "x"})),
        ]

        report = evaluate_unified_strict(gold, pred, dataset="toy")

        self.assertEqual(report["subset_metrics"]["single_event"]["f1"], 1.0)
        self.assertEqual(report["subset_metrics"]["multi_event"]["tp"], 1)
        self.assertEqual(report["subset_metrics"]["multi_event"]["fn"], 1)


class LegacyDoc2EDAGMetricTests(unittest.TestCase):
    def test_fixed_slot_exact_match_gives_perfect_scores(self):
        schema = EventSchema.from_mapping({"E": ["A", "B"]})
        gold = [doc("d1", record("d1", "E", {"A": "x", "B": "y"}))]
        pred = [doc("d1", record("d1", "E", {"A": "x", "B": "y"}))]

        report = evaluate_legacy_doc2edag(gold, pred, dataset="toy", schema=schema)

        self.assertEqual(report["metric_family"], "legacy_doc2edag_native_fixed_slot")
        self.assertEqual(report["overall"]["tp"], 2)
        self.assertEqual(report["overall"]["fp"], 0)
        self.assertEqual(report["overall"]["fn"], 0)
        self.assertEqual(report["overall"]["precision"], 1.0)
        self.assertEqual(report["overall"]["recall"], 1.0)
        self.assertEqual(report["overall"]["f1"], 1.0)

    def test_fixed_slot_missing_role_creates_fn(self):
        schema = EventSchema.from_mapping({"E": ["A", "B"]})
        gold = [doc("d1", record("d1", "E", {"A": "x", "B": "y"}))]
        pred = [doc("d1", record("d1", "E", {"A": "x"}))]

        report = evaluate_legacy_doc2edag(gold, pred, dataset="toy", schema=schema)

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 0)
        self.assertEqual(report["overall"]["fn"], 1)

    def test_fixed_slot_extra_role_creates_fp(self):
        schema = EventSchema.from_mapping({"E": ["A", "B"]})
        gold = [doc("d1", record("d1", "E", {"A": "x"}))]
        pred = [doc("d1", record("d1", "E", {"A": "x", "B": "extra"}))]

        report = evaluate_legacy_doc2edag(gold, pred, dataset="toy", schema=schema)

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 1)
        self.assertEqual(report["overall"]["fn"], 0)

    def test_fixed_slot_wrong_non_empty_value_creates_fp_and_fn(self):
        schema = EventSchema.from_mapping({"E": ["A", "B"]})
        gold = [doc("d1", record("d1", "E", {"A": "x", "B": "y"}))]
        pred = [doc("d1", record("d1", "E", {"A": "x", "B": "z"}))]

        report = evaluate_legacy_doc2edag(gold, pred, dataset="toy", schema=schema)

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 1)
        self.assertEqual(report["overall"]["fn"], 1)

    def test_none_slots_help_matching_but_do_not_count_as_metric_units(self):
        schema = EventSchema.from_mapping({"E": ["A", "B", "C", "D"]})
        gold = [
            doc(
                "d1",
                record("d1", "E", {"A": "y"}),
                record("d1", "E", {"A": "x", "B": "b"}),
            )
        ]
        pred = [doc("d1", record("d1", "E", {"A": "x"}))]

        report = evaluate_legacy_doc2edag(gold, pred, dataset="toy", schema=schema)

        self.assertEqual(report["overall"]["tp"], 0)
        self.assertEqual(report["overall"]["fp"], 1)
        self.assertEqual(report["overall"]["fn"], 3)

    def test_duplicate_values_under_one_role_do_not_inflate_denominator(self):
        schema = EventSchema.from_mapping({"E": ["A"]})
        gold = [doc("d1", record("d1", "E", {"A": ["x"]}))]
        pred = [doc("d1", record("d1", "E", {"A": ["x", "x"]}))]

        report = evaluate_legacy_doc2edag(gold, pred, dataset="toy", schema=schema)

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 0)
        self.assertEqual(report["overall"]["fn"], 0)
        self.assertEqual(report["diagnostics"]["fixed_slot_pred_unit_count"], 1)

    def test_multi_value_role_collapses_with_last_value_wins(self):
        schema = EventSchema.from_mapping({"E": ["A"]})
        gold = [doc("d1", record("d1", "E", {"A": ["x", "y"]}))]
        pred = [doc("d1", record("d1", "E", {"A": "y"}))]

        report = evaluate_legacy_doc2edag(gold, pred, dataset="toy", schema=schema)

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 0)
        self.assertEqual(report["overall"]["fn"], 0)
        self.assertEqual(report["diagnostics"]["fixed_slot_gold_unit_count"], 1)
        self.assertEqual(report["diagnostics"]["collapsed_multi_value_role_count"], 1)

    def test_multi_value_denominator_differs_from_unified_strict(self):
        schema = EventSchema.from_mapping({"E": ["A"]})
        gold = [doc("d1", record("d1", "E", {"A": ["x", "y"]}))]
        pred = [doc("d1", record("d1", "E", {"A": "y"}))]

        legacy = evaluate_legacy_doc2edag(gold, pred, dataset="toy", schema=schema)
        unified = evaluate_unified_strict(gold, pred, dataset="toy", schema=schema)

        self.assertEqual(legacy["overall"]["tp"] + legacy["overall"]["fn"], 1)
        self.assertEqual(unified["overall"]["tp"] + unified["overall"]["fn"], 2)

    def test_event_id_and_record_id_are_not_scored(self):
        schema = EventSchema.from_mapping({"E": ["A"]})
        gold = [
            adapt_document(
                {
                    "document_id": "d1",
                    "events": [
                        {
                            "event_type": "E",
                            "record_id": "gold-record",
                            "event_id": "gold-event",
                            "arguments": {"A": ["x"]},
                        }
                    ],
                },
                dataset="toy",
            )
        ]
        pred = [
            adapt_document(
                {
                    "document_id": "d1",
                    "events": [
                        {
                            "event_type": "E",
                            "record_id": "pred-record",
                            "event_id": "pred-event",
                            "arguments": {"A": ["x"]},
                        }
                    ],
                },
                dataset="toy",
            )
        ]

        report = evaluate_legacy_doc2edag(gold, pred, dataset="toy", schema=schema)

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 0)
        self.assertEqual(report["overall"]["fn"], 0)

    def test_wrong_event_type_does_not_match(self):
        schema = EventSchema.from_mapping({"Gold": ["A"], "Pred": ["A"]})
        gold = [doc("d1", record("d1", "Gold", {"A": "x"}))]
        pred = [doc("d1", record("d1", "Pred", {"A": "x"}))]

        report = evaluate_legacy_doc2edag(gold, pred, dataset="toy", schema=schema)

        self.assertEqual(report["overall"]["tp"], 0)
        self.assertEqual(report["overall"]["fp"], 1)
        self.assertEqual(report["overall"]["fn"], 1)

    def test_doc2edag_style_greedy_matching_is_deterministic(self):
        schema = EventSchema.from_mapping({"E": ["A", "B"]})
        gold = [
            doc(
                "d1",
                record("d1", "E", {"A": "a1", "B": "b1"}),
                record("d1", "E", {"A": "a2", "B": "b2"}),
            )
        ]
        pred = [
            doc(
                "d1",
                record("d1", "E", {"A": "a1", "B": "wrong", "C": "extra"}),
                record("d1", "E", {"A": "a2", "B": "b2"}),
            )
        ]

        first = evaluate_legacy_doc2edag(gold, pred, dataset="toy", schema=schema)
        second = evaluate_legacy_doc2edag(gold, pred, dataset="toy", schema=schema)

        self.assertEqual(first["overall"], second["overall"])
        self.assertEqual(first["overall"]["tp"], 3)
        self.assertEqual(first["overall"]["fp"], 1)
        self.assertEqual(first["overall"]["fn"], 1)

    def test_fixed_slot_policy_is_reported(self):
        schema = EventSchema.from_mapping({"E": ["A"]})
        gold = [doc("d1", record("d1", "E", {"A": "x"}))]
        pred = [doc("d1", record("d1", "E", {"A": "x"}))]

        report = evaluate_legacy_doc2edag(gold, pred, dataset="toy", schema=schema)

        self.assertEqual(report["matching_policy"]["unit"], "fixed_schema_role_slot")
        self.assertEqual(report["slot_policy"]["multi_value_collapse"], "last_normalized_non_empty_value_wins")

    def test_native_event_table_exact_match_gives_perfect_scores(self):
        report = evaluate_native_table_payload(
            native_table_payload(gold=[[["x", "y"]]], pred=[[["x", "y"]]])
        )

        self.assertEqual(report["metric_family"], "legacy_doc2edag_native_fixed_slot")
        self.assertEqual(report["input_format"], "native-event-table")
        self.assertEqual(report["dataset"], "toy")
        self.assertEqual(report["split"], "test")
        self.assertEqual(report["seed"], 44)
        self.assertEqual(report["overall"]["tp"], 2)
        self.assertEqual(report["overall"]["fp"], 0)
        self.assertEqual(report["overall"]["fn"], 0)
        self.assertEqual(report["event_type_fields"], {"E": ["A", "B"]})
        self.assertEqual(report["document_count"], 1)
        self.assertEqual(report["event_record_counts"]["gold"]["E"], 1)
        self.assertEqual(report["event_record_counts"]["pred"]["E"], 1)

    def test_native_event_table_wrong_value_creates_fp_and_fn(self):
        report = evaluate_native_table_payload(
            native_table_payload(gold=[[["x", "y"]]], pred=[[["x", "z"]]])
        )

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 1)
        self.assertEqual(report["overall"]["fn"], 1)

    def test_native_event_table_none_slots_help_matching_but_do_not_count(self):
        payload = native_table_payload(
            event_type_fields={"E": ["A", "B", "C", "D"]},
            gold=[[["y", None, None, None], ["x", "b", None, None]]],
            pred=[[["x", None, None, None]]],
        )

        report = evaluate_native_table_payload(payload)

        self.assertEqual(report["overall"]["tp"], 0)
        self.assertEqual(report["overall"]["fp"], 1)
        self.assertEqual(report["overall"]["fn"], 3)

    def test_native_event_table_unmatched_prediction_adds_fp_per_non_empty_slot(self):
        report = evaluate_native_table_payload(
            native_table_payload(gold=[[["x", "y"]]], pred=[[["x", "y"], ["extra", "value"]]])
        )

        self.assertEqual(report["overall"]["tp"], 2)
        self.assertEqual(report["overall"]["fp"], 2)
        self.assertEqual(report["overall"]["fn"], 0)

    def test_native_event_table_unmatched_gold_adds_fn_per_non_empty_slot(self):
        report = evaluate_native_table_payload(
            native_table_payload(gold=[[["x", None], ["g1", "g2"]]], pred=[[["x", None]]])
        )

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 0)
        self.assertEqual(report["overall"]["fn"], 2)

    def test_native_event_table_wrong_event_type_index_cannot_cross_match(self):
        report = evaluate_native_table_payload(
            native_table_payload(
                event_types=("Gold", "Pred"),
                event_type_fields={"Gold": ["A"], "Pred": ["A"]},
                gold=[[["x"]], []],
                pred=[[], [["x"]]],
            )
        )

        self.assertEqual(report["overall"]["tp"], 0)
        self.assertEqual(report["overall"]["fp"], 1)
        self.assertEqual(report["overall"]["fn"], 1)


class DocFeeMetricTests(unittest.TestCase):
    def test_docfee_style_role_value_matching_is_deterministic(self):
        gold = [
            doc(
                "d1",
                record("d1", "股东减持", {"减持的股东": "张三", "减持金额": "100"}),
                record("d1", "股东减持", {"减持的股东": "李四", "减持金额": "200"}),
            )
        ]
        pred = [
            doc(
                "d1",
                record("d1", "股东减持", {"减持的股东": "李四", "减持金额": "200"}),
                record("d1", "股东减持", {"减持的股东": "张三", "减持金额": "wrong"}),
            )
        ]

        report = evaluate_docfee_official(gold, pred, dataset="DocFEE-dev1000")

        self.assertEqual(report["overall"]["tp"], 3)
        self.assertEqual(report["overall"]["fp"], 1)
        self.assertEqual(report["overall"]["fn"], 1)
        self.assertEqual(report["per_event"]["股东减持"]["tp"], 3)

    def test_docfee_official_ignores_prediction_only_documents(self):
        gold = [doc("d1", record("d1", "股东减持", {"减持的股东": "张三"}))]
        pred = [
            doc("d1", record("d1", "股东减持", {"减持的股东": "张三"})),
            doc("extra", record("extra", "股东减持", {"减持的股东": "李四"})),
        ]

        report = evaluate_docfee_official(gold, pred, dataset="DocFEE-dev1000")

        self.assertEqual(report["overall"]["tp"], 1)
        self.assertEqual(report["overall"]["fp"], 0)
        self.assertEqual(report["overall"]["fn"], 0)
        self.assertEqual(report["overall"]["f1"], 1.0)


class EvaluatorCliTests(unittest.TestCase):
    def test_cli_help_works(self):
        result = subprocess.run(
            [sys.executable, "-m", "evaluator", "--help"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("legacy-doc2edag", result.stdout)
        self.assertIn("docfee-official", result.stdout)
        self.assertIn("unified-strict", result.stdout)

    def test_legacy_doc2edag_cli_help_describes_native_fixed_slot_metric(self):
        result = subprocess.run(
            [sys.executable, "-m", "evaluator", "legacy-doc2edag", "--help"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Native Doc2EDAG/ProcNet-compatible fixed-slot evaluator", result.stdout)
        self.assertIn("--input-format", result.stdout)
        self.assertIn("native-event-table", result.stdout)

    def test_legacy_doc2edag_cli_default_canonical_jsonl_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold_path = root / "gold.jsonl"
            pred_path = root / "pred.jsonl"
            schema_path = root / "schema.json"
            out_path = root / "report.json"
            row = {"document_id": "d1", "events": [{"event_type": "E", "arguments": {"A": ["x"], "B": ["y"]}}]}
            gold_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            pred_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            schema_path.write_text(json.dumps({"E": ["A", "B"]}, ensure_ascii=False), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evaluator",
                    "legacy-doc2edag",
                    "--dataset",
                    "toy",
                    "--gold",
                    str(gold_path),
                    "--pred",
                    str(pred_path),
                    "--schema",
                    str(schema_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["input_format"], "canonical-jsonl")
            self.assertEqual(report["overall"]["f1"], 1.0)

    def test_legacy_doc2edag_cli_native_event_table_writes_report(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            native_path = root / "native_event_table.json"
            out_path = root / "report.json"
            payload = native_table_payload(gold=[[["x", "y"]]], pred=[[["x", "z"]]])
            native_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evaluator",
                    "legacy-doc2edag",
                    "--input-format",
                    "native-event-table",
                    "--native-table",
                    str(native_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["input_format"], "native-event-table")
            self.assertEqual(report["overall"]["tp"], 1)
            self.assertEqual(report["overall"]["fp"], 1)
            self.assertEqual(report["overall"]["fn"], 1)

    def test_legacy_doc2edag_cli_native_event_table_rejects_canonical_arguments(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            native_path = root / "native_event_table.json"
            native_path.write_text(json.dumps(native_table_payload(), ensure_ascii=False), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evaluator",
                    "legacy-doc2edag",
                    "--input-format",
                    "native-event-table",
                    "--native-table",
                    str(native_path),
                    "--gold",
                    "gold.jsonl",
                ],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not compatible with --input-format native-event-table", result.stderr)

    def test_cli_unified_strict_writes_json_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold_path = root / "gold.jsonl"
            pred_path = root / "pred.jsonl"
            out_path = root / "report.json"
            row = {"document_id": "d1", "events": [{"event_type": "E", "arguments": {"A": ["x"]}}]}
            gold_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            pred_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evaluator",
                    "unified-strict",
                    "--dataset",
                    "toy",
                    "--gold",
                    str(gold_path),
                    "--pred",
                    str(pred_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["overall"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
