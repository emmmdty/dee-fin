import json
import tempfile
import unittest
from pathlib import Path

from scripts.data_split import split_utils


class SplitUtilsTests(unittest.TestCase):
    def test_load_records_supports_json_array_object_list_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            array_path = root / "array.json"
            object_path = root / "object.json"
            jsonl_path = root / "rows.jsonl"

            array_path.write_text(json.dumps([{"id": "a"}, {"id": "b"}]), encoding="utf-8")
            object_path.write_text(json.dumps({"data": [{"id": "c"}]}), encoding="utf-8")
            jsonl_path.write_text('{"id": "d"}\n{"id": "e"}\n', encoding="utf-8")

            array_records, array_format, array_warnings = split_utils.load_records(array_path)
            object_records, object_format, object_warnings = split_utils.load_records(object_path)
            jsonl_records, jsonl_format, jsonl_warnings = split_utils.load_records(jsonl_path)

        self.assertEqual([r.obj["id"] for r in array_records], ["a", "b"])
        self.assertEqual(array_format, "json_array")
        self.assertEqual(array_warnings, [])
        self.assertEqual([r.obj["id"] for r in object_records], ["c"])
        self.assertEqual(object_format, "json_object_list_key:data")
        self.assertEqual(object_warnings, [])
        self.assertEqual([r.obj["id"] for r in jsonl_records], ["d", "e"])
        self.assertEqual(jsonl_format, "jsonl")
        self.assertEqual(jsonl_warnings, [])

    def test_load_records_iterates_jsonl_by_physical_lines_not_unicode_splitlines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unicode_separator.jsonl"
            rows = [
                {"id": "a", "text": "contains\u2028unicode line separator"},
                {"id": "b", "text": "normal"},
            ]
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            records, fmt, warnings = split_utils.load_records(path)

        self.assertEqual(fmt, "jsonl")
        self.assertEqual(warnings, [])
        self.assertEqual([record.obj["id"] for record in records], ["a", "b"])
        self.assertIn("\u2028", records[0].raw_text)

    def test_document_uid_prefers_existing_ids_and_records_fallback_sources(self):
        duee = {"id": "doc-1", "title": "t", "text": "body"}
        docfee_missing_id = {"content": "same content", "events": []}
        empty_text = {"events": []}

        duee_uid = split_utils.derive_doc_uid(duee, "DuEE-Fin", 7)
        fallback_uid = split_utils.derive_doc_uid(docfee_missing_id, "DocFEE", 3)
        ambiguous_uid = split_utils.derive_doc_uid(empty_text, "DocFEE", 9)

        self.assertEqual(duee_uid.uid, "doc-1")
        self.assertEqual(duee_uid.source, "id")
        self.assertTrue(fallback_uid.uid.startswith("sha256:"))
        self.assertEqual(fallback_uid.source, "text_hash")
        self.assertTrue(ambiguous_uid.uid.startswith("line_index_text_hash:9:"))
        self.assertEqual(ambiguous_uid.source, "line_index_text_hash")

    def test_event_signature_extracts_dataset_specific_event_types(self):
        duee = {
            "event_list": [
                {"event_type": "质押", "arguments": []},
                {"event_type": "解除质押", "arguments": []},
                {"event_type": "质押", "arguments": []},
            ]
        }
        docfee = {
            "events": [
                {"event_type": "股东减持", "减持金额": "100"},
                {"event_type": "股权质押", "质押方": "A"},
            ]
        }
        chfinann = [
            "doc-1",
            {
                "recguid_eventname_eventdict_list": [
                    [0, "EquityPledge", {"Pledger": "A"}],
                    [1, "EquityFreeze", {"EquityHolder": "B"}],
                ]
            },
        ]

        self.assertEqual(split_utils.event_signature(duee, "DuEE-Fin"), "解除质押|质押")
        self.assertEqual(split_utils.event_signature(docfee, "DocFEE"), "股东减持|股权质押")
        self.assertEqual(split_utils.event_signature(chfinann, "ChFinAnn"), "EquityFreeze|EquityPledge")
        self.assertEqual(split_utils.event_signature({"event_list": []}, "DuEE-Fin"), "NO_EVENT")

    def test_stratified_hash_split_is_exact_deterministic_and_preserves_output_order(self):
        records = []
        for index in range(30):
            event_type = "A" if index < 20 else "B"
            records.append(
                split_utils.LoadedRecord(
                    index=index,
                    obj={"id": f"doc-{index:02d}", "text": f"text {index}", "event_list": [{"event_type": event_type}]},
                    raw_text=None,
                )
            )

        first, first_diag = split_utils.stratified_hash_dev_indices(records, "DuEE-Fin", dev_size=6, seed=42)
        second, second_diag = split_utils.stratified_hash_dev_indices(records, "DuEE-Fin", dev_size=6, seed=42)
        different_seed, _ = split_utils.stratified_hash_dev_indices(records, "DuEE-Fin", dev_size=6, seed=43)

        self.assertEqual(first, second)
        self.assertEqual(first_diag["dev_size"], 6)
        self.assertEqual(len(first), 6)
        self.assertNotEqual(first, different_seed)
        self.assertEqual(first_diag["strata"]["A"]["selected"], 4)
        self.assertEqual(first_diag["strata"]["B"]["selected"], 2)

        selected = split_utils.select_records_preserving_order(records, first)
        self.assertEqual([r.index for r in selected], sorted(first))

    def test_duplicate_diagnostics_reports_without_removing_records(self):
        train = [
            split_utils.LoadedRecord(0, {"id": "a", "text": "same", "event_list": []}, None),
            split_utils.LoadedRecord(1, {"id": "b", "text": "same", "event_list": []}, None),
        ]
        dev = [split_utils.LoadedRecord(0, {"id": "c", "text": "same", "event_list": []}, None)]
        test = [split_utils.LoadedRecord(0, {"id": "d", "text": "different", "event_list": []}, None)]

        diagnostics = split_utils.duplicate_diagnostics({"train": train, "dev": dev, "test": test}, "DuEE-Fin")

        self.assertEqual(diagnostics["within_split_exact_text_duplicate_count"]["train"], 1)
        self.assertEqual(diagnostics["within_split_exact_text_duplicate_count"]["dev"], 0)
        self.assertEqual(diagnostics["cross_split_exact_text_duplicate_count"]["train__dev"], 1)
        self.assertEqual(len(train), 2)


if __name__ == "__main__":
    unittest.main()
