"""news -> SARGE-input exporters (the closed-loop entry point).

Astock rows live in quote-aware TSVs (embedded newlines) and carry the stock
code as a de-zero-padded int; CMIN-CN raw news is one tokenized headline per
line under news/preprocessed/<stock>/<date>. The exporters normalise both into
the one JSONL contract SARGE inference and `sarge_to_event_nodes.py --meta`
share: {"doc_id", "text", "date", "stock", ...}.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_preprocess_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "preprocess_datasets.py"
    spec = importlib.util.spec_from_file_location("preprocess_datasets", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preprocess = _load_preprocess_module()

_ASTOCK_HEADER = "CODE\tDATE\tlabel\ttext_a\tNAME\tTITLE\tDESCRIPTION"


def _fake_astock(processed: Path) -> None:
    astock = processed / "astock"
    astock.mkdir(parents=True, exist_ok=True)
    quoted_text = '"中泰化学披露三季报，\n净利润亏损4.46亿元。"'
    quoted_row = f"2092\t2020-10-27 20:02:00\t0\t{quoted_text}\t中泰化学\t三季报\t亏损扩大\n"
    plain_row = "600030\t2021-03-05 09:00:00\t1\t\t中信证券\t年报发布\t业绩增长稳健\n"
    (astock / "train.tsv").write_text(
        _ASTOCK_HEADER + "\n" + quoted_row + plain_row, encoding="utf-8"
    )
    (astock / "val.tsv").write_text(
        _ASTOCK_HEADER + "\n300750\tbad-date\t1\t宁德时代发布公告。\t宁德时代\t公告\t无\n",
        encoding="utf-8",
    )


def test_export_astock_sarge_input(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(preprocess, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(preprocess, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(preprocess, "PROCESSED_ROOT", tmp_path / "processed")
    _fake_astock(tmp_path / "processed")

    counts = preprocess.export_astock_sarge_input(splits=("train", "val"))

    out = tmp_path / "processed" / "astock" / "sarge_input.jsonl"
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    # the bad-date val row is dropped; the two train rows survive
    assert counts["train"] == 2 and counts["val"] == 0
    assert len(records) == 2
    first, second = records
    # quote-aware TSV read keeps the embedded newline inside one record's text
    assert "净利润亏损" in first["text"]
    # stock codes re-gain their leading zeros; dates truncate to ISO days
    assert first["stock"] == "002092" and first["date"] == "2020-10-27"
    assert first["doc_id"].startswith("astock-train-")
    assert first["label"] == "0"
    # empty text_a falls back to TITLE + DESCRIPTION
    assert second["stock"] == "600030"
    assert "年报发布" in second["text"] and "业绩增长稳健" in second["text"]


def test_export_cmin_cn_sarge_input_detokenizes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(preprocess, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(preprocess, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(preprocess, "PROCESSED_ROOT", tmp_path / "processed")
    news = tmp_path / "raw" / "cmin_cn" / "CMIN-CN" / "news" / "preprocessed"
    (news / "600030" / "2021-03-05").parent.mkdir(parents=True, exist_ok=True)
    (news / "600030" / "2021-03-05").write_text(
        "中信 证券 发布 年报\n业绩 稳健 增长\n", encoding="utf-8"
    )
    (news / "600030" / "not-a-date").write_text("忽略 我\n", encoding="utf-8")

    counts = preprocess.export_cmin_cn_sarge_input()

    out = tmp_path / "processed" / "cmin_cn" / "sarge_input.jsonl"
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert counts["docs"] == 1 and len(records) == 1
    record = records[0]
    assert record["doc_id"] == "cmin-600030-2021-03-05"
    assert record["stock"] == "600030" and record["date"] == "2021-03-05"
    # tokenizer spaces between CJK characters are removed; headlines joined
    assert "中信证券发布年报" in record["text"]
    assert "业绩稳健增长" in record["text"]


def test_export_cmin_cn_skips_when_raw_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(preprocess, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(preprocess, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(preprocess, "PROCESSED_ROOT", tmp_path / "processed")
    assert preprocess.export_cmin_cn_sarge_input() == {}
