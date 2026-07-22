"""run_sarge_news_inference stages exported news JSONL into SARGE's staged
dataset layout (schema + slot-plan train rows borrowed from a source dataset,
news docs as the predict split) without touching SARGE code. The staging half
is pure CPU and covered here via --dry-run; the vLLM half only runs on the
server. Staging is idempotent: a pre-staged source dir is reused as-is, which
is also what makes this test independent of the real SARGE data download.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _load_script():
    path = _REPO / "scripts" / "run_sarge_news_inference.py"
    spec = importlib.util.spec_from_file_location("run_sarge_news_inference", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rsni = _load_script()


def _run_main(argv: list[str]) -> int:
    old_argv = sys.argv
    sys.argv = argv
    try:
        return rsni.main()
    finally:
        sys.argv = old_argv


def _prestage_source(staging: Path, dataset: str = "DuEE-Fin-dev500") -> None:
    src = staging / dataset
    src.mkdir(parents=True, exist_ok=True)
    schema = {"dataset": dataset, "properties": {"股份回购": {"properties": {"回购方": {}}}}}
    (src / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (src / "train.jsonl").write_text(
        json.dumps({"doc_id": "duee-train-0", "content": "训练文档", "split": "train"}) + "\n",
        encoding="utf-8",
    )


def _news_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "sarge_input.jsonl"
    rows = [
        {"doc_id": "astock-train-000000", "text": "中泰化学披露三季报。", "date": "2020-10-27",
         "stock": "002092", "split": "train", "label": "0"},
        {"doc_id": "astock-val-000001", "text": "", "date": "2021-03-05",
         "stock": "600030", "split": "val", "label": "1"},
        {"doc_id": "astock-test-000002", "text": "宁德时代发布公告。", "date": "2021-06-01",
         "stock": "300750", "split": "test", "label": "1"},
    ]
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def test_dry_run_stages_news_dataset(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    _prestage_source(staging)
    news = _news_jsonl(tmp_path)
    assert (
        _run_main(
            [
                "x",
                "--news-jsonl", str(news),
                "--staging-root", str(staging),
                "--dataset-name", "Astock-news",
                "--dry-run",
            ]
        )
        == 0
    )
    staged = staging / "Astock-news"
    # schema + slot-plan train rows are borrowed from the pre-staged source
    assert json.loads((staged / "schema.json").read_text(encoding="utf-8"))["dataset"]
    assert (staged / "train.jsonl").read_text(encoding="utf-8").strip()
    rows = [
        json.loads(line)
        for line in (staged / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # the empty-text row is dropped; doc ids survive for the --meta join later
    assert [r["doc_id"] for r in rows] == ["astock-train-000000", "astock-test-000002"]
    assert all(r["content"] and r["split"] == "test" for r in rows)
    # sidecar fields ride along in meta (date/stock/label for provenance)
    assert rows[0]["meta"]["date"] == "2020-10-27"
    assert rows[0]["meta"]["stock"] == "002092"


def test_dry_run_is_idempotent(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    _prestage_source(staging)
    news = _news_jsonl(tmp_path)
    argv = [
        "x",
        "--news-jsonl", str(news),
        "--staging-root", str(staging),
        "--dataset-name", "Astock-news",
        "--dry-run",
    ]
    assert _run_main(argv) == 0
    assert _run_main(argv) == 0  # re-run reuses the staged source without error
