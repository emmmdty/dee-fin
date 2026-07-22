import csv
import importlib.util
import json
from pathlib import Path

import yaml


def _load_preprocess_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "preprocess_datasets.py"
    spec = importlib.util.spec_from_file_location("preprocess_datasets", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preprocess = _load_preprocess_module()


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_split_ccks_preserves_unlabeled_eval_and_creates_labeled_local_test(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(preprocess, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(preprocess, "PROCESSED_ROOT", tmp_path / "processed")
    raw = tmp_path / "raw" / "ccks_fin_causal"
    processed = tmp_path / "processed" / "ccks_fin_causal"

    labeled_rows = [
        {"text_id": str(i), "text": f"text {i}", "causal_pairs": [{"cause": {}, "effect": {}}]}
        for i in range(5)
    ]
    eval_rows = [{"text_id": "eval-1", "text": "eval text", "causal_pairs": []}]
    _write_jsonl(raw / "train.jsonl", labeled_rows)
    _write_jsonl(raw / "eval.jsonl", eval_rows)

    preprocess.split_ccks(val_n=1, test_n=2, seed=0)

    train_rows = (processed / "train.jsonl").read_text(encoding="utf-8").splitlines()
    val_rows = (processed / "val.jsonl").read_text(encoding="utf-8").splitlines()
    test_rows = [
        json.loads(line)
        for line in (processed / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_all_rows = (processed / "train_all.jsonl").read_text(encoding="utf-8").splitlines()
    eval_unlabeled_rows = [
        json.loads(line)
        for line in (processed / "eval_unlabeled.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))

    assert len(train_rows) == 2
    assert len(val_rows) == 1
    assert len(test_rows) == 2
    assert len(train_all_rows) == 5
    assert all(row["causal_pairs"] for row in test_rows)
    assert eval_unlabeled_rows == eval_rows
    assert manifest["splits"]["train"]["records"] == 2
    assert (
        manifest["official_split_note"]
        == "7000 labeled train + 1000 unlabeled eval; no public labeled test."
    )


def test_prepare_tkg_splits_writes_split_combined_and_smoke_files(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(preprocess, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(preprocess, "PROCESSED_ROOT", tmp_path / "processed")
    raw = tmp_path / "raw" / "icews14"
    raw.mkdir(parents=True)
    (raw / "train.txt").write_text("a\tr\tb\t1\nc\tr\td\t2\textra\n", encoding="utf-8")
    (raw / "valid.txt").write_text("e\tr\tf\t3\n", encoding="utf-8")
    (raw / "test.txt").write_text("g\tr\th\t4\n", encoding="utf-8")

    counts = preprocess.prepare_tkg_splits("icews14", smoke_n=1, tiny_n=1)

    canon = tmp_path / "processed" / "icews14"
    assert counts == {"train": 2, "valid": 1, "test": 1}
    assert (canon / "train.tsv").read_text(encoding="utf-8").splitlines() == [
        "a\tr\tb\t1",
        "c\tr\td\t2",
    ]
    assert len((canon / "icews14.tsv").read_text(encoding="utf-8").splitlines()) == 4
    assert (canon / "icews14_smoke.tsv").read_text(encoding="utf-8").splitlines() == [
        "a\tr\tb\t1"
    ]
    assert (canon / "icews14_tiny.tsv").read_text(encoding="utf-8").splitlines() == [
        "a\tr\tb\t1"
    ]
    manifest = json.loads((canon / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"]["combined"]["records"] == 4


def test_prepare_astock_preserves_official_splits_and_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(preprocess, "PROCESSED_ROOT", tmp_path / "processed")
    raw = tmp_path / "raw" / "astock" / "Astock-main" / "data"
    raw.mkdir(parents=True)
    header = ["CODE", "DATE", "label", "text_a"]
    for split in ("train", "val", "test", "ood"):
        with (raw / f"{split}.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(header)
            writer.writerow(["000001", "2021-01-01", "1", "news"])

    counts = preprocess.prepare_astock()

    processed = tmp_path / "processed" / "astock"
    assert counts == {"train": 1, "val": 1, "test": 1, "ood": 1}
    assert (processed / "train.tsv").exists()
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"]["ood"]["records"] == 1
    assert manifest["schema"]["train"] == header


def test_prepare_cmin_cn_writes_inventory_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(preprocess, "PROCESSED_ROOT", tmp_path / "processed")
    root = tmp_path / "raw" / "cmin_cn" / "CMIN-CN"
    price = root / "price" / "preprocessed"
    news = root / "news" / "preprocessed" / "平安银行"
    price.mkdir(parents=True)
    news.mkdir(parents=True)
    (price / "平安银行.txt").write_text("2018-01-03\t0.1\t1\t2\t3\t4\t5\n", encoding="utf-8")
    (news / "2018-01-03").write_text(
        '{"text":["新","闻"],"created_at":"2018-01-03"}\n',
        encoding="utf-8",
    )

    manifest = preprocess.prepare_cmin_cn()

    processed = tmp_path / "processed" / "cmin_cn"
    assert manifest["price_files"] == 1
    assert manifest["news_stock_dirs"] == 1
    assert manifest["news_date_files"] == 1
    assert (processed / "manifest.json").exists()


def test_prepare_event_graph_zh_manifest_uses_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(preprocess, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(preprocess, "RAW_ROOT", tmp_path / "data" / "raw")
    monkeypatch.setattr(preprocess, "PROCESSED_ROOT", tmp_path / "data" / "processed")
    raw = tmp_path / "data" / "raw" / "event_graph_zh"
    raw.mkdir(parents=True)
    (raw / "event_graph.json").write_text('{"nodes": {}, "edges": []}', encoding="utf-8")

    manifest = preprocess.prepare_event_graph_zh()

    assert manifest is not None
    assert manifest["source"] == "data/raw/event_graph_zh/event_graph.json"
    assert manifest["processed_dir"] == "data/processed/event_graph_zh"


def test_config_data_paths_use_processed_layout():
    repo_root = Path(__file__).resolve().parents[2]
    bad_paths = []
    for path in sorted((repo_root / "configs").rglob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, value in item.items():
                    if key == "path" and isinstance(value, str) and value.startswith("data/"):
                        if "/canonical/" in value or not value.startswith("data/processed/"):
                            bad_paths.append((str(path.relative_to(repo_root)), value))
                    else:
                        stack.append(value)
            elif isinstance(item, list):
                stack.extend(item)

    assert bad_paths == []
