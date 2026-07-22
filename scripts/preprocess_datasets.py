#!/usr/bin/env python
"""Preprocess raw datasets into the project-ready data/processed layout.

Raw inputs live under data/raw/{dataset}. Project code and configs should read
only data/processed/{dataset}.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"

RAW_ALIASES = {
    "maven_ere": ("maven_ere", "MAVEN_ERE"),
    "ccks_fin_causal": ("ccks_fin_causal", "tianchi_event_relaiton_train_eval"),
}


def _raw_dir(dataset: str) -> Path:
    for name in RAW_ALIASES.get(dataset, (dataset,)):
        path = RAW_ROOT / name
        if path.exists():
            return path
    return RAW_ROOT / dataset


def _processed_dir(dataset: str) -> Path:
    return PROCESSED_ROOT / dataset


def _project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _convert_ccks_record(record: dict[str, Any]) -> dict[str, Any]:
    pairs = []
    for item in record.get("result", []):
        pairs.append(
            {
                "cause": {
                    "type": item.get("reason_type", ""),
                    "region": item.get("reason_region", ""),
                    "product": item.get("reason_product", ""),
                    "industry": item.get("reason_industry", ""),
                },
                "effect": {
                    "type": item.get("result_type", ""),
                    "region": item.get("result_region", ""),
                    "product": item.get("result_product", ""),
                    "industry": item.get("result_industry", ""),
                },
                "cause_span": [],
                "effect_span": [],
            }
        )
    return {
        "text_id": str(record.get("text_id", "")),
        "text": record.get("text", ""),
        "causal_pairs": pairs,
    }


def _read_ccks_jsonl_or_tianchi(path: Path) -> list[str]:
    lines = []
    for line in _read_nonempty_lines(path):
        record = json.loads(line)
        if "causal_pairs" not in record:
            record = _convert_ccks_record(record)
        lines.append(json.dumps(record, ensure_ascii=False))
    return lines


def _find_ccks_file(raw: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = raw / name
        if path.exists():
            return path
    raise FileNotFoundError(f"none of {candidates} found under {raw}")


def split_ccks(val_n: int = 700, test_n: int = 700, seed: int = 42) -> dict[str, int]:
    raw = _raw_dir("ccks_fin_causal")
    out = _processed_dir("ccks_fin_causal")
    _reset_dir(out)

    train_src = _find_ccks_file(raw, ("train.jsonl", "ccks_task2_train.txt"))
    eval_src = _find_ccks_file(raw, ("eval.jsonl", "ccks_task2_eval_data.txt"))

    lines = _read_ccks_jsonl_or_tianchi(train_src)
    if val_n + test_n >= len(lines):
        raise ValueError(
            f"ccks split needs val_n + test_n < {len(lines)}, got {val_n + test_n}"
        )

    shuffled = list(lines)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    test_lines = shuffled[:test_n]
    val_lines = shuffled[test_n : test_n + val_n]
    train_lines = shuffled[test_n + val_n :]
    eval_lines = _read_ccks_jsonl_or_tianchi(eval_src)

    _write_lines(out / "train_all.jsonl", lines)
    _write_lines(out / "train.jsonl", train_lines)
    _write_lines(out / "train_split.jsonl", train_lines)
    _write_lines(out / "val.jsonl", val_lines)
    _write_lines(out / "test.jsonl", test_lines)
    _write_lines(out / "eval_unlabeled.jsonl", eval_lines)

    counts = {
        "train_all": len(lines),
        "train": len(train_lines),
        "val": len(val_lines),
        "test": len(test_lines),
        "eval_unlabeled": len(eval_lines),
    }
    _write_json(
        out / "manifest.json",
        {
            "dataset": "ccks_fin_causal",
            "task": "Chinese financial event causality extraction",
            "source": "CCKS-2021/Tianchi financial event causality task",
            "raw_dir": str(raw),
            "processed_dir": str(out),
            "official_split_note": (
                "7000 labeled train + 1000 unlabeled eval; no public labeled test."
            ),
            "local_split": {"seed": seed, "train": len(train_lines), "val": val_n, "test": test_n},
            "splits": {name: {"records": count} for name, count in counts.items()},
        },
    )
    print(
        f"[ccks] train={counts['train']} val={counts['val']} test={counts['test']} "
        f"train_all={counts['train_all']} eval_unlabeled={counts['eval_unlabeled']}"
    )
    return counts


def prepare_maven_ere(train_smoke_n: int = 30, valid_smoke_n: int = 20) -> dict[str, int]:
    raw = _raw_dir("maven_ere")
    out = _processed_dir("maven_ere")
    _reset_dir(out)

    for split in ("train", "valid", "test"):
        src = raw / f"{split}.jsonl"
        if not src.exists():
            raise FileNotFoundError(src)

    shutil.copyfile(raw / "train.jsonl", out / "train.jsonl")
    shutil.copyfile(raw / "valid.jsonl", out / "valid.jsonl")
    shutil.copyfile(raw / "test.jsonl", out / "test_unlabeled.jsonl")

    train_lines = _read_nonempty_lines(out / "train.jsonl")
    valid_lines = _read_nonempty_lines(out / "valid.jsonl")
    _write_lines(out / "train_smoke.jsonl", train_lines[:train_smoke_n])
    _write_lines(out / "valid_smoke.jsonl", valid_lines[:valid_smoke_n])

    counts = {
        "train": len(train_lines),
        "valid": len(valid_lines),
        "test_unlabeled": _count_lines(out / "test_unlabeled.jsonl"),
        "train_smoke": min(train_smoke_n, len(train_lines)),
        "valid_smoke": min(valid_smoke_n, len(valid_lines)),
    }
    _write_json(
        out / "manifest.json",
        {
            "dataset": "maven_ere",
            "task": "event coreference, temporal, causal, and subevent relation extraction",
            "source": "MAVEN-ERE, EMNLP 2022",
            "raw_dir": str(raw),
            "processed_dir": str(out),
            "official_split_note": "Official train/valid/test exists, but test labels are hidden.",
            "splits": {name: {"records": count} for name, count in counts.items()},
        },
    )
    print(
        f"[maven_ere] train={counts['train']} valid={counts['valid']} "
        f"test_unlabeled={counts['test_unlabeled']}"
    )
    return counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_maven_arg(train_smoke_n: int = 30, valid_smoke_n: int = 20) -> dict[str, int]:
    """MAVEN-Arg (event arguments), same docs as MAVEN-ERE. Official splits kept verbatim."""
    raw = _raw_dir("maven_arg")
    out = _processed_dir("maven_arg")
    _reset_dir(out)
    for split in ("train", "valid", "test"):
        if not (raw / f"{split}.jsonl").exists():
            raise FileNotFoundError(raw / f"{split}.jsonl")

    shutil.copyfile(raw / "train.jsonl", out / "train.jsonl")
    shutil.copyfile(raw / "valid.jsonl", out / "valid.jsonl")
    shutil.copyfile(raw / "test.jsonl", out / "test_unlabeled.jsonl")

    train_lines = _read_nonempty_lines(out / "train.jsonl")
    valid_lines = _read_nonempty_lines(out / "valid.jsonl")
    _write_lines(out / "train_smoke.jsonl", train_lines[:train_smoke_n])
    _write_lines(out / "valid_smoke.jsonl", valid_lines[:valid_smoke_n])

    counts = {
        "train": len(train_lines),
        "valid": len(valid_lines),
        "test_unlabeled": _count_lines(out / "test_unlabeled.jsonl"),
        "train_smoke": min(train_smoke_n, len(train_lines)),
        "valid_smoke": min(valid_smoke_n, len(valid_lines)),
    }
    _write_json(
        out / "manifest.json",
        {
            "dataset": "maven_arg",
            "task": "document-level event argument extraction (162 types / 612 roles)",
            "source": "MAVEN-Arg, ACL 2024 (arXiv 2311.09105)",
            "raw_dir": str(raw),
            "processed_dir": str(out),
            "official_split_note": "Official train/valid/test; test labels hidden (CodaLab).",
            "same_documents_as": "maven_ere / maven_fact (identical doc ids, verified)",
            "splits": {name: {"records": count} for name, count in counts.items()},
            "source_sha256": {
                f"{s}.jsonl": _sha256(raw / f"{s}.jsonl")
                for s in ("train", "valid", "test")
            },
        },
    )
    print(
        f"[maven_arg] train={counts['train']} valid={counts['valid']} "
        f"test_unlabeled={counts['test_unlabeled']}"
    )
    return counts


def prepare_maven_fact(train_smoke_n: int = 30, valid_smoke_n: int = 20) -> dict[str, int]:
    """MAVEN-FACT (event factuality + evidence), same 4480 docs. Only train/valid are public."""
    raw = _raw_dir("maven_fact")
    out = _processed_dir("maven_fact")
    _reset_dir(out)
    for split in ("train", "valid"):
        if not (raw / f"{split}.jsonl").exists():
            raise FileNotFoundError(raw / f"{split}.jsonl")

    shutil.copyfile(raw / "train.jsonl", out / "train.jsonl")
    shutil.copyfile(raw / "valid.jsonl", out / "valid.jsonl")

    train_lines = _read_nonempty_lines(out / "train.jsonl")
    valid_lines = _read_nonempty_lines(out / "valid.jsonl")
    _write_lines(out / "train_smoke.jsonl", train_lines[:train_smoke_n])
    _write_lines(out / "valid_smoke.jsonl", valid_lines[:valid_smoke_n])

    counts = {
        "train": len(train_lines),
        "valid": len(valid_lines),
        "train_smoke": min(train_smoke_n, len(train_lines)),
        "valid_smoke": min(valid_smoke_n, len(valid_lines)),
    }
    _write_json(
        out / "manifest.json",
        {
            "dataset": "maven_fact",
            "task": "event factuality detection (CT+/CT-/PS+/PS-/Uu) + supporting evidence",
            "source": "MAVEN-FACT, Findings of EMNLP 2024 (arXiv 2407.15352)",
            "raw_dir": str(raw),
            "processed_dir": str(out),
            "official_split_note": "Only train/valid public; official test hidden.",
            "same_documents_as": "maven_ere / maven_arg (identical doc ids, verified)",
            "bundled_fields": "docs also bundle args + causal/subevent/temporal relations",
            "splits": {name: {"records": count} for name, count in counts.items()},
            "source_sha256": {
                f"{s}.jsonl": _sha256(raw / f"{s}.jsonl") for s in ("train", "valid")
            },
        },
    )
    print(f"[maven_fact] train={counts['train']} valid={counts['valid']} (no public test)")
    return counts


def prepare_modafact(smoke_n: int = 20) -> dict[str, Any]:
    """ModaFact (Italian event factuality + modality, 5-fold CV).

    Keep the fine-grained seq2seq variant (fg/s2s_gen); FG CERTAINTY x POLARITY axes
    map to MAVEN-FACT's 5-class scheme in the loader (mapping recorded in manifest).
    """
    raw = _raw_dir("modafact")
    out = _processed_dir("modafact")
    variant = raw / "fg" / "s2s_gen"
    if not variant.exists():
        raise FileNotFoundError(variant)
    _reset_dir(out)

    folds = sorted(p.name for p in variant.iterdir() if p.is_dir())
    counts: dict[str, Any] = {}
    shas: dict[str, str] = {}
    for fold in folds:
        counts[fold] = {}
        dst_fold = out / "fg_s2s_gen" / fold
        dst_fold.mkdir(parents=True, exist_ok=True)
        for split in ("train", "dev", "test"):
            src = variant / fold / f"{split}_set.jsonl"
            if not src.exists():
                continue
            dst = dst_fold / f"{split}.jsonl"
            shutil.copyfile(src, dst)
            counts[fold][split] = _count_lines(dst)
            shas[f"{fold}/{split}.jsonl"] = _sha256(src)

    ref = variant / "fold_84" / "test_set.jsonl"
    if ref.exists():
        _write_lines(out / "smoke.jsonl", _read_nonempty_lines(ref)[:smoke_n])

    _write_json(
        out / "manifest.json",
        {
            "dataset": "modafact",
            "task": "Italian event factuality + modality (Ch3 cross-lingual robustness)",
            "source": "ModaFact, COLING 2025; HuggingFace dhfbk/modafact-ita",
            "license": "CC-BY-SA-4.0",
            "language": "it",
            "raw_dir": str(raw),
            "processed_dir": str(out),
            "variant": "fg/s2s_gen (fine-grained; jsonl Input + Output 'token=labels')",
            "cv": "5-fold (21/42/63/84/105), 60/20/20 stratified; all folds kept",
            "fg_axes": {
                "CERTAINTY": ["CERTAIN", "PROBABLE", "POSSIBLE", "UNDERSPECIFIED"],
                "POLARITY": ["POSITIVE", "NEGATIVE", "UNDERSPECIFIED"],
                "TIME": ["PRESENT/PAST", "FUTURE", "UNDERSPECIFIED"],
            },
            "maven_fact_5class_map": {
                "CERTAIN+POSITIVE": "CT+",
                "CERTAIN+NEGATIVE": "CT-",
                "PROBABLE|POSSIBLE + POSITIVE": "PS+",
                "PROBABLE|POSSIBLE + NEGATIVE": "PS-",
                "UNDERSPECIFIED": "Uu",
            },
            "splits": counts,
            "source_sha256": shas,
        },
    )
    print(f"[modafact] {len(folds)} folds (fg/s2s_gen) -> {out}")
    return counts


def prepare_it_happened(version: str = "07092017") -> dict[str, int]:
    """UDS It-Happened (UDS-IH2) event factuality on UD-EWT. Split by the 'Split' column.

    Rows are per-annotator; the loader aggregates to per-predicate signed factuality.
    """
    raw = _raw_dir("it_happened")
    out = _processed_dir("it_happened")
    src = raw / f"it-happened_eng_ud1.2_{version}.tsv"
    if not src.exists():
        raise FileNotFoundError(src)
    _reset_dir(out)

    rows = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    header = rows[0]
    split_idx = header.split("\t").index("Split")
    buckets: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for row in rows[1:]:
        sp = row.split("\t")[split_idx]
        if sp in buckets:
            buckets[sp].append(row)

    name_map = {"train": "train", "dev": "valid", "test": "test"}
    counts: dict[str, int] = {}
    for src_name, out_name in name_map.items():
        _write_lines(out / f"{out_name}.tsv", [header, *buckets[src_name]])
        counts[out_name] = len(buckets[src_name])

    _write_json(
        out / "manifest.json",
        {
            "dataset": "it_happened",
            "task": "event factuality (UDS-IH2 signed [-3,3]); Ch3 English 2nd (2018 benchmark)",
            "source": "UDS It-Happened v2 (Rudinger NAACL 2018); UD English Web Treebank 1.2",
            "version": version,
            "raw_dir": str(raw),
            "processed_dir": str(out),
            "note": "per-annotator rows; loader aggregates per-predicate signed factuality (mean)",
            "columns": header.split("\t"),
            "splits": {name: {"annotation_rows": c} for name, c in counts.items()},
            "source_sha256": {src.name: _sha256(src)},
        },
    )
    print(f"[it_happened] train={counts['train']} valid={counts['valid']} test={counts['test']}")
    return counts


def _count_json_list(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        return len(json.load(fh))


def prepare_docee() -> dict[str, Any]:
    """DocEE document-level event extraction (en normal/cross-domain + zh). Ch1 generalization."""
    raw = _raw_dir("docee")
    out = _processed_dir("docee")
    _reset_dir(out)
    counts: dict[str, Any] = {}
    shas: dict[str, str] = {}

    name_map = {"train": "train", "dev": "valid", "test": "test"}
    en_norm = raw / "docee-en" / "normal_setting"
    if en_norm.exists():
        (out / "en").mkdir(parents=True, exist_ok=True)
        counts["en_normal"] = {}
        for src_name, out_name in name_map.items():
            src = en_norm / f"{src_name}.json"
            if src.exists():
                dst = out / "en" / f"{out_name}.json"
                shutil.copyfile(src, dst)
                counts["en_normal"][out_name] = _count_json_list(dst)
                shas[f"en/{out_name}.json"] = _sha256(src)

    en_cd = raw / "docee-en" / "cross_domain_setting"
    if en_cd.exists():
        (out / "en_cross_domain").mkdir(parents=True, exist_ok=True)
        counts["en_cross_domain"] = {}
        for src in sorted(en_cd.glob("*.json")):
            dst = out / "en_cross_domain" / src.name
            shutil.copyfile(src, dst)
            counts["en_cross_domain"][src.stem] = _count_json_list(dst)

    zh_files = sorted((raw / "docee-zh").glob("*.json"))
    if zh_files:
        (out / "zh").mkdir(parents=True, exist_ok=True)
        dst = out / "zh" / "docee_zh.json"
        shutil.copyfile(zh_files[0], dst)
        counts["zh"] = _count_json_list(dst)
        shas["zh/docee_zh.json"] = _sha256(zh_files[0])

    _write_json(
        out / "manifest.json",
        {
            "dataset": "docee",
            "task": "document-level event extraction (classification + args); Ch1 generalization",
            "source": "DocEE, NAACL 2022; github tongmeihan1995/DocEE",
            "raw_dir": str(raw),
            "processed_dir": str(out),
            "languages": ["en", "zh"],
            "record_format": "each record is a 4-element list; see Event_Schema.md / README",
            "note": "en normal_setting = standard split; full en json + zh xlsx kept in raw only",
            "splits": counts,
            "source_sha256": shas,
        },
    )
    print(f"[docee] en_normal={counts.get('en_normal')} zh={counts.get('zh')}")
    return counts


def _read_quadruples(path: Path) -> list[str]:
    lines = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) >= 4:
            lines.append("\t".join(parts[:4]))
    return lines


def prepare_tkg_splits(name: str, smoke_n: int = 1000, tiny_n: int = 200) -> dict[str, int]:
    raw = _raw_dir(name)
    out = _processed_dir(name)
    _reset_dir(out)
    split_lines = {}

    for split in ("train", "valid", "test"):
        src = raw / f"{split}.txt"
        if not src.exists():
            raise FileNotFoundError(src)
        lines = _read_quadruples(src)
        split_lines[split] = lines
        _write_lines(out / f"{split}.tsv", lines)

    combined = split_lines["train"] + split_lines["valid"] + split_lines["test"]
    _write_lines(out / f"{name}.tsv", combined)
    _write_lines(out / f"{name}_smoke.tsv", split_lines["train"][:smoke_n])
    _write_lines(out / f"{name}_tiny.tsv", split_lines["train"][:tiny_n])

    counts = {split: len(lines) for split, lines in split_lines.items()}
    counts["combined"] = len(combined)
    _write_json(
        out / "manifest.json",
        {
            "dataset": name,
            "task": "temporal knowledge graph forecasting",
            "source": "FinDKG official repo" if name == "findkg" else "ICEWS14 TiRGN/TLogic split",
            "raw_dir": str(raw),
            "processed_dir": str(out),
            "format": "subject<TAB>relation<TAB>object<TAB>timestamp",
            "splits": {split: {"records": count} for split, count in counts.items()},
            "smoke": {"records": min(smoke_n, counts["train"])},
            "tiny": {"records": min(tiny_n, counts["train"])},
        },
    )
    print(
        f"[{name}] train={counts['train']} valid={counts['valid']} "
        f"test={counts['test']} combined={counts['combined']}"
    )
    return {split: counts[split] for split in ("train", "valid", "test")}


def prepare_astock() -> dict[str, int]:
    raw = _raw_dir("astock") / "Astock-main" / "data"
    out = _processed_dir("astock")
    _reset_dir(out)
    counts: dict[str, int] = {}
    schema: dict[str, list[str]] = {}

    # Astock text fields contain embedded newlines/tabs and are CSV-quoted, so
    # records must be counted with a quote-aware reader, not by physical lines.
    csv.field_size_limit(10**7)
    for split in ("train", "val", "test", "ood"):
        src = raw / f"{split}.csv"
        if not src.exists():
            raise FileNotFoundError(src)
        dst = out / f"{split}.tsv"
        shutil.copyfile(src, dst)
        with src.open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            header = next(reader)
            counts[split] = sum(1 for _ in reader)
        schema[split] = header

    _write_json(
        out / "manifest.json",
        {
            "dataset": "astock",
            "task": "stock-specific news movement/trading prediction",
            "source": "Astock, FinNLP 2022",
            "raw_dir": str(raw),
            "processed_dir": str(out),
            "official_split_note": "Official train/val/test/ood files are preserved.",
            "splits": {split: {"records": count} for split, count in counts.items()},
            "schema": schema,
        },
    )
    print(
        f"[astock] train={counts['train']} val={counts['val']} "
        f"test={counts['test']} ood={counts['ood']}"
    )
    return counts


def prepare_cmin_cn() -> dict[str, Any]:
    raw = _raw_dir("cmin_cn") / "CMIN-CN"
    out = _processed_dir("cmin_cn")
    _reset_dir(out)

    price_dir = raw / "price" / "preprocessed"
    news_dir = raw / "news" / "preprocessed"
    if not price_dir.exists() or not news_dir.exists():
        raise FileNotFoundError(f"expected {price_dir} and {news_dir}")

    price_files = sorted(price_dir.glob("*.txt"))
    news_stock_dirs = sorted(path for path in news_dir.iterdir() if path.is_dir())
    rows = []
    news_date_files = 0
    for stock_dir in news_stock_dirs:
        n_dates = sum(1 for path in stock_dir.iterdir() if path.is_file())
        news_date_files += n_dates
        has_price = int((price_dir / f"{stock_dir.name}.txt").exists())
        rows.append(f"{stock_dir.name}\t{has_price}\t{n_dates}")

    _write_lines(out / "stocks.tsv", ["stock\thas_price\tnews_date_files", *rows])
    manifest = {
        "dataset": "cmin_cn",
        "task": "Chinese stock movement prediction with news and prices",
        "source": "CMIN-CN, ACL 2023",
        "raw_dir": str(raw),
        "processed_dir": str(out),
        "official_split_note": (
            "Original release is organized by stock/date; no project loader is wired yet."
        ),
        "price_files": len(price_files),
        "news_stock_dirs": len(news_stock_dirs),
        "news_date_files": news_date_files,
        "processed_files": {"stocks": "stocks.tsv"},
    }
    _write_json(out / "manifest.json", manifest)
    print(
        f"[cmin_cn] price_files={manifest['price_files']} "
        f"news_stock_dirs={manifest['news_stock_dirs']} "
        f"news_date_files={manifest['news_date_files']}"
    )
    return manifest


_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def _detokenize_cjk(text: str) -> str:
    """Remove tokenizer spaces between CJK characters/punctuation."""
    cjk = "一-鿿，。；：、！？（）"
    return re.sub(rf"(?<=[{cjk}])\s+(?=[{cjk}])", "", text)


def _cell(row: list[str], col: dict[str, int], name: str) -> str:
    idx = col.get(name)
    return row[idx].strip() if idx is not None and idx < len(row) else ""


def export_astock_sarge_input(
    splits: tuple[str, ...] = ("train", "val", "test", "ood"),
    max_docs: int | None = None,
) -> dict[str, int]:
    """Astock news -> the SARGE inference-input JSONL (closed-loop entry point).

    One record per news item: ``{"doc_id", "text", "date", "stock",
    "stock_name", "split", "label"}``. The same file doubles as the
    `sarge_to_event_nodes.py --meta` sidecar (publication date + stock code per
    doc_id anchor the graph timeline). Astock stores the stock code as a
    de-zero-padded int and quotes text fields with embedded newlines, so rows
    are read quote-aware and codes re-padded; rows without a parseable ISO date
    or any text are dropped.
    """
    src_dir = _processed_dir("astock")
    csv.field_size_limit(10**7)
    records: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    done = False
    for split in splits:
        path = src_dir / f"{split}.tsv"
        if not path.exists():
            raise FileNotFoundError(path)
        kept = 0
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            header = next(reader)
            col = {name: idx for idx, name in enumerate(header)}
            for i, row in enumerate(reader):
                date = _cell(row, col, "DATE")[:10]
                if not _ISO_DAY.fullmatch(date):
                    continue
                text = _cell(row, col, "text_a")
                if not text:
                    text = "。".join(
                        part
                        for part in (_cell(row, col, "TITLE"), _cell(row, col, "DESCRIPTION"))
                        if part
                    )
                if not text:
                    continue
                code = _cell(row, col, "CODE")
                records.append(
                    {
                        "doc_id": f"astock-{split}-{i:06d}",
                        "text": text,
                        "date": date,
                        "stock": code.zfill(6) if code.isdigit() else code,
                        "stock_name": _cell(row, col, "NAME"),
                        "split": split,
                        "label": _cell(row, col, "label"),
                    }
                )
                kept += 1
                if max_docs is not None and len(records) >= max_docs:
                    done = True
                    break
        counts[split] = kept
        if done:
            break

    out = src_dir / "sarge_input.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    print(f"[astock] sarge_input: {len(records)} docs -> {_project_path(out)} ({counts})")
    return counts


def export_cmin_cn_sarge_input(
    max_docs: int | None = None, max_headlines: int = 20
) -> dict[str, Any]:
    """CMIN-CN raw news -> SARGE inference-input JSONL (one doc per stock-day).

    Raw layout is ``news/preprocessed/<stock>/<YYYY-MM-DD>`` with one tokenized
    headline per line; tokenizer spaces between CJK characters are removed and
    up to ``max_headlines`` headlines join into one document. Skips gracefully
    when the raw release is absent (mirrors `prepare_event_graph_zh`), so the
    Astock-first closed loop does not depend on this download.
    """
    news_dir = _raw_dir("cmin_cn") / "CMIN-CN" / "news" / "preprocessed"
    if not news_dir.exists():
        print("[cmin_cn] raw news not found — skipping sarge_input export")
        return {}

    records: list[dict[str, str]] = []
    n_stocks = 0
    done = False
    for stock_dir in sorted(p for p in news_dir.iterdir() if p.is_dir()):
        n_stocks += 1
        for date_file in sorted(p for p in stock_dir.iterdir() if p.is_file()):
            date = date_file.stem[:10]
            if not _ISO_DAY.fullmatch(date):
                continue
            lines = [
                line.strip()
                for line in date_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                continue
            records.append(
                {
                    "doc_id": f"cmin-{stock_dir.name}-{date}",
                    "text": "；".join(_detokenize_cjk(line) for line in lines[:max_headlines]),
                    "date": date,
                    "stock": stock_dir.name,
                    "split": "all",
                }
            )
            if max_docs is not None and len(records) >= max_docs:
                done = True
                break
        if done:
            break

    out = _processed_dir("cmin_cn") / "sarge_input.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    print(
        f"[cmin_cn] sarge_input: {len(records)} docs from {n_stocks} stocks "
        f"-> {_project_path(out)}"
    )
    return {"docs": len(records), "stocks": n_stocks}


def prepare_event_graph_zh() -> dict[str, Any] | None:
    candidates = [
        RAW_ROOT / "event_graph_zh" / "event_graph.json",
        DATA_ROOT / "event_graph_zh" / "event_graph.json",
    ]
    src = next((path for path in candidates if path.exists()), None)
    if src is None:
        print("[event_graph_zh] no event_graph.json found — skipping")
        return None

    out = _processed_dir("event_graph_zh")
    _reset_dir(out)
    dst = out / "event_graph.json"
    shutil.copyfile(src, dst)
    manifest = {
        "dataset": "event_graph_zh",
        "task": "self-built Chinese financial event graph forecasting",
        "source": _project_path(src),
        "processed_dir": _project_path(out),
        "processed_files": {"event_graph": "event_graph.json"},
    }
    _write_json(out / "manifest.json", manifest)
    print(f"[event_graph_zh] copied {src} -> {dst}")
    return manifest


def verify_tsv_order(name: str) -> None:
    tsv = _processed_dir(name) / f"{name}.tsv"
    if not tsv.exists():
        print(f"[{name}] processed TSV not found — skipping")
        return
    timestamps = []
    with tsv.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                try:
                    timestamps.append(int(float(parts[3])))
                except ValueError:
                    pass
    if not timestamps:
        print(f"[{name}] empty TSV")
        return
    n_out = sum(1 for a, b in zip(timestamps, timestamps[1:], strict=False) if a > b)
    status = "chronological" if n_out == 0 else f"{n_out} out-of-order rows"
    print(f"[{name}] {len(timestamps)} quads t=[{timestamps[0]}, {timestamps[-1]}] {status}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccks-val-n", type=int, default=700)
    parser.add_argument("--ccks-test-n", type=int, default=700)
    parser.add_argument("--ccks-seed", type=int, default=42)
    parser.add_argument("--maven-smoke-n", type=int, default=30)
    parser.add_argument("--maven-smoke-valid-n", type=int, default=20)
    parser.add_argument("--tkg-smoke-n", type=int, default=1000)
    parser.add_argument("--tkg-tiny-n", type=int, default=200)
    parser.add_argument(
        "--export-sarge-input",
        choices=("astock", "cmin_cn"),
        help="only export this dataset's news -> SARGE input JSONL, then exit "
        "(no other dataset is touched)",
    )
    parser.add_argument("--max-docs", type=int, help="cap for --export-sarge-input")
    parser.add_argument(
        "--only",
        nargs="*",
        help="run only these dataset preparers (default: all), e.g. --only maven_arg maven_fact",
    )
    args = parser.parse_args()

    if args.export_sarge_input == "astock":
        export_astock_sarge_input(max_docs=args.max_docs)
        return 0
    if args.export_sarge_input == "cmin_cn":
        export_cmin_cn_sarge_input(max_docs=args.max_docs)
        return 0

    only = set(args.only or [])

    def want(name: str) -> bool:
        return not only or name in only

    print("── MAVEN suite (node / relation / factuality, same 4480 docs) ─")
    if want("maven_ere"):
        prepare_maven_ere(train_smoke_n=args.maven_smoke_n, valid_smoke_n=args.maven_smoke_valid_n)
    if want("maven_arg"):
        prepare_maven_arg(train_smoke_n=args.maven_smoke_n, valid_smoke_n=args.maven_smoke_valid_n)
    if want("maven_fact"):
        prepare_maven_fact(train_smoke_n=args.maven_smoke_n, valid_smoke_n=args.maven_smoke_valid_n)
    if want("modafact"):
        prepare_modafact()
    if want("it_happened"):
        prepare_it_happened()
    if want("docee"):
        prepare_docee()
    if want("ccks_fin_causal"):
        split_ccks(val_n=args.ccks_val_n, test_n=args.ccks_test_n, seed=args.ccks_seed)

    if want("icews14") or want("findkg"):
        print("── Forecasting datasets ────────────────────")
        if want("icews14"):
            prepare_tkg_splits("icews14", smoke_n=args.tkg_smoke_n, tiny_n=args.tkg_tiny_n)
            verify_tsv_order("icews14")
        if want("findkg"):
            prepare_tkg_splits("findkg", smoke_n=args.tkg_smoke_n, tiny_n=args.tkg_tiny_n)
            verify_tsv_order("findkg")

    if want("cmin_cn") or want("astock"):
        print("── Stock movement datasets ─────────────────")
        if want("cmin_cn"):
            prepare_cmin_cn()
        if want("astock"):
            prepare_astock()

    if want("event_graph_zh"):
        print("── Self-built event graph ──────────────────")
        prepare_event_graph_zh()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
