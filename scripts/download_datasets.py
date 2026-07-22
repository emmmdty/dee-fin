#!/usr/bin/env python
"""Fetch public datasets into data/raw and point users to preprocessing.

Project-ready targets are produced by scripts/preprocess_datasets.py:
- raw:       data/raw/{dataset}/...
- processed: data/processed/{dataset}/...

Downloads prefer mirrors (set FINEKG_MIRROR or rely on the listed mirror URL).
Datasets needing a login / manual export (Tianchi CCKS, hidden MAVEN test labels)
print clear instructions instead of failing silently.

    uv run python scripts/download_datasets.py --list
    uv run python scripts/download_datasets.py --dataset icews14
"""

from __future__ import annotations

import argparse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


@dataclass
class Dataset:
    name: str
    kind: str  # "quadruple" | "maven" | "manual"
    note: str
    urls: tuple[str, ...] = ()


DATASETS: dict[str, Dataset] = {
    "maven_ere": Dataset(
        name="maven_ere",
        kind="maven",
        note="MAVEN-ERE (EMNLP'22). Official train/valid jsonl; test labels are hidden.",
        urls=("https://cloud.tsinghua.edu.cn/d/d520f5db5c1b4e2bb006/",),
    ),
    "icews14": Dataset(
        name="icews14",
        kind="quadruple",
        note="ICEWS14 TKG forecasting benchmark (quadruple train/valid/test).",
        urls=("https://raw.githubusercontent.com/Liyyy2122/TiRGN/main/data/ICEWS14/",),
    ),
    "findkg": Dataset(
        name="findkg",
        kind="quadruple",
        note="FinDKG (ICAIF'24). Financial dynamic KG quadruples (GPL-3.0).",
        urls=("https://raw.githubusercontent.com/xiaohui-victor-li/FinDKG/main/FinDKG_dataset/FinDKG/",),
    ),
    "ccks_fin_causal": Dataset(
        name="ccks_fin_causal",
        kind="manual",
        note=(
            "CCKS-2021 financial event causality (Ant + CAS). Export from Tianchi "
            "after login, then convert to the causal_pairs jsonl shown in "
            "data/fixtures/ccks_fin_causal/sample.jsonl."
        ),
    ),
    "cmin_cn": Dataset(
        name="cmin_cn",
        kind="manual",
        note="CMIN-CN (ACL'23) stock movement. Clone github.com/BigRoddy/CMIN-Dataset.",
    ),
}


def _download(urls: tuple[str, ...], dest: Path) -> bool:
    mirror = None
    import os

    if os.environ.get("FINEKG_MIRROR"):
        mirror = os.environ["FINEKG_MIRROR"]
    candidates = ([mirror] if mirror else []) + list(urls)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in candidates:
        try:
            print(f"[download] {url} -> {dest}")
            urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted dataset hosts)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[download] failed: {exc}")
    return False


def _convert_quadruples(raw_dir: Path, canonical: Path) -> None:
    """Concatenate train/valid/test quadruple files into one TSV.

    Accepts the common `head<TAB>rel<TAB>tail<TAB>time[...]` layout (extra
    trailing columns are ignored); keeps the first four columns.
    """
    canonical.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with canonical.open("w", encoding="utf-8") as out:
        for split in ("train.txt", "valid.txt", "test.txt"):
            path = raw_dir / split
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                cols = line.replace("\t", " ").split()
                if len(cols) >= 4:
                    out.write("\t".join(cols[:4]) + "\n")
                    n += 1
    print(f"[convert] wrote {n} quadruples -> {canonical}")


def prepare(dataset: Dataset) -> int:
    raw_dir = DATA_ROOT / "raw" / dataset.name
    processed_dir = DATA_ROOT / "processed" / dataset.name

    if dataset.kind == "manual":
        print(f"[{dataset.name}] manual dataset.\n  {dataset.note}")
        print(f"  Place original files under: {raw_dir}")
        print("  Then run: uv run python scripts/preprocess_datasets.py")
        return 0

    if dataset.kind == "quadruple":
        for split in ("train.txt", "valid.txt", "test.txt"):
            _download(tuple(u + split for u in dataset.urls), raw_dir / split)
        _convert_quadruples(raw_dir, processed_dir / f"{dataset.name}.tsv")
        print("  For full split/manifests, run: uv run python scripts/preprocess_datasets.py")
        return 0

    if dataset.kind == "maven":
        print(f"[{dataset.name}] {dataset.note}")
        print(f"  Download the official jsonl from: {dataset.urls[0]}")
        print(f"  Then place train.jsonl/valid.jsonl/test.jsonl under: {raw_dir}")
        print("  Then run: uv run python scripts/preprocess_datasets.py")
        print("  The loader reads the official MAVEN-ERE format directly.")
        return 0

    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), help="dataset to prepare")
    parser.add_argument("--list", action="store_true", help="list known datasets")
    args = parser.parse_args()

    if args.list or not args.dataset:
        for ds in DATASETS.values():
            print(f"- {ds.name:16s} [{ds.kind:9s}] {ds.note}")
        return 0
    return prepare(DATASETS[args.dataset])


if __name__ == "__main__":
    raise SystemExit(main())
