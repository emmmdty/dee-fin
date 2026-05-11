from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data_split import split_utils as u
from scripts.data_split.prepare_chfinann import OUTPUT_NAME as CHFINANN_OUTPUT
from scripts.data_split.prepare_chfinann import prepare as prepare_chfinann
from scripts.data_split.prepare_docfee import OUTPUT_NAME as DOCFEE_OUTPUT
from scripts.data_split.prepare_docfee import prepare as prepare_docfee
from scripts.data_split.prepare_duee_fin import OUTPUT_NAME as DUEE_OUTPUT
from scripts.data_split.prepare_duee_fin import prepare as prepare_duee_fin


BEGIN_MARKER = "<!-- BEGIN LOCAL_PROCESSED_SPLITS -->"
END_MARKER = "<!-- END LOCAL_PROCESSED_SPLITS -->"
RAW_INPUT_PATHS = (
    "data/raw/ChFinAnn/train.json",
    "data/raw/ChFinAnn/dev.json",
    "data/raw/ChFinAnn/test.json",
    "data/raw/ChFinAnn/schema.json",
    "data/raw/DuEE-Fin/duee_fin_train.json",
    "data/raw/DuEE-Fin/duee_fin_dev.json",
    "data/raw/DuEE-Fin/duee_fin_test.json",
    "data/raw/DuEE-Fin/duee_fin_event_schema.json",
    "data/raw/DocFEE/train.jsonl",
    "data/raw/DocFEE/test.jsonl",
    "data/raw/DocFEE/schema.json",
)
EXPECTED_OUTPUT_FILES = {
    CHFINANN_OUTPUT: (
        "train.json",
        "dev.json",
        "test.json",
        "schema.json",
        "schema_validation_report.json",
        "split_manifest.json",
        "README.md",
    ),
    DUEE_OUTPUT: (
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "blind_test_unlabeled.jsonl",
        "schema.json",
        "split_manifest.json",
        "README.md",
    ),
    DOCFEE_OUTPUT: (
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "schema.json",
        "split_manifest.json",
        "README.md",
    ),
}


def _counts_row(label: str, manifest: dict[str, Any]) -> str:
    counts = manifest["document_counts_per_split"]
    events = manifest["event_record_counts_per_split"]
    parts = []
    for split in counts:
        parts.append(f"`{split}` {counts[split]} docs / {events[split]} events")
    return f"- **{label}**：{'; '.join(parts)}"


def _docfee_duplicate_readme_bullet(manifest: dict[str, Any]) -> str:
    duplicate_diagnostics = manifest["duplicate_diagnostics"]
    cross_counts = duplicate_diagnostics["cross_split_exact_text_duplicate_count"]
    if not any(cross_counts.values()):
        return "- `DocFEE-dev1000` 跨 split exact-text duplicate 审计未发现重复正文。"

    audit_entries = duplicate_diagnostics["cross_split_exact_text_duplicate_json_hash_audit"]
    same_raw_count = sum(1 for entry in audit_entries if entry["same_raw_json_line_hash"])
    same_canonical_count = sum(1 for entry in audit_entries if entry["same_canonical_json_hash"])
    count_text = "，".join(f"`{pair}`={count}" for pair, count in sorted(cross_counts.items()))
    return (
        "- `DocFEE-dev1000` 跨 split exact-text duplicate 审计："
        f"{count_text}；整条 JSON hash 对比为 "
        f"`same_raw_json_line_hash`={same_raw_count}、`same_canonical_json_hash`={same_canonical_count}，"
        "说明这些是 same-content/different-record 的审计项，不是整条 JSON 样本被复制到多个 split。"
    )


def update_data_readme(project_root: Path, manifests: dict[str, dict[str, Any]]) -> None:
    readme_path = project_root / "data/README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    section = f"""{BEGIN_MARKER}

## 本地处理后数据划分

本节记录由 `scripts/data_split/prepare_all_splits.py` 生成的可复现实验划分。复现命令：

```bash
python scripts/data_split/prepare_all_splits.py --project-root .
```

输出目录：

- `data/processed/{CHFINANN_OUTPUT}/`
- `data/processed/{DUEE_OUTPUT}/`
- `data/processed/{DOCFEE_OUTPUT}/`

划分摘要：

{_counts_row(CHFINANN_OUTPUT, manifests[CHFINANN_OUTPUT])}
{_counts_row(DUEE_OUTPUT, manifests[DUEE_OUTPUT])}
{_counts_row(DOCFEE_OUTPUT, manifests[DOCFEE_OUTPUT])}

处理原则：

- 不修改 `data/raw/` 下的原始文件。
- 不做去重；重复文本只在 `split_manifest.json` 中作为审计诊断报告。
- 不把 offset/drange 作为规范 gold 角色值抽取的必要条件。
- `ChFinAnn-Doc2EDAG` 保留既有 Doc2EDAG 风格官方 train/dev/test 划分，并在处理目录生成经验验证后的 `schema.json` 与 `schema_validation_report.json`。
- `DuEE-Fin-dev500` 使用 raw dev 作为本地 labeled test，从 raw train 中用 `seed=42` 的确定性分层 hash 方法抽取 500 条 dev，其余作为 train；raw test 仅保留为 `blind_test_unlabeled.jsonl`。
- `DocFEE-dev1000` 保留官方 raw test，从 raw train 中用 `seed=42` 的确定性分层 hash 方法抽取 1000 条 dev，其余作为 train。

重要提醒：

- `DuEE-Fin-dev500` 是公开数据上的确定性 dev500，本地复现目标不是恢复任何隐藏 dev id。
- ChFinAnn 原始 `schema.json` 只作为候选 schema 使用；若需引用处理后 schema，请使用 `data/processed/{CHFINANN_OUTPUT}/schema.json` 和对应验证报告。
{_docfee_duplicate_readme_bullet(manifests[DOCFEE_OUTPUT])}

{END_MARKER}
"""

    if BEGIN_MARKER in existing and END_MARKER in existing:
        before, rest = existing.split(BEGIN_MARKER, 1)
        _, after = rest.split(END_MARKER, 1)
        new_content = before.rstrip() + "\n\n" + section + after.lstrip()
    else:
        new_content = existing.rstrip() + "\n\n" + section if existing.strip() else section
    readme_path.write_text(new_content, encoding="utf-8")


def _hash_raw_inputs(project_root: Path) -> dict[str, str]:
    return {path: u.sha256_file(project_root / path) for path in RAW_INPUT_PATHS}


def _assert_no_labeled_split_overlap(name: str, manifest: dict[str, Any]) -> None:
    split_ids = manifest["split_ids"]
    labeled_splits = [split for split in ("train", "dev", "test") if split in split_ids]
    for index, left in enumerate(labeled_splits):
        left_ids = set(split_ids[left])
        for right in labeled_splits[index + 1 :]:
            overlap = left_ids & set(split_ids[right])
            if overlap:
                examples = sorted(overlap)[:5]
                raise AssertionError(f"{name} has document id overlap between {left} and {right}: {examples}")


def validate_outputs(project_root: Path, manifests: dict[str, dict[str, Any]], raw_hashes_before: dict[str, str]) -> None:
    raw_hashes_after = _hash_raw_inputs(project_root)
    if raw_hashes_after != raw_hashes_before:
        changed = sorted(path for path in raw_hashes_before if raw_hashes_before[path] != raw_hashes_after.get(path))
        raise AssertionError(f"raw input files changed unexpectedly: {changed}")

    for dataset_name, filenames in EXPECTED_OUTPUT_FILES.items():
        output_dir = project_root / "data/processed" / dataset_name
        missing = [filename for filename in filenames if not (output_dir / filename).exists()]
        if missing:
            raise AssertionError(f"{dataset_name} missing expected output files: {missing}")
        manifest_path = output_dir / "split_manifest.json"
        manifest_from_disk = u.read_json_file(manifest_path)
        if manifest_from_disk["document_counts_per_split"] != manifests[dataset_name]["document_counts_per_split"]:
            raise AssertionError(f"{dataset_name} manifest JSON on disk does not match in-memory counts")
        _assert_no_labeled_split_overlap(dataset_name, manifest_from_disk)

    duee_counts = manifests[DUEE_OUTPUT]["document_counts_per_split"]
    if duee_counts.get("dev") != 500:
        raise AssertionError(f"{DUEE_OUTPUT} dev count is {duee_counts.get('dev')}, expected 500")
    docfee_counts = manifests[DOCFEE_OUTPUT]["document_counts_per_split"]
    if docfee_counts.get("dev") != 1000:
        raise AssertionError(f"{DOCFEE_OUTPUT} dev count is {docfee_counts.get('dev')}, expected 1000")

    chfinann_raw_dir = project_root / "data/raw/ChFinAnn"
    chfinann_output_counts = manifests[CHFINANN_OUTPUT]["document_counts_per_split"]
    for split in ("train", "dev", "test"):
        raw_records, _, _ = u.load_records(chfinann_raw_dir / f"{split}.json")
        if chfinann_output_counts.get(split) != len(raw_records):
            raise AssertionError(
                f"{CHFINANN_OUTPUT} {split} count is {chfinann_output_counts.get(split)}, "
                f"expected raw count {len(raw_records)}"
            )

    data_readme = project_root / "data/README.md"
    readme_text = data_readme.read_text(encoding="utf-8")
    if BEGIN_MARKER not in readme_text or END_MARKER not in readme_text:
        raise AssertionError("data/README.md is missing local processed split markers")

    print("\n== Validation checks ==")
    print("  expected output files: ok")
    print("  manifest JSON files: ok")
    print("  DuEE-Fin dev=500: ok")
    print("  DocFEE dev=1000: ok")
    print("  labeled split id overlap: none")
    print("  ChFinAnn counts match raw split counts: ok")
    print("  raw input file hashes unchanged: ok")
    print("  data/README.md markers: ok")


def print_final_summary(manifests: dict[str, dict[str, Any]]) -> None:
    print("\n== Final split summary ==")
    for name, manifest in manifests.items():
        print(f"{name}:")
        for split, count in manifest["document_counts_per_split"].items():
            events = manifest["event_record_counts_per_split"][split]
            event_counts = manifest["per_document_event_count_distribution"][split]
            print(
                f"  {split}: docs={count}, events={events}, "
                f"single={event_counts['single_event_documents']}, "
                f"multi={event_counts['multi_event_documents']}"
            )
    print("\nSchema validation:")
    chfinann_schema = manifests[CHFINANN_OUTPUT]["schema_validation"]
    print(f"  {CHFINANN_OUTPUT}: {chfinann_schema['status']}")
    print("\nImbalance warnings:")
    any_warnings = False
    for name, manifest in manifests.items():
        split_quality = manifest.get("split_quality_diagnostics")
        warnings = split_quality.get("warnings", []) if isinstance(split_quality, dict) else []
        if warnings:
            any_warnings = True
            print(f"  {name}:")
            for warning in warnings:
                print(f"    - {warning}")
    if not any_warnings:
        print("  none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare all local document-level EE dataset splits.")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()

    project_root = u.resolve_project_root(args.project_root)
    raw_hashes_before = _hash_raw_inputs(project_root)
    manifests = {
        CHFINANN_OUTPUT: prepare_chfinann(project_root),
        DUEE_OUTPUT: prepare_duee_fin(project_root),
        DOCFEE_OUTPUT: prepare_docfee(project_root),
    }
    update_data_readme(project_root, manifests)
    validate_outputs(project_root, manifests, raw_hashes_before)
    print_final_summary(manifests)


if __name__ == "__main__":
    main()
