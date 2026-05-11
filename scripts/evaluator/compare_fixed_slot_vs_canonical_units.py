#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def load_audit_module():
    script_path = Path(__file__).with_name("audit_multi_value_roles.py")
    spec = importlib.util.spec_from_file_location("audit_multi_value_roles", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load audit module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ordered fixed-slot non-empty units with canonical unique role-value units."
    )
    parser.add_argument("--project-root", default=".", help="Project root containing data/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    audit = load_audit_module()
    report = audit.run_audit(project_root)

    print("# Fixed-slot vs canonical role-value units")
    print("dataset\tsplit\tfixed_slots\tcanonical_unique_units\tdelta\tfixed_slot_status")
    for row in report["results"]:
        fixed_slots = row["fixed_slot_non_empty_unit_count"]
        canonical_unique = row["canonical_unique_role_value_unit_count"]
        if isinstance(fixed_slots, int):
            delta = canonical_unique - fixed_slots
        else:
            delta = "not_applicable"
        print(
            "\t".join(
                str(value)
                for value in (
                    row["dataset"],
                    row["split"],
                    fixed_slots,
                    canonical_unique,
                    delta,
                    row["fixed_slot_status"],
                )
            )
        )


if __name__ == "__main__":
    main()
