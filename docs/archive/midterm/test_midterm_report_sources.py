from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_report_module():
    path = Path("docs/midterm/report/build_midterm_report.py")
    spec = importlib.util.spec_from_file_location("build_midterm_report", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_midterm_report_reads_latest_ch3_artifacts() -> None:
    report = _load_report_module()

    ch3 = report.load_ch3_results()

    assert ch3["frequency"]["mrr"] == 0.1050
    assert ch3["path_rl"]["mrr"] == 0.3599
    assert ch3["re_gcn"]["mrr_tfilt"] == 0.3796
    assert ch3["hybrid_single"]["mrr_tfilt"] == 0.3859
    assert ch3["fusion_sweep"]["mrr_tfilt"] == 0.4110
    assert ch3["fusion_sweep"]["weights"] == "ws=1.0, wc=0.3"


def test_midterm_report_reads_conformal_headline() -> None:
    report = _load_report_module()

    conformal = report.load_conformal_results()

    assert conformal["split"]["coverage"] == 0.8579
    assert conformal["split"]["drift_gap"] == 0.29
    assert conformal["aci"]["coverage"] == 0.8993
    assert conformal["aci"]["drift_gap"] == 0.20
