"""sarge_to_event_nodes derives the subject + ISO time anchor that the event
graph needs from SARGE's surface-only arguments. The Chinese date parsing and the
role-cue heuristics are the load-bearing bits, checked on tiny inputs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "sarge_to_event_nodes.py"
    spec = importlib.util.spec_from_file_location("sarge_to_event_nodes", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s2e = _load_module()


def _run_main(argv: list[str]) -> int:
    old_argv = sys.argv
    sys.argv = argv
    try:
        return s2e.main()
    finally:
        sys.argv = old_argv


def test_iso_date_parses_chinese_and_iso() -> None:
    assert s2e._iso_date("2019年10月15日") == "2019-10-15"
    assert s2e._iso_date("2019-10-15") == "2019-10-15"
    assert s2e._iso_date("2019年10月") == "2019-10-01"  # month-only -> day 01
    assert s2e._iso_date("近期") is None
    assert s2e._iso_date(None) is None


def test_derive_subject_prefers_actor_roles() -> None:
    assert s2e._derive_subject({"回购方": "众安集团", "披露时间": "2019年10月15日"}) == "众安集团"
    # no actor cue -> first non-date argument
    assert s2e._derive_subject({"金额": "1亿", "交易日期": "2020-01-01"}) == "1亿"


def test_derive_time_anchor_reads_date_roles() -> None:
    assert s2e._derive_time_anchor({"披露时间": "2019年10月15日"}) == "2019-10-15"
    assert s2e._derive_time_anchor({"回购方": "A"}) is None


def test_main_flattens_nested_events(tmp_path) -> None:
    pred = tmp_path / "pred.jsonl"
    pred.write_text(
        json.dumps(
            {
                "doc_id": "d1",
                "events": [
                    {
                        "event_type": "股份回购",
                        "arguments": {
                            "回购方": [{"text": "众安集团"}],
                            "披露时间": [{"text": "2019年10月15日"}],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "nodes.jsonl"
    assert _run_main(["x", "--pred", str(pred), "--output", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["event_type"] == "股份回购"
    assert rec["subject"] == "众安集团"
    assert rec["time_anchor"] == "2019-10-15"
    assert rec["arguments"]["披露时间"] == "2019年10月15日"


def test_main_locates_argument_evidence_from_source_docs(tmp_path) -> None:
    pred = tmp_path / "pred.jsonl"
    pred.write_text(
        json.dumps(
            {
                "doc_id": "d1",
                "events": [
                    {
                        "event_type": "股份回购",
                        "arguments": {
                            "回购方": [{"text": "众安集团"}],
                            "回购完成时间": [{"text": "2019年10月15日"}],
                            "交易金额": [{"text": "94.56万港元"}],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    source_docs = tmp_path / "dev.jsonl"
    text = "客户端\n众安集团发布公告，于2019年10月15日耗资94.56万港元回购。"
    source_docs.write_text(
        json.dumps({"id": "d1", "text": text}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "nodes.jsonl"

    assert (
        _run_main(
            [
                "x",
                "--pred",
                str(pred),
                "--source-docs",
                str(source_docs),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])

    span = rec["argument_evidence"]["交易金额"][0]
    assert span == {
        "doc_id": "d1",
        "char_start": text.index("94.56万港元"),
        "char_end": text.index("94.56万港元") + len("94.56万港元"),
        "sent_id": None,
        "text": "94.56万港元",
    }
    assert rec["trigger"] == ""
    assert rec["trigger_evidence"] == []


def test_main_preserves_existing_evidence_spans(tmp_path) -> None:
    pred = tmp_path / "pred.jsonl"
    pred.write_text(
        json.dumps(
            {
                "doc_id": "d1",
                "events": [
                    {
                        "event_type": "股份回购",
                        "trigger": "回购",
                        "trigger_evidence": [
                            {"doc_id": "d1", "char_start": 20, "char_end": 22, "text": "回购"}
                        ],
                        "arguments": {
                            "回购方": [
                                {
                                    "text": "众安集团",
                                    "doc_id": "d1",
                                    "char_start": 4,
                                    "char_end": 8,
                                }
                            ]
                        },
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "nodes.jsonl"

    assert _run_main(["x", "--pred", str(pred), "--output", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])

    assert rec["trigger"] == "回购"
    assert rec["trigger_evidence"] == [
        {"doc_id": "d1", "char_start": 20, "char_end": 22, "sent_id": None, "text": "回购"}
    ]
    assert rec["argument_evidence"]["回购方"] == [
        {"doc_id": "d1", "char_start": 4, "char_end": 8, "sent_id": None, "text": "众安集团"}
    ]


def test_main_does_not_fabricate_missing_evidence_or_gold_trigger(tmp_path) -> None:
    pred = tmp_path / "pred.jsonl"
    pred.write_text(
        json.dumps(
            {
                "doc_id": "d1",
                "events": [
                    {
                        "event_type": "股份回购",
                        "arguments": {"回购方": [{"text": "不存在公司"}]},
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    source_docs = tmp_path / "dev.jsonl"
    source_docs.write_text(
        json.dumps(
            {
                "id": "d1",
                "text": "众安集团发布公告。",
                "event_list": [{"trigger": "回购", "event_type": "股份回购"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "nodes.jsonl"

    assert (
        _run_main(
            [
                "x",
                "--pred",
                str(pred),
                "--source-docs",
                str(source_docs),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])

    assert rec["argument_evidence"] == {}
    assert rec["trigger"] == ""
    assert rec["trigger_evidence"] == []


def test_main_meta_sidecar_overrides_anchor_and_subject(tmp_path) -> None:
    pred = tmp_path / "pred.jsonl"
    pred.write_text(
        "".join(
            json.dumps(doc, ensure_ascii=False) + "\n"
            for doc in (
                {
                    "doc_id": "astock-train-000001",
                    "events": [
                        {"event_type": "业绩预告", "arguments": {"公司": [{"text": "众安集团"}]}}
                    ],
                },
                {
                    "doc_id": "no-meta-doc",
                    "events": [
                        {
                            "event_type": "股份回购",
                            "arguments": {
                                "回购方": [{"text": "中泰化学"}],
                                "披露时间": [{"text": "2019年10月15日"}],
                            },
                        }
                    ],
                },
            )
        ),
        encoding="utf-8",
    )
    meta = tmp_path / "sarge_input.jsonl"
    meta.write_text(
        json.dumps(
            {"doc_id": "astock-train-000001", "date": "2021-03-05", "stock": "600030"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "nodes.jsonl"
    assert _run_main(["x", "--pred", str(pred), "--meta", str(meta), "--output", str(out)]) == 0
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    with_meta, without_meta = lines
    # meta sidecar wins: publication date anchors the timeline, the stock code
    # becomes the canonical cross-doc subject
    assert with_meta["time_anchor"] == "2021-03-05"
    assert with_meta["subject"] == "600030"
    # docs missing from the sidecar keep the role-cue derivation fallback
    assert without_meta["time_anchor"] == "2019-10-15"
    assert without_meta["subject"] == "中泰化学"


def _pred_doc(doc_id: str, events: list[dict]) -> str:
    return json.dumps({"doc_id": doc_id, "events": events}, ensure_ascii=False) + "\n"


def test_main_quality_gates_filter_domain_shift_noise(tmp_path) -> None:
    pred = tmp_path / "pred.jsonl"
    pred.write_text(
        _pred_doc(
            "d1",
            [
                # self-loop: two distinct actor roles share one value (mislabel)
                {
                    "event_type": "企业收购",
                    "arguments": {
                        "收购方": [{"text": "紫光国微"}],
                        "被收购方": [{"text": "紫光国微"}],
                    },
                },
                # thin event: only one non-date argument -> below --min-args 2
                {"event_type": "中标", "arguments": {"中标公司": [{"text": "江苏有线"}]}},
                # healthy event: distinct actors, three non-date arguments
                {
                    "event_type": "企业收购",
                    "arguments": {
                        "收购方": [{"text": "江苏有线"}],
                        "被收购方": [{"text": "吉视传媒"}],
                        "交易金额": [{"text": "2亿元"}],
                        "披露时间": [{"text": "2020年10月13日"}],
                    },
                },
            ],
        ),
        encoding="utf-8",
    )
    out = tmp_path / "nodes.jsonl"
    assert (
        _run_main(
            [
                "x",
                "--pred", str(pred),
                "--output", str(out),
                "--drop-self-loops",
                "--min-args", "2",
            ]
        )
        == 0
    )
    recs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    # only the healthy acquisition survives both gates
    assert len(recs) == 1
    assert recs[0]["arguments"]["被收购方"] == "吉视传媒"


def test_main_gates_off_by_default(tmp_path) -> None:
    pred = tmp_path / "pred.jsonl"
    pred.write_text(
        _pred_doc(
            "d1",
            [
                {
                    "event_type": "企业收购",
                    "arguments": {
                        "收购方": [{"text": "紫光国微"}],
                        "被收购方": [{"text": "紫光国微"}],
                    },
                }
            ],
        ),
        encoding="utf-8",
    )
    out = tmp_path / "nodes.jsonl"
    assert _run_main(["x", "--pred", str(pred), "--output", str(out)]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1


def test_main_type_cue_gate_grounds_event_type_in_source(tmp_path) -> None:
    pred = tmp_path / "pred.jsonl"
    pred.write_text(
        _pred_doc(
            "d1",
            [  # "acquisition" over a joint-venture news: no 收购 cue in text -> drop
                {
                    "event_type": "企业收购",
                    "arguments": {
                        "收购方": [{"text": "江苏有线"}],
                        "被收购方": [{"text": "吉视传媒"}],
                        "交易金额": [{"text": "2亿元"}],
                    },
                }
            ],
        )
        + _pred_doc(
            "d2",
            [  # real acquisition: cue present -> keep
                {
                    "event_type": "企业收购",
                    "arguments": {
                        "收购方": [{"text": "A公司"}],
                        "被收购方": [{"text": "B公司"}],
                        "交易金额": [{"text": "3亿元"}],
                    },
                },
                # unknown type stays (fail-open: the cue map only covers the schema)
                {"event_type": "神秘类型", "arguments": {"主体": [{"text": "C公司"}],
                                                     "对象": [{"text": "D公司"}]}},
            ],
        ),
        encoding="utf-8",
    )
    meta = tmp_path / "meta.jsonl"
    meta.write_text(
        json.dumps({"doc_id": "d1", "date": "2020-10-13", "stock": "600959",
                    "text": "江苏有线、吉视传媒共同发起组建中国广电网络股份有限公司。"},
                   ensure_ascii=False) + "\n"
        + json.dumps({"doc_id": "d2", "date": "2020-11-01", "stock": "000001",
                      "text": "A公司公告，拟以3亿元收购B公司全部股权。"},
                     ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "nodes.jsonl"
    assert (
        _run_main(
            ["x", "--pred", str(pred), "--meta", str(meta), "--output", str(out),
             "--require-type-cue"]
        )
        == 0
    )
    recs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["doc_id"] for r in recs] == ["d2", "d2"]
    assert {r["event_type"] for r in recs} == {"企业收购", "神秘类型"}


# --------------------------------------------------------------------------- #
# Counterparty (object) derivation: the entity-level closed-loop graph needs it.
# --------------------------------------------------------------------------- #
def test_derive_object_prefers_exact_object_role() -> None:
    # "收购方" is a substring of "被收购方": a plain `cue in role` scan could
    # return the acquirer. Exact role match must win.
    args = {"收购方": "甲公司", "被收购方": "乙公司", "交易金额": "3亿元"}
    assert s2e._derive_object(args, "甲公司") == "乙公司"
    assert s2e._derive_object(args) == "乙公司"


def test_derive_object_never_returns_the_subject() -> None:
    assert s2e._derive_object({"被收购方": "甲公司"}, "甲公司") is None


def test_derive_object_is_none_for_unary_events() -> None:
    assert s2e._derive_object({"回购方": "甲公司", "回购股份数量": "100万股"}, "甲公司") is None


def _run_one_event(tmp_path, event: dict) -> dict:
    pred = tmp_path / "pred.jsonl"
    pred.write_text(
        json.dumps({"doc_id": "d1", "events": [event]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "nodes.jsonl"
    assert _run_main(["x", "--pred", str(pred), "--output", str(out)]) == 0
    return json.loads(out.read_text(encoding="utf-8").splitlines()[0])


def test_main_writes_object_into_metadata(tmp_path) -> None:
    rec = _run_one_event(
        tmp_path,
        {"event_type": "企业收购",
         "arguments": {"收购方": [{"text": "甲公司"}], "被收购方": [{"text": "乙公司"}]}},
    )
    assert rec["metadata"] == {"object": "乙公司"}
    assert rec["subject"] == "甲公司"


def test_main_omits_metadata_for_objectless_event(tmp_path) -> None:
    rec = _run_one_event(
        tmp_path,
        {"event_type": "股份回购",
         "arguments": {"回购方": [{"text": "甲公司"}], "回购股份数量": [{"text": "100万股"}]}},
    )
    assert "object" not in rec.get("metadata", {})
