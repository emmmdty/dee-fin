"""Sanity check that the sarge package and all subpackages import cleanly."""

import importlib

SUBPACKAGES = [
    "sarge",
    "sarge.data",
    "sarge.models",
    "sarge.generation",
    "sarge.selection",
    "sarge.surface_memory",
    "sarge.slot_planning",
    "sarge.postprocess",
    "sarge.record_planning",
    "sarge.record_binding",
    "sarge.pipeline",
    "sarge.evaluation",
    "sarge.utils",
]


def test_all_subpackages_importable():
    for name in SUBPACKAGES:
        module = importlib.import_module(name)
        assert module is not None, f"failed to import {name}"


def test_sarge_version_present():
    import sarge

    assert hasattr(sarge, "__version__")
    assert isinstance(sarge.__version__, str)
