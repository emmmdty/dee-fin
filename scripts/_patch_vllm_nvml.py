#!/usr/bin/env python
"""Make vLLM tolerate a faulted GPU during its import-time NVML enumeration.

vllm/platforms/cuda.py runs ``CudaPlatform.log_warnings()`` at *import* time, which
probes EVERY physical GPU via ``nvmlDeviceGetHandleByIndex``. On the gpu-4090 box
card3 (0000:CA:00.0) intermittently falls off the bus and raises
``NVMLError_Unknown``, which aborts the whole vLLM import — and therefore every
``trl`` GRPOTrainer / ``trl vllm-serve`` process — regardless of
``CUDA_VISIBLE_DEVICES`` (NVML ignores it). This is environment-level breakage,
not a code bug, so we patch the installed dependency (it lives under .venv, which
is not tracked / is rsync-excluded). Wrap the per-device probe in try/except so a
faulted card is reported as "unknown" instead of crashing the import. The card is
never *used* for compute (we pin good cards via CUDA_VISIBLE_DEVICES); only this
cosmetic name-enumeration touched it.

Idempotent. Usage:  python scripts/_patch_vllm_nvml.py [path/to/vllm/platforms/cuda.py]
"""
import sys
from pathlib import Path

DEFAULT = ".venv/lib/python3.10/site-packages/vllm/platforms/cuda.py"
MARK = "faulted GPU tolerated"

NEEDLE = (
    "    @classmethod\n"
    "    def _get_physical_device_name(cls, device_id: int = 0) -> str:\n"
    "        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)\n"
    "        return pynvml.nvmlDeviceGetName(handle)\n"
)
PATCHED = (
    "    @classmethod\n"
    "    def _get_physical_device_name(cls, device_id: int = 0) -> str:\n"
    "        try:\n"
    "            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)\n"
    "            return pynvml.nvmlDeviceGetName(handle)\n"
    f"        except Exception:  # {MARK}: a card off the bus must not abort import\n"
    "            return \"unknown\"\n"
)


def main() -> int:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    src = target.read_text()
    if MARK in src:
        print("ALREADY_PATCHED")
        return 0
    if NEEDLE not in src:
        print("NEEDLE_NOT_FOUND -- vllm cuda.py layout changed; inspect manually")
        return 1
    target.write_text(src.replace(NEEDLE, PATCHED))
    print(f"PATCHED_OK -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
