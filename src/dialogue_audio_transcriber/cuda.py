from __future__ import annotations

import ctypes
import logging
from importlib.util import find_spec
from pathlib import Path

logger = logging.getLogger(__name__)

_NVIDIA_PACKAGES = ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc")


def preload_cuda_libraries() -> None:
    """Load pip-installed CUDA libs so ctranslate2 can find libcublas.so.12."""
    libraries: list[Path] = []
    for package in _NVIDIA_PACKAGES:
        spec = find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            continue
        lib_dir = Path(spec.submodule_search_locations[0]) / "lib"
        if lib_dir.is_dir():
            libraries.extend(sorted(lib_dir.glob("lib*.so.*")))

    if not libraries:
        logger.warning("NVIDIA CUDA libraries were not found in the virtualenv")
        return

    pending = libraries
    for _ in range(4):
        still_pending: list[Path] = []
        for path in pending:
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                still_pending.append(path)
        if len(still_pending) == len(pending):
            break
        pending = still_pending

    if pending:
        logger.warning(
            "Could not preload CUDA libraries: %s",
            ", ".join(path.name for path in pending),
        )
