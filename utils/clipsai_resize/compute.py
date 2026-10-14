"""The pieces of clipsai.utils the resize package needs, without the clipsai imports.

Clips AI reaches into clipsai.utils.pytorch for device selection and free-memory
figures and into clipsai.utils.conversions for one unit conversion. Those modules pull
in psutil and pynvml, so the three functions actually used are reimplemented here.
"""
import ctypes
import os
import sys

import torch

GIBI = 1024**3
_DEVICES = ("cpu", "cuda", "mps")
_MEMORY_FALLBACK = 2 * GIBI  # only sizes face-detection batches, so a floor is enough


def bytes_to_gibibytes(n: int) -> float:
    """Bytes as gibibytes."""
    return n / GIBI


def compute_device() -> str:
    """The best torch device available on this machine."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def require_compute_device(device: str) -> None:
    """Raise unless `device` is a torch device this machine can actually compute on."""
    if device not in _DEVICES:
        raise ValueError(f"Compute device must be one of {_DEVICES}, not {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Device 'cuda' is not available on this machine")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Device 'mps' is not available on this machine")


def free_cpu_memory() -> int:
    """Free physical memory in bytes, best effort."""
    try:
        import psutil  # not a requirement; used when it happens to be installed
    except ImportError:
        pass
    else:
        return int(psutil.virtual_memory().available)

    if sys.platform == "win32":
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return _MEMORY_FALLBACK

    if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return _MEMORY_FALLBACK
