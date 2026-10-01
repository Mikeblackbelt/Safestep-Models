"""
Hardware profiling and live device metrics.

`profile_hardware()` is a one-off snapshot used for planning (cores, free RAM,
free VRAM). `DeviceMonitor` samples utilisation over time for the dashboard.
GPU stats come from NVML when `nvidia-ml-py` is installed, otherwise from
torch (memory only, no utilisation).
"""

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import psutil
import torch

try:  # optional: richer GPU stats without creating a CUDA context
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    _NVML = True
except Exception:  # noqa: BLE001 - any failure means "no NVML"
    _NVML = False

GB = 1024 ** 3


@dataclass
class GPUInfo:
    index: int
    name: str
    total_gb: float
    free_gb: float
    util_pct: Optional[float] = None  # None when utilisation is unavailable


@dataclass
class HardwareProfile:
    device: str  # "cpu" | "cuda" | "mps"
    physical_cores: int
    logical_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpus: List[GPUInfo] = field(default_factory=list)


def resolve_device(requested: str = "auto") -> str:
    """Pick the best available backend, falling back to CPU."""
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if (requested == "cuda" and not cuda) or (requested == "mps" and not mps):
        return "cpu"
    return requested


def _gpu_stats() -> List[GPUInfo]:
    gpus: List[GPUInfo] = []
    if _NVML:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            gpus.append(GPUInfo(
                index=i,
                name=name.decode() if isinstance(name, bytes) else name,
                total_gb=mem.total / GB,
                free_gb=mem.free / GB,
                util_pct=float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu),
            ))
    elif torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            gpus.append(GPUInfo(
                index=i,
                name=torch.cuda.get_device_name(i),
                total_gb=total / GB,
                free_gb=free / GB,
            ))
    return gpus


def profile_hardware(device: str) -> HardwareProfile:
    vm = psutil.virtual_memory()
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or logical
    return HardwareProfile(
        device=device,
        physical_cores=physical,
        logical_cores=logical,
        ram_total_gb=vm.total / GB,
        ram_available_gb=vm.available / GB,
        gpus=_gpu_stats() if device == "cuda" else [],
    )


@dataclass
class DeviceSample:
    t: float
    cpu_pct: float
    ram_pct: float
    ram_used_gb: float
    gpus: List[GPUInfo]


@dataclass
class ProcessSample:
    cpu_pct: float  # may exceed 100 on multi-core
    rss_gb: float


class DeviceMonitor:
    """Rolling window of system and per-process utilisation samples."""

    def __init__(self, device: str, window: int = 240):
        self.device = device
        self.samples: Deque[DeviceSample] = deque(maxlen=window)
        self._procs: Dict[int, psutil.Process] = {}
        psutil.cpu_percent(interval=None)  # prime the counter; first reading is 0

    def sample(self) -> DeviceSample:
        vm = psutil.virtual_memory()
        s = DeviceSample(
            t=time.time(),
            cpu_pct=psutil.cpu_percent(interval=None),
            ram_pct=vm.percent,
            ram_used_gb=(vm.total - vm.available) / GB,
            gpus=_gpu_stats() if self.device == "cuda" else [],
        )
        self.samples.append(s)
        return s

    def process(self, pid: int) -> Optional[ProcessSample]:
        """CPU% and RSS of a worker process (None once it has exited)."""
        try:
            proc = self._procs.get(pid)
            if proc is None:
                proc = self._procs[pid] = psutil.Process(pid)
                proc.cpu_percent(interval=None)
            return ProcessSample(proc.cpu_percent(interval=None), proc.memory_info().rss / GB)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self._procs.pop(pid, None)
            return None
