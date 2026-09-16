from __future__ import annotations

import threading
import time
from dataclasses import dataclass

try:
    import pynvml
except Exception:  # pragma: no cover
    pynvml = None


@dataclass
class GpuSnapshot:
    timestamp: float
    used_mb: float
    total_mb: float
    utilization_gpu: int | None = None
    utilization_memory: int | None = None


class GpuMonitor:
    def __init__(self, gpu_index: int = 0, interval_sec: float = 0.2) -> None:
        self.gpu_index = gpu_index
        self.interval_sec = interval_sec
        self.snapshots: list[GpuSnapshot] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None

    def __enter__(self) -> "GpuMonitor":
        if pynvml is None:
            return self
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self.snapshots.append(
                    GpuSnapshot(
                        timestamp=time.time(),
                        used_mb=mem.used / 1024**2,
                        total_mb=mem.total / 1024**2,
                        utilization_gpu=int(util.gpu),
                        utilization_memory=int(util.memory),
                    )
                )
            except Exception:
                pass
            time.sleep(self.interval_sec)

    @property
    def peak_used_mb(self) -> float | None:
        if not self.snapshots:
            return None
        return max(s.used_mb for s in self.snapshots)

    @property
    def mean_gpu_utilization(self) -> float | None:
        values = [s.utilization_gpu for s in self.snapshots if s.utilization_gpu is not None]
        if not values:
            return None
        return sum(values) / len(values)

