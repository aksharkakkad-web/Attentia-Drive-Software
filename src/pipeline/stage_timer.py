"""Per-stage timing instrumentation for the pipeline loop.

Records per-stage wall-clock durations and periodically logs p50/p95 so
we can tell where latency goes without wiring a profiler. Target for the
desk MVP: p95 end-to-end frame cost <80 ms.

Not a hardware interface — lives in src/pipeline/, not src/pipeline/interfaces/.
Safe to use from both pipeline code and interface implementations.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Dict, List

logger = logging.getLogger(__name__)


class StageTimer:
    """Collects per-stage durations and periodically logs p50/p95.

    Usage:
        t = StageTimer(report_interval_s=2.0)
        with t.measure('capture'):
            ...
        t.maybe_report()

    Window caps at 600 samples per stage (~20s at 30 fps) to bound memory.
    Reporting is best-effort and non-blocking.
    """

    _MAX_SAMPLES = 600

    def __init__(self, report_interval_s: float = 2.0) -> None:
        self._interval = report_interval_s
        self._samples: Dict[str, List[float]] = {}
        self._last_report = time.monotonic()

    @contextmanager
    def measure(self, stage: str):
        start = time.monotonic()
        try:
            yield
        finally:
            dur_ms = (time.monotonic() - start) * 1000.0
            bucket = self._samples.setdefault(stage, [])
            bucket.append(dur_ms)
            if len(bucket) > self._MAX_SAMPLES:
                del bucket[: len(bucket) - self._MAX_SAMPLES]

    def maybe_report(self) -> None:
        now = time.monotonic()
        if now - self._last_report < self._interval:
            return
        self._last_report = now
        if not self._samples:
            return
        parts = []
        for stage, samples in self._samples.items():
            if not samples:
                continue
            ordered = sorted(samples)
            p50 = ordered[len(ordered) // 2]
            p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
            parts.append(f"{stage}: p50={p50:.1f}ms p95={p95:.1f}ms (n={len(samples)})")
        logger.info("stage timings — %s", " | ".join(parts))
        # Reset window each report so we see recent behaviour, not cumulative.
        self._samples.clear()
