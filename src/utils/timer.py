"""Context manager timer for profiling pipeline stages.

Measures elapsed wall-clock time in milliseconds for any code block.

Phase: 1-2 (active).
"""

import time


class Timer:
    """Context manager that measures elapsed time in milliseconds.

    Usage:
        with Timer() as t:
            result = detector.detect(frame)
        print(t.elapsed_ms)
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return (self._end - self._start) * 1000.0
