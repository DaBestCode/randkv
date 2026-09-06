"""Thread-safe, dependency-free policy metrics."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True, slots=True)
class PolicyStats:
    eviction_events: int
    positions_considered: int
    positions_evicted: int
    prompt_positions_retained: int


class PolicyMetrics:
    """Mutable counters owned by one policy instance."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._eviction_events = 0
        self._positions_considered = 0
        self._positions_evicted = 0
        self._prompt_positions_retained = 0

    def record(self, *, before: int, after: int, prompt_retained: int) -> None:
        with self._lock:
            self._eviction_events += 1
            self._positions_considered += before
            self._positions_evicted += before - after
            self._prompt_positions_retained += prompt_retained

    def snapshot(self) -> PolicyStats:
        with self._lock:
            return PolicyStats(
                eviction_events=self._eviction_events,
                positions_considered=self._positions_considered,
                positions_evicted=self._positions_evicted,
                prompt_positions_retained=self._prompt_positions_retained,
            )
