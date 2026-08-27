"""End-to-end latency accounting and hotspot profiling.

A :class:`Cycle` follows one frame from the camera grab to the relay write,
stamping each stage as it passes. :class:`Metrics` keeps rolling percentiles
for the HUD and appends every cycle to CSV. :class:`Profiler` wraps cProfile
so ``--profile`` prints the real hotspots instead of guesses.
"""

from __future__ import annotations

import cProfile
import csv
import io
import pstats
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

# Stages of one end-to-end cycle, in the order they occur.
STAGES = ("wait", "preprocess", "inference", "postprocess", "logic", "relay")

# perf_counter is monotonic and the highest resolution clock available.
now = time.perf_counter


class Cycle:
    """Stage timings for a single frame, in milliseconds."""

    __slots__ = ("t0", "_last", "stages", "total")

    def __init__(self, captured_at: float) -> None:
        self.t0 = captured_at          # when the frame left the camera
        self._last = captured_at
        self.stages: dict[str, float] = {}
        self.total = 0.0

    def stamp(self, stage: str) -> float:
        """Record the time since the previous stamp against ``stage``."""
        t = now()
        ms = (t - self._last) * 1000.0
        self._last = t
        self.stages[stage] = self.stages.get(stage, 0.0) + ms
        return ms

    def merge(self, timings: dict[str, float]) -> None:
        """Fold in stage durations measured elsewhere, e.g. a batched call."""
        self.stages.update(timings)
        self._last = now()

    def finish(self) -> float:
        """Close the cycle; returns end-to-end latency in ms."""
        self.total = (now() - self.t0) * 1000.0
        return self.total


class Metrics:
    """Thread-safe rolling stats plus an optional CSV trace."""

    def __init__(self, window: int = 300, csv_path: str | Path | None = None) -> None:
        self._win = max(1, window)
        self._series: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._writer = None
        self._fh = None
        if csv_path:
            path = Path(csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists() or path.stat().st_size == 0
            self._fh = path.open("a", newline="")
            self._writer = csv.writer(self._fh)
            if new:
                self._writer.writerow(["wall_time", "camera", *STAGES, "end_to_end_ms"])

    # -- recording ---------------------------------------------------------
    def add(self, name: str, value: float) -> None:
        with self._lock:
            self._series.setdefault(name, deque(maxlen=self._win)).append(value)

    def record(self, camera: str, cycle: Cycle) -> None:
        """Fold a finished cycle into the rolling stats and the CSV trace."""
        with self._lock:
            for stage, ms in cycle.stages.items():
                self._series.setdefault(stage, deque(maxlen=self._win)).append(ms)
            self._series.setdefault("end_to_end", deque(maxlen=self._win)).append(cycle.total)
            if self._writer is not None:
                self._writer.writerow(
                    [f"{time.time():.3f}", camera]
                    + [f"{cycle.stages.get(s, 0.0):.2f}" for s in STAGES]
                    + [f"{cycle.total:.2f}"]
                )

    @contextmanager
    def measure(self, name: str):
        """Time an arbitrary block, e.g. ``with metrics.measure('render'): ...``"""
        t = now()
        try:
            yield
        finally:
            self.add(name, (now() - t) * 1000.0)

    def flush(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = self._writer = None

    # -- reporting ---------------------------------------------------------
    def stats(self, name: str) -> dict[str, float]:
        with self._lock:
            data = sorted(self._series.get(name, ()))
        if not data:
            return {"n": 0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
        return {
            "n": len(data),
            "p50": _pct(data, 0.50),
            "p95": _pct(data, 0.95),
            "max": data[-1],
            "mean": sum(data) / len(data),
        }

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            names = list(self._series)
        return {n: self.stats(n) for n in names}

    def report(self) -> str:
        """One-screen breakdown: which stage owns the end-to-end budget."""
        snap = self.snapshot()
        total = snap.get("end_to_end", {}).get("p50", 0.0) or 1.0
        lines = [f"{'stage':<12}{'p50 ms':>9}{'p95 ms':>9}{'max ms':>9}{'share':>8}{'n':>7}"]
        order = [s for s in STAGES if s in snap] + [
            k for k in snap if k not in STAGES and k != "end_to_end"
        ]
        for name in order:
            s = snap[name]
            share = f"{100 * s['p50'] / total:5.1f}%" if name in STAGES else "     -"
            lines.append(
                f"{name:<12}{s['p50']:9.2f}{s['p95']:9.2f}{s['max']:9.2f}{share:>8}{int(s['n']):7d}"
            )
        if "end_to_end" in snap:
            s = snap["end_to_end"]
            lines.append("-" * 54)
            lines.append(
                f"{'end-to-end':<12}{s['p50']:9.2f}{s['p95']:9.2f}{s['max']:9.2f}"
                f"{'100.0%':>8}{int(s['n']):7d}"
            )
        return "\n".join(lines)


def _pct(sorted_data: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already sorted list."""
    if len(sorted_data) == 1:
        return sorted_data[0]
    pos = q * (len(sorted_data) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (pos - lo)


class Profiler:
    """cProfile across every thread that opts in, merged into one report.

    cProfile is per-thread, and the work worth profiling happens on the
    pipeline thread — so each thread starts its own and the results are merged
    at the end.
    """

    def __init__(self, out: str | Path = "logs/profile.prof", top: int = 20) -> None:
        self.out = Path(out)
        self.top = top
        self._active: dict[int, cProfile.Profile] = {}
        self._done: list[cProfile.Profile] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        """Begin profiling the calling thread."""
        prof = cProfile.Profile()
        with self._lock:
            self._active[threading.get_ident()] = prof
        prof.enable()

    def stop_thread(self) -> None:
        """End profiling on the calling thread, keeping its samples.

        A profile must be disabled by the thread that enabled it, so every
        profiled thread calls this on its way out.
        """
        with self._lock:
            prof = self._active.pop(threading.get_ident(), None)
        if prof is None:
            return
        prof.disable()
        with self._lock:
            self._done.append(prof)

    def stop(self) -> str:
        """Stop the calling thread, merge every thread's samples, and report."""
        self.stop_thread()
        with self._lock:
            profs, self._done = self._done, []
        if not profs:
            return ""

        self.out.parent.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        stats = pstats.Stats(*profs, stream=buf)
        stats.dump_stats(self.out)
        # tottime first: that is where the CPU actually goes.
        buf.write(f"\n=== hotspots: {self.top} functions by self time ===\n")
        stats.sort_stats("tottime").print_stats(self.top)
        buf.write(f"\n=== {self.top} functions by cumulative time ===\n")
        stats.sort_stats("cumulative").print_stats(self.top)
        return f"profile written to {self.out}\n{buf.getvalue()}"

    def __enter__(self) -> Profiler:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        text = self.stop()
        if text:
            print(text)
