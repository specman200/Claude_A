"""The detection loop: newest frames in, tower-light state and overlays out."""

from __future__ import annotations

import copy
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .capture import CameraSet, Frame
from .config import Config
from .detector import Detection, Detector
from .latency import Cycle, Metrics, Profiler, now
from .subject import Focus, focus
from .tower import ClassState, ComplianceMonitor, Status, make_tower

log = logging.getLogger(__name__)

STALE_AFTER = 1.5   # seconds without a frame before a camera counts as down
OFFLINE_PERIOD = 0.2  # how often to re-evaluate while no camera is delivering


@dataclass(slots=True)
class Result:
    """One cycle's output — everything the UI needs to draw a full update."""

    status: Status
    classes: list[ClassState]
    detections: list[list[Detection]] = field(default_factory=list)
    ignored: list[list[Detection]] = field(default_factory=list)   # off-subject
    subjects: list[Detection | None] = field(default_factory=list)  # who is checked
    seqs: list[int] = field(default_factory=list)
    latency_ms: float = 0.0
    infer_fps: float = 0.0
    tower_ok: bool = False
    missing: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    banned: list[str] = field(default_factory=list)


class Pipeline(threading.Thread):
    """Runs detection as fast as frames arrive, never queueing stale work."""

    def __init__(
        self,
        cfg: Config,
        cameras: CameraSet,
        on_result: Callable[[Result], None] | None = None,
        profiler: Profiler | None = None,
    ) -> None:
        super().__init__(name="pipeline", daemon=True)
        self.cfg = cfg
        self.cameras = cameras
        self.on_result = on_result
        self.profiler = profiler
        self.metrics = Metrics(cfg.telemetry.window, cfg.telemetry.csv or None)
        self.detector = Detector(cfg.model, cfg.ppe)
        self.monitor = ComplianceMonitor(cfg.ppe, self.detector.missing)
        self.tower = make_tower(cfg.tower)

        self._halt = threading.Event()
        self._swap = threading.Lock()
        self._focus: list[Focus] = [Focus() for _ in range(len(cameras))]
        self._seqs: list[int] = [-1] * len(cameras)
        self.infer_fps = 0.0
        self.cycles = 0
        self.result: Result | None = None

    # -- loop --------------------------------------------------------------
    def run(self) -> None:
        if self.profiler is not None:
            self.profiler.start()  # cProfile is per-thread; opt this one in
        last_print = now()
        last_cycle = 0.0
        last_offline = 0.0
        while not self._halt.is_set():
            frames = self.cameras.sample()
            fresh = [
                f for i, f in enumerate(frames) if f is not None and f.seq != self._seqs[i]
            ]
            if not fresh:
                # Poll cheaply for the next frame, but re-evaluate the offline
                # state at a human rate rather than a thousand times a second.
                t = now()
                if t - last_offline >= OFFLINE_PERIOD and self._all_stale(frames):
                    last_offline = t
                    self._go_offline()
                time.sleep(0.001)  # nothing new; yield rather than spin
                continue

            self._cycle(fresh)
            self.cycles += 1

            t = now()
            if last_cycle:
                dt = t - last_cycle
                self.infer_fps = 0.9 * self.infer_fps + 0.1 / dt if self.infer_fps else 1 / dt
            last_cycle = t

            every = self.cfg.telemetry.print_every
            if every and t - last_print >= every:
                last_print = t
                print(f"\n[{time.strftime('%H:%M:%S')}] {self.infer_fps:.1f} infer fps")
                print(self.metrics.report(), flush=True)

        self._shutdown()

    def _go_offline(self) -> None:
        """No camera is delivering: clear the overlays and hold the lamp amber.

        Boxes from the last good frame would otherwise sit on a dead feed,
        which reads as a live detection.
        """
        self._focus = [Focus() for _ in self._focus]
        self._publish(self.monitor.degrade())

    def _all_stale(self, frames: list[Frame | None]) -> bool:
        t = now()
        return all(f is None or (t - f.ts) > STALE_AFTER for f in frames)

    def _cycle(self, fresh: list[Frame]) -> None:
        # One cycle per frame, all started at their own capture instant, so
        # end-to-end latency is measured per camera and not averaged away.
        cycles = {f.index: Cycle(f.ts) for f in fresh}
        for f in fresh:
            cycles[f.index].stamp("wait")  # camera grab -> picked up here

        with self._swap:
            dets, timings = self.detector.detect([f.image for f in fresh])
        for cyc in cycles.values():
            cyc.merge(timings)

        for f, d in zip(fresh, dets, strict=True):
            # Each camera picks its own subject: with two views of one cell,
            # a single global "largest" would silently discard the other view.
            self._focus[f.index] = focus(d, self.cfg.ppe.subject, self.cfg.ppe.containment)
            self._seqs[f.index] = f.seq

        flat = [det for f in self._focus for det in f.accepted]
        with self._swap:
            status = self.monitor.update(flat)
        for cyc in cycles.values():
            cyc.stamp("logic")

        self.tower.apply(status)
        for cyc in cycles.values():
            cyc.stamp("relay")
            cyc.finish()

        worst = max(cycles.values(), key=lambda c: c.total)
        for idx, cyc in cycles.items():
            self.metrics.record(self.cameras.cameras[idx].cfg.name, cyc)

        self._publish(status, worst.total)

    def _publish(self, status: Status, latency_ms: float = 0.0) -> None:
        result = Result(
            status=status,
            classes=[copy.copy(c) for c in self.monitor.classes],
            detections=[list(f.accepted) for f in self._focus],
            ignored=[list(f.rejected) for f in self._focus],
            subjects=[f.subject for f in self._focus],
            seqs=list(self._seqs),
            latency_ms=latency_ms,
            infer_fps=self.infer_fps,
            tower_ok=getattr(self.tower, "connected", False),
            missing=self.monitor.missing(),
            unavailable=self.monitor.unavailable(),
            banned=self.monitor.banned(),
        )
        self.result = result
        if self.on_result is not None:
            self.on_result(result)

    def reconfigure(self) -> None:
        """Adopt an edited class list without restarting the model or cameras."""
        with self._swap:
            missing = self.detector.set_classes(self.cfg.ppe)
            self.monitor = ComplianceMonitor(self.cfg.ppe, missing)

    # -- lifecycle ---------------------------------------------------------
    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=5.0)

    def _shutdown(self) -> None:
        if self.profiler is not None:
            self.profiler.stop_thread()
        self.tower.close()
        self.metrics.flush()
        self.metrics.close()
        log.info("pipeline stopped after %d cycles", self.cycles)
