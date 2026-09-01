"""The detection loop: newest frames in, tower-light state and overlays out."""

from __future__ import annotations

import copy
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .annunciator import Annunciator
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
    # The decision behind the lamp, for the debug view: what this cycle
    # actually said, what is waiting to be confirmed, and for how long.
    raw: Status = Status.DEGRADED
    candidate: Status = Status.DEGRADED
    candidate_age: float = 0.0
    confirm_wait: float = 0.0
    audio_due: float | None = None
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
        on_ready: Callable[[], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        profiler: Profiler | None = None,
    ) -> None:
        super().__init__(name="pipeline", daemon=True)
        self.cfg = cfg
        self.cameras = cameras
        self.on_result = on_result
        self.on_ready = on_ready    # called once the model has loaded
        self.on_error = on_error   # called once, instead, if loading fails
        self.profiler = profiler
        self.metrics = Metrics(cfg.telemetry.window, cfg.telemetry.csv or None)

        # The model can take seconds to load and warm up. Loading it here,
        # in the constructor, would block whoever creates the Pipeline —
        # including a UI that wants to show a window and live video before
        # the model is ready. So construction is cheap; loading happens in
        # run(), on the pipeline thread, where it belongs.
        self.detector: Detector | None = None
        self.monitor: ComplianceMonitor | None = None
        self.tower = None
        self.annunciator = Annunciator(
            cfg.audio.file,
            cfg.audio.grace_sec,
            cfg.audio.repeat_sec,
            cfg.base_dir,
            # Debugging a station should not mean listening to it nag.
            mute=cfg.ui.mode == "debug",
        )
        self.ready = threading.Event()
        self.error: Exception | None = None

        self._halt = threading.Event()
        self._swap = threading.Lock()
        self._focus: list[Focus] = [Focus() for _ in range(len(cameras))]
        self._seqs: list[int] = [-1] * len(cameras)
        self._turn = 0  # round-robin cursor when cameras are served one at a time
        self.infer_fps = 0.0
        self.cycles = 0
        self.result: Result | None = None

    # -- loop --------------------------------------------------------------
    def _load(self) -> bool:
        """Build the model and the state machines that depend on it.

        Runs on the pipeline thread in normal use; test code that never
        starts the thread may call this directly. Returns False, with
        ``self.error`` set and ``on_error`` fired, rather than raising —
        a Pipeline that failed to load must say so and stop cleanly, not
        take the app down or hang forever looking like it is still busy.
        """
        try:
            self.detector = Detector(self.cfg.model, self.cfg.ppe)
            self.monitor = ComplianceMonitor(self.cfg.ppe, self.detector.missing)
            self.tower = make_tower(self.cfg.tower)
        except Exception as exc:  # noqa: BLE001 — reported via on_error, never silent
            log.exception("failed to load model %s", self.cfg.model.weights)
            self.error = exc
            if self.on_error is not None:
                self.on_error(exc)
            return False
        self.ready.set()
        if self.on_ready is not None:
            self.on_ready()
        return True

    def run(self) -> None:
        if self.profiler is not None:
            self.profiler.start()  # cProfile is per-thread; opt this one in
        if not self._load():
            self._shutdown()
            return
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

            self._cycle(self._take(fresh))
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

    def _take(self, fresh: list[Frame]) -> list[Frame]:
        """Which fresh frames to run this cycle.

        Batching only pays on a GPU. On a CPU two frames in one call cost about
        twice one frame *and* make each frame wait for the other's result, so
        serving one camera per cycle halves end-to-end latency at the same
        per-camera update rate. Cameras take turns so neither starves.

        The turn advances past whichever camera was served, including on a
        cycle where it was the only one with a fresh frame. Skipping the
        advance there let a camera be served without spending its turn, so it
        still held first claim on the next contested cycle and ran twice in a
        row — which penalised the *slower* camera precisely when the two were
        already uneven. A camera delivering half as often ended up inferred a
        third as often rather than half, compounding a bandwidth problem into
        a scheduling one.
        """
        if self.detector.batches:
            return fresh
        n = len(self.cameras)
        chosen = min(fresh, key=lambda f: (f.index - self._turn) % n)
        self._turn = (chosen.index + 1) % n
        return [chosen]

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

        # Per camera, not flattened: counts are the best single view, since
        # both cameras see the same two gloves on one worker.
        with self._swap:
            status = self.monitor.update([f.accepted for f in self._focus])
        for cyc in cycles.values():
            cyc.stamp("logic")

        self.tower.apply(status)
        self.annunciator.update(status)
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
            raw=self.monitor.raw,
            candidate=self.monitor.candidate,
            candidate_age=self.monitor.candidate_age(),
            confirm_wait=self.monitor.confirm_wait(),
            audio_due=self.annunciator.due_in(),
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
            if self.detector is None:
                return  # not loaded yet — the load under way will already see the edit
            missing = self.detector.set_classes(self.cfg.ppe)
            self.monitor = ComplianceMonitor(self.cfg.ppe, missing)

    # -- lifecycle ---------------------------------------------------------
    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=5.0)

    def _shutdown(self) -> None:
        if self.profiler is not None:
            self.profiler.stop_thread()
        self.annunciator.close()
        if self.tower is not None:
            self.tower.close()
        self.metrics.flush()
        self.metrics.close()
        log.info("pipeline stopped after %d cycles", self.cycles)
