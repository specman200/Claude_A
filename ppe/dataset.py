"""Capture mode: keep frames worth training on, and pre-label them.

A station that has been running for a week has already seen every awkward
pose, every bad light and every near-miss the model gets wrong — and thrown
all of it away. This keeps the interesting frames, as raw images plus the
detections the model made at the time, in the layout an annotation tool
expects:

    <dir>/images/000042_miss_cam1.jpg
    <dir>/labels/000042_miss_cam1.txt     one "cls cx cy w h" line per box
    <dir>/data.yaml                       the class list, in index order
    <dir>/captures.csv                    what fired, and when

The labels are the model's own opinion, not ground truth. They are there so
an annotator corrects boxes instead of drawing them, which is the difference
between an afternoon and a week. Anything already right is already done.

Writing happens on its own thread behind a bounded queue. This station drives
a motor: a JPEG encode on the cycle that decides whether the machine may run
is not a trade worth making, so a full queue drops the frame and says so
rather than ever holding the loop up.

Note what this is. These are photographs of identifiable people at work,
written unencrypted to local disk. Turn it on deliberately, for as long as
you need it, and handle what comes out the way the rest of the site's
personal data is handled.
"""

from __future__ import annotations

import csv
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from .capture import Frame
from .detector import Detection
from .tower import ClassState, Status

log = logging.getLogger(__name__)

# What can ask for a frame. Names are config-facing, so they are the ones a
# person types, not the ones the code finds convenient.
INTERVAL, VIOLATION, MOTOR_START, MOTOR_STOP, MISS = (
    "interval", "violation", "motor_start", "motor_stop", "miss",
)
TRIGGERS = (INTERVAL, VIOLATION, MOTOR_START, MOTOR_STOP, MISS)

WHY = {
    INTERVAL: "a timer, for the ordinary frames that make a dataset general",
    VIOLATION: "the station called a violation — right or wrong, worth seeing",
    MOTOR_START: "the moment a worker was judged compliant enough to cut",
    MOTOR_STOP: "the moment the machine was taken away",
    MISS: "a required item just went out of sight — the frames the model failed",
}


@dataclass(slots=True)
class Shot:
    """One frame on its way to disk."""

    name: str
    image: object            # BGR ndarray, already copied off the capture buffer
    labels: list[str]
    row: list[object]


class DatasetRecorder:
    """Watches the cycle, keeps the frames worth keeping."""

    def __init__(self, cfg, classes: list[str], queue_depth: int = 8) -> None:
        self.cfg = cfg
        self.triggers = {t for t in cfg.triggers if t in TRIGGERS}
        self.index = {name: i for i, name in enumerate(classes)}
        self.saved = 0
        self.dropped = 0
        self.full = False        # max_images reached

        root = Path(cfg.dir)
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        self.root = root
        (root / "data.yaml").write_text(
            "# Written by ppe.dataset. Labels are the model's own detections,\n"
            "# not ground truth: correct them before training on them.\n"
            "path: .\ntrain: images\nval: images\n"
            f"nc: {len(classes)}\n"
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(classes))
        )
        fresh = not (root / "captures.csv").exists() or \
            (root / "captures.csv").stat().st_size == 0
        self._fh = (root / "captures.csv").open("a", newline="")
        self._csv = csv.writer(self._fh)
        if fresh:
            self._csv.writerow(["image", "wall_time", "elapsed_s", "camera",
                                "trigger", "status", "motor", "boxes"])

        self._q: queue.Queue = queue.Queue(maxsize=queue_depth)
        self._writer = threading.Thread(target=self._drain, name="dataset", daemon=True)
        self._writer.start()

        self._t0: float | None = None
        self._last_save = float("-inf")
        self._last_interval = float("-inf")
        self._was_status: Status | None = None
        self._was_running = False
        self._was_short: set[str] = set()
        log.info("capture mode on: %s -> %s (max %d images)",
                 ", ".join(sorted(self.triggers)) or "nothing", root, cfg.max_images)

    # -- deciding ----------------------------------------------------------
    def observe(
        self,
        pairs: list[tuple[Frame, list[Detection]]],
        status: Status,
        classes: list[ClassState],
        grinder_on: bool,
        t: float,
    ) -> None:
        """Fold one cycle in, and keep its frames if something asked for them."""
        if self._t0 is None:
            self._t0 = t
        fired = self._fired(status, classes, grinder_on, t)
        # Edges are tracked whether or not the frame is kept, so a shot that
        # the rate limit refuses does not leave the next one thinking the
        # violation only just started.
        self._was_status, self._was_running = status, grinder_on
        if not fired or self.full:
            return
        if t - self._last_save < self.cfg.min_gap_sec:
            return
        if self.saved >= self.cfg.max_images:
            self.full = True
            log.warning("capture mode: %d images reached, stopping", self.cfg.max_images)
            return
        self._last_save = t
        for frame, dets in pairs:
            self._keep(frame, dets, fired, status, grinder_on, t)

    def _fired(self, status, classes, grinder_on, t) -> str:
        """The first trigger that wants this cycle, or "" for none."""
        if INTERVAL in self.triggers and t - self._last_interval >= self.cfg.interval_sec:
            self._last_interval = t
            return INTERVAL
        if MOTOR_START in self.triggers and grinder_on and not self._was_running:
            return MOTOR_START
        if MOTOR_STOP in self.triggers and self._was_running and not grinder_on:
            return MOTOR_STOP
        if (VIOLATION in self.triggers and status is Status.VIOLATION
                and self._was_status is not Status.VIOLATION):
            return VIOLATION
        if MISS in self.triggers and self._opened_a_gap(classes):
            return MISS
        return ""

    def _opened_a_gap(self, classes: list[ClassState]) -> bool:
        """Did a required item just drop out of sight?

        The edge, not the state: a class that has been missing for a minute
        is one frame's worth of information, and sampling it every cycle
        would bury the dataset in the same picture. What is worth keeping is
        the moment it went.
        """
        short = {
            c.name for c in classes
            if c.required and c.available and c.seen < c.need
        }
        opened = bool(short - self._was_short)
        self._was_short = short
        return opened

    # -- keeping -----------------------------------------------------------
    def _keep(self, frame, dets, trigger, status, grinder_on, t) -> None:
        name = f"{self.saved:06d}_{trigger}_cam{frame.index}"
        height, width = frame.image.shape[:2]
        labels = []
        for d in dets:
            i = self.index.get(d.name)
            if i is None:
                continue          # not a class this dataset is labelled for
            x1, y1, x2, y2 = d.xyxy
            cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
            bw, bh = abs(x2 - x1) / width, abs(y2 - y1) / height
            labels.append(f"{i} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        row = [f"{name}.jpg", f"{time.time():.3f}", f"{t - self._t0:.2f}",
               frame.index, trigger, status.value, int(grinder_on), len(labels)]
        # Copied off the capture buffer here rather than on the writer thread:
        # the camera owns that array and is free to have moved on by the time
        # a queued shot is encoded.
        shot = Shot(name, frame.image.copy(), labels, row)
        try:
            self._q.put_nowait(shot)
            self.saved += 1
        except queue.Full:
            self.dropped += 1
            log.warning("capture mode: writer behind, dropped a frame (%d so far)",
                        self.dropped)

    def _drain(self) -> None:
        while True:
            shot = self._q.get()
            if shot is None:
                return
            try:
                cv2.imwrite(str(self.root / "images" / f"{shot.name}.jpg"), shot.image,
                            [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
                if self.cfg.labels:
                    (self.root / "labels" / f"{shot.name}.txt").write_text(
                        "\n".join(shot.labels) + ("\n" if shot.labels else ""))
                self._csv.writerow(shot.row)
                self._fh.flush()
            except Exception:       # a full disk must not take the station down
                log.exception("capture mode: could not write %s", shot.name)

    def close(self) -> None:
        self._q.put(None)
        self._writer.join(timeout=5.0)
        if not self._fh.closed:
            self._fh.close()
        log.info("capture mode: %d images written, %d dropped", self.saved, self.dropped)
