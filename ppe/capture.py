"""Threaded camera capture: one grabber per camera, latest frame wins.

The grabber never blocks the pipeline and the pipeline never blocks the
grabber. Frames the detector was too busy to take are dropped rather than
queued, which is what keeps the displayed feed live instead of drifting
further behind real time the longer the app runs.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CameraCfg
from .latency import now

log = logging.getLogger(__name__)

# MSMF's hardware-transform negotiation can stall opening a UVC webcam for
# minutes rather than failing outright — indistinguishable from a hang until
# you have waited long enough to doubt it ever was one. Measured on a
# Logitech C920: DirectShow opens in under a second but caps at 10 fps at
# 720p because it never actually applies an MJPG request the driver silently
# ignores; MSMF with hardware transforms off opens in under a second AND
# reaches the full 30 fps, on the same camera and port. setdefault so an
# operator who has already set this — or is relying on the hardware
# transform on purpose — is never overridden. Read once, lazily, on first
# MSMF backend use, so setting it here at import time is early enough; it is
# inert on Linux, where MSMF does not exist.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

_APIS = {
    "any": cv2.CAP_ANY,
    "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),        # Linux
    "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),       # Windows, the older/broader backend
    "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),         # Windows, the modern default
    "ffmpeg": getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY),
    "gstreamer": getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY),
}


@dataclass(frozen=True, slots=True)
class Frame:
    index: int          # which camera
    seq: int            # monotonically increasing per camera
    image: np.ndarray   # BGR, never mutated downstream
    ts: float           # perf_counter at grab — the start of end-to-end latency


class Camera(threading.Thread):
    """Reads one source as fast as it will go, keeping only the newest frame."""

    def __init__(self, index: int, cfg: CameraCfg) -> None:
        super().__init__(name=f"camera-{index}", daemon=True)
        self.index = index
        self.cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._frame: Frame | None = None
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self._seq = 0
        self.fps = 0.0
        self.connected = False

    # -- lifecycle ---------------------------------------------------------
    def _open(self) -> bool:
        src = self.cfg.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src, _APIS.get(self.cfg.api, cv2.CAP_ANY))
        if not cap.isOpened():
            cap.release()
            # OpenCV's own reason (no such device, permission denied, device
            # busy...) goes to its native logger, not this one — surface it
            # here rather than leaving "unavailable" as the only clue.
            log.debug(
                "%s: cv2.VideoCapture(%r, api=%s) did not open "
                "(run with -v for OpenCV's own diagnostic, or `python -m ppe.camcheck`)",
                self.cfg.name, src, self.cfg.api,
            )
            return False
        # FOURCC first: on DirectShow/MSMF the pixel format changes which
        # resolutions and frame rates the driver will negotiate at all, so
        # setting it after width/height can silently lock in the wrong mode.
        # Without this most UVC webcams (including the Logitech C920) hand
        # back raw YUY2 by default — 1280x720@30 raw is ~55 MB/s, more than
        # USB 2.0 carries for ONE camera, let alone two on a shared hub.
        # MJPG is the same sensor data, compressed in the camera itself.
        if self.cfg.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.fourcc))
        # A 1-frame driver buffer is the difference between live and lagging.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.cfg.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        if self.cfg.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        if self.cfg.fps:
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        self._cap = cap
        self.connected = True
        got_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        got_tag = got_fourcc.to_bytes(4, "little").decode("ascii", "replace").strip()
        log.info(
            "%s open: %dx%d @ %.0f fps (%s)%s",
            self.cfg.name,
            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            cap.get(cv2.CAP_PROP_FPS),
            got_tag or "?",
            "" if not self.cfg.fourcc or got_tag == self.cfg.fourcc
            else f"  <- asked for {self.cfg.fourcc}, driver ignored it",
        )
        return True

    def run(self) -> None:
        backoff = 0.5
        last = now()
        # A live camera paces itself inside read(); a file or a free-running
        # source does not, and would burn a core decoding frames nobody wants.
        period = 1.0 / self.cfg.fps if self.cfg.fps else 0.0
        while not self._halt.is_set():
            if self._cap is None:
                if not self._open():
                    self.connected = False
                    log.warning("%s unavailable, retrying in %.1fs", self.cfg.name, backoff)
                    self._halt.wait(backoff)
                    backoff = min(backoff * 2, 5.0)
                    continue
                backoff = 0.5

            if period:
                behind = period - (now() - last)
                if behind > 0.001:
                    self._halt.wait(behind)

            ok, image = self._cap.read()
            ts = now()
            if not ok:
                log.warning("%s read failed, reconnecting", self.cfg.name)
                self._release()
                continue

            self._seq += 1
            with self._lock:
                self._frame = Frame(self.index, self._seq, image, ts)

            dt = ts - last
            last = ts
            if dt > 0:  # smoothed so the HUD reads steady
                self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1.0 / dt

    def _release(self) -> None:
        self.connected = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=2.0)
        self._release()

    # -- consumption -------------------------------------------------------
    def latest(self) -> Frame | None:
        """Newest frame, or ``None`` if nothing has arrived yet."""
        with self._lock:
            return self._frame


class CameraSet:
    """The cameras as one unit: start, stop, and sample them together."""

    def __init__(self, cfgs: list[CameraCfg]) -> None:
        self.cameras = [Camera(i, c) for i, c in enumerate(cfgs)]

    def __len__(self) -> int:
        return len(self.cameras)

    def start(self) -> CameraSet:
        for cam in self.cameras:
            cam.start()
        return self

    def stop(self) -> None:
        for cam in self.cameras:
            cam.stop()

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """Wait until every camera has produced at least one frame."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(c.latest() is not None for c in self.cameras):
                return True
            time.sleep(0.05)
        return False

    def sample(self) -> list[Frame | None]:
        """One frame per camera, taken as close together as we can manage."""
        return [c.latest() for c in self.cameras]
