#!/usr/bin/env python3
"""Capture a still from two USB cameras at the same moment.

    python capture.py                    # cameras 0 and 1, 2-second timer
    python capture.py --cameras 0 2      # pick the device indices yourself
    python capture.py --list             # show which indices actually open
    python capture.py --demo             # synthetic cameras, no hardware needed

Hit the Capture button or the spacebar to start the timer; hit either again
(or Esc) to call it off. The timer box next to the button sets the delay --
it starts at 2 seconds. Both frames are written to ./captures under one
shared timestamp, so a pair always stays a pair.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:  # only the UI needs Tk -- --list and the tests do not
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # pragma: no cover - depends on the Python build
    tk = ttk = None

try:
    import cv2
except ImportError:  # pragma: no cover - depends on the environment
    cv2 = None

log = logging.getLogger("dualcap")

DEFAULT_DELAY = 2.0       # seconds between the shutter press and the shot
MAX_DELAY = 60.0
PREVIEW_WIDTH = 480       # each live view is scaled to this many pixels wide
POLL_MS = 33              # preview refresh, ~30 fps
STALE_AFTER = 2.0         # a frame older than this means the camera stopped
FLASH_MS = 700            # how long "Saved" stays on the banner


class CaptureError(RuntimeError):
    """A shot could not be taken or written."""


# --- timer -----------------------------------------------------------------


def clamp_delay(value, default: float = DEFAULT_DELAY) -> float:
    """Read a delay off the timer box, falling back to *default* for junk."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(seconds):
        return default
    return max(0.0, min(MAX_DELAY, seconds))


@dataclass
class Countdown:
    """The delay between the shutter press and the shot."""

    duration: float
    deadline: float

    @classmethod
    def start(cls, duration: float, now: float) -> "Countdown":
        return cls(duration=duration, deadline=now + duration)

    def remaining(self, now: float) -> float:
        return max(0.0, self.deadline - now)

    def expired(self, now: float) -> bool:
        return now >= self.deadline


def countdown_text(remaining: float) -> str:
    """The big number on screen: whole seconds, rounded up."""
    return str(max(1, math.ceil(remaining)))


# --- where the shots land --------------------------------------------------


def shot_paths(outdir: Path, labels, when: datetime, suffix: str = ".jpg", exists=None):
    """One path per camera, sharing a stem so a pair reads as a pair.

    Two shots inside the same second get -2, -3, ... rather than overwriting.
    """
    exists = exists or Path.exists
    stem = when.strftime("%Y%m%d-%H%M%S")
    attempt = 1
    while True:
        tag = stem if attempt == 1 else f"{stem}-{attempt}"
        paths = [Path(outdir) / f"{tag}_{label}{suffix}" for label in labels]
        if not any(exists(path) for path in paths):
            return paths
        attempt += 1


# --- cameras ---------------------------------------------------------------


@dataclass
class Frame:
    """One image and the moment it was read."""

    image: object
    at: float


class UsbCamera:
    """One USB camera, opened lazily so an absent device is never fatal."""

    def __init__(self, index: int, label: str, width: int = 0, height: int = 0):
        self.index = index
        self.label = label
        self.width = width
        self.height = height
        self._cap = None

    def open(self) -> bool:
        cap = cv2.VideoCapture(self.index)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            cap.release()
            log.debug("camera %s did not open", self.index)
            return False
        self._cap = cap
        log.info("opened camera %s as %s", self.index, self.label)
        return True

    def read(self):
        if self._cap is None and not self.open():
            return None
        ok, image = self._cap.read()
        if not ok:
            log.warning("camera %s stopped delivering frames", self.index)
            self.close()
            return None
        return image

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SyntheticCamera:
    """Test pattern that stands in for a camera under --demo."""

    def __init__(self, label: str, width: int = 640, height: int = 480, hue: int = 0):
        self.label = label
        self.width = width
        self.height = height
        self.hue = hue

    def read(self):
        import numpy as np

        now = time.time()
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[:, :, 0] = int(96 + 96 * math.sin(now + self.hue))
        image[:, :, 1] = int(96 + 96 * math.sin(now * 0.7 + self.hue))
        sweep = int(now * 180 % self.width)
        image[:, max(0, sweep - 20):sweep + 20] = 255
        cv2.putText(image, self.label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6)
        cv2.putText(image, self.label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
        cv2.putText(image, f"{now:.2f}", (20, self.height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        time.sleep(1 / 30)
        return image

    def close(self) -> None:
        pass


class CameraStream:
    """Pulls frames off a camera in the background so the UI never blocks."""

    def __init__(self, source, clock=time.monotonic):
        self.source = source
        self.label = source.label
        self._clock = clock
        self._lock = threading.Lock()
        self._frame = None
        self._fps = 0.0
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> "CameraStream":
        self._thread = threading.Thread(target=self._loop, name=self.label, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            image = self.source.read()
            if image is None:
                self._stop.wait(0.5)  # let a missing camera come back on its own
                continue
            now = self._clock()
            with self._lock:
                if self._frame is not None:
                    gap = now - self._frame.at
                    if gap > 0:  # exponential average, so the readout is steady
                        self._fps = 0.9 * self._fps + 0.1 / gap if self._fps else 1 / gap
                self._frame = Frame(image, now)

    def latest(self):
        with self._lock:
            return self._frame

    def fps(self) -> float:
        with self._lock:
            return self._fps

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.source.close()


def collect(streams, clock=time.monotonic, stale_after: float = STALE_AFTER):
    """The newest frame from every camera, or a complaint naming the bad one."""
    now = clock()
    frames = []
    for stream in streams:
        frame = stream.latest()
        if frame is None:
            raise CaptureError(f"{stream.label} has not delivered a frame yet")
        if now - frame.at > stale_after:
            raise CaptureError(f"{stream.label} stopped delivering frames")
        frames.append(frame)
    return frames


def skew_ms(frames) -> float:
    """How far apart the two frames were read, in milliseconds."""
    stamps = [frame.at for frame in frames]
    return (max(stamps) - min(stamps)) * 1000


def save(frames, paths, quality: int = 95) -> None:
    for frame, path in zip(frames, paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        params = []
        if path.suffix.lower() in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        if not cv2.imwrite(str(path), frame.image, params):
            raise CaptureError(f"could not write {path}")


# --- UI --------------------------------------------------------------------


def to_photo(image, width: int = PREVIEW_WIDTH):
    """A Tk image of *image*, scaled to *width* -- no Pillow needed."""
    height, source_width = image.shape[:2]
    if source_width > width:
        scale = width / source_width
        image = cv2.resize(image, (width, max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".ppm", image)  # OpenCV writes PPM in RGB order
    if not ok:
        raise CaptureError("could not encode the preview")
    return tk.PhotoImage(data=buffer.tobytes())  # Tk reads raw PPM; Pillow stays optional


class CaptureApp:
    """Two live views, a shutter on the spacebar, and an adjustable timer."""

    def __init__(self, root, streams, args):
        self.root = root
        self.streams = streams
        self.args = args
        self.outdir = Path(args.outdir)
        self.suffix = ".png" if args.format == "png" else ".jpg"
        self.countdown = None
        self.shots = 0
        self._photos = [None] * len(streams)  # Tk drops an image it holds no reference to
        self._last_beep = None

        self.delay_var = tk.StringVar(value=f"{args.delay:g}")
        self.beep_var = tk.BooleanVar(value=not args.no_beep)
        self.banner_var = tk.StringVar(value="Ready")
        self.status_var = tk.StringVar(value=f"Shots go to {self.outdir.resolve()}")
        self.caption_vars = [tk.StringVar(value=f"{s.label} — waiting…") for s in streams]

        self._build()
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Escape>", self._on_escape)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self._tick()

    def _build(self) -> None:
        self.root.title("Dual USB Capture")
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        views = ttk.Frame(frame)
        views.pack(fill="both", expand=True)
        self.view_labels = []
        for column, caption in enumerate(self.caption_vars):
            views.columnconfigure(column, weight=1)
            box = ttk.LabelFrame(views, labelwidget=ttk.Label(views, textvariable=caption))
            box.grid(row=0, column=column, padx=5, pady=5, sticky="nsew")
            view = ttk.Label(box, text="no signal", anchor="center",
                             width=PREVIEW_WIDTH // 8, padding=20)
            view.pack(fill="both", expand=True)
            self.view_labels.append(view)

        ttk.Label(frame, textvariable=self.banner_var, anchor="center",
                  font=("TkDefaultFont", 40)).pack(fill="x", pady=(6, 2))

        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=6)
        self.shutter = ttk.Button(controls, text="Capture  (Space)", takefocus=False,
                                  command=self.toggle)
        self.shutter.pack(side="left", ipady=8, padx=(0, 16))
        ttk.Label(controls, text="Timer").pack(side="left")
        ttk.Spinbox(controls, from_=0, to=MAX_DELAY, increment=0.5, width=6,
                    textvariable=self.delay_var).pack(side="left", padx=4)
        ttk.Label(controls, text="seconds").pack(side="left", padx=(0, 16))
        ttk.Checkbutton(controls, text="Beep", variable=self.beep_var,
                        takefocus=False).pack(side="left")

        ttk.Label(frame, textvariable=self.status_var, anchor="w").pack(fill="x")

    # -- input

    def _on_space(self, event):
        if isinstance(event.widget, (ttk.Entry, ttk.Spinbox, tk.Entry, tk.Spinbox)):
            return None  # someone is typing in the timer box
        self.toggle()
        return "break"

    def _on_escape(self, _event=None):
        if self.countdown is not None:
            self.cancel()
        return "break"

    def toggle(self) -> None:
        if self.countdown is not None:
            self.cancel()
        else:
            self.arm()

    def arm(self) -> None:
        delay = clamp_delay(self.delay_var.get())
        self.delay_var.set(f"{delay:g}")
        if delay <= 0:
            self.shoot()
            return
        self.countdown = Countdown.start(delay, time.monotonic())
        self._last_beep = None
        self.shutter.configure(text="Cancel  (Space)")
        self.status_var.set(f"Capturing in {delay:g}s — space or Esc cancels")

    def cancel(self) -> None:
        self.countdown = None
        self.shutter.configure(text="Capture  (Space)")
        self.banner_var.set("Ready")
        self.status_var.set("Cancelled")

    # -- the shot

    def shoot(self) -> None:
        self.countdown = None
        self.shutter.configure(text="Capture  (Space)")
        try:
            frames = collect(self.streams)
            paths = shot_paths(self.outdir, [s.label for s in self.streams],
                               datetime.now(), self.suffix)
            save(frames, paths, self.args.quality)
        except CaptureError as exc:
            self.banner_var.set("Ready")
            self.status_var.set(f"Nothing captured — {exc}")
            log.warning("capture failed: %s", exc)
            return
        self.shots += 1
        names = " + ".join(path.name for path in paths)
        self.status_var.set(
            f"Saved {names} — {skew_ms(frames):.0f} ms apart — "
            f"{self.shots} shot(s) in {self.outdir.resolve()}"
        )
        log.info("saved %s", names)
        self.banner_var.set("Saved")
        self.root.after(FLASH_MS, self._clear_banner)

    def _clear_banner(self) -> None:
        if self.countdown is None and self.banner_var.get() == "Saved":
            self.banner_var.set("Ready")

    # -- the loop

    def _tick(self) -> None:
        self._refresh_views()
        if self.countdown is not None:
            now = time.monotonic()
            if self.countdown.expired(now):
                self.shoot()
            else:
                left = self.countdown.remaining(now)
                self.banner_var.set(countdown_text(left))
                whole = math.ceil(left)
                if whole != self._last_beep:
                    self._last_beep = whole
                    if self.beep_var.get():
                        self.root.bell()
        self.root.after(POLL_MS, self._tick)

    def _refresh_views(self) -> None:
        now = time.monotonic()
        for index, stream in enumerate(self.streams):
            frame = stream.latest()
            if frame is None or now - frame.at > STALE_AFTER:
                self.caption_vars[index].set(f"{stream.label} — no signal")
                continue
            height, width = frame.image.shape[:2]
            self.caption_vars[index].set(
                f"{stream.label} — {width}x{height} — {stream.fps():.0f} fps"
            )
            photo = to_photo(frame.image, self.args.preview_width)
            self._photos[index] = photo
            self.view_labels[index].configure(image=photo, text="")

    def quit(self) -> None:
        self.root.destroy()


# --- wiring ----------------------------------------------------------------


def list_cameras(limit: int = 10) -> int:
    """Print the device indices that open, so --cameras can be filled in."""
    try:  # probing empty indices is noisy on Linux, and the noise is expected
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:  # pragma: no cover - older OpenCV builds
        pass
    found = 0
    for index in range(limit):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, image = cap.read()
            size = f"{image.shape[1]}x{image.shape[0]}" if ok else "opened, no frame"
            print(f"  --cameras {index}   ({size})")
            found += 1
        cap.release()
    if not found:
        print("no cameras found — check the USB connections, or try --demo")
    return 0 if found else 1


def build_streams(args):
    if args.demo:
        sources = [SyntheticCamera(label, hue=hue) for hue, label in enumerate(args.labels)]
    else:
        sources = [UsbCamera(index, label, args.width, args.height)
                   for index, label in zip(args.cameras, args.labels)]
    return [CameraStream(source).start() for source in sources]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--cameras", nargs=2, type=int, metavar=("A", "B"),
                        default=[0, 1], help="the two USB device indices (default: 0 1)")
    parser.add_argument("-l", "--labels", nargs=2, metavar=("A", "B"),
                        help="names for the two cameras (default: cam<index>)")
    parser.add_argument("-o", "--outdir", default="captures",
                        help="where the shots are written (default: captures)")
    parser.add_argument("-d", "--delay", type=float, default=DEFAULT_DELAY,
                        help=f"timer in seconds (default: {DEFAULT_DELAY:g})")
    parser.add_argument("--width", type=int, default=0, help="requested capture width")
    parser.add_argument("--height", type=int, default=0, help="requested capture height")
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg",
                        help="image format (default: jpg)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (default: 95)")
    parser.add_argument("--preview-width", type=int, default=PREVIEW_WIDTH,
                        help=f"width of each live view (default: {PREVIEW_WIDTH})")
    parser.add_argument("--no-beep", action="store_true", help="silence the countdown beeps")
    parser.add_argument("--demo", action="store_true",
                        help="run against synthetic cameras, with no hardware")
    parser.add_argument("--list", action="store_true",
                        help="list the camera indices that open, then exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)
    args.delay = clamp_delay(args.delay)
    if not args.labels:
        args.labels = ["demoA", "demoB"] if args.demo else [f"cam{i}" for i in args.cameras]
    if args.labels[0] == args.labels[1]:
        parser.error("the two cameras need different labels")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if cv2 is None:
        print("OpenCV is missing — pip install -r requirements.txt", file=sys.stderr)
        return 1
    if args.list:
        return list_cameras()
    if tk is None:
        print("Tkinter is missing — install python3-tk (Debian/Ubuntu) or use a "
              "Python built with Tk", file=sys.stderr)
        return 1

    streams = build_streams(args)
    root = tk.Tk()
    app = CaptureApp(root, streams, args)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.quit()
    finally:
        for stream in streams:
            stream.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
