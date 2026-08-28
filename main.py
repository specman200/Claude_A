#!/usr/bin/env python3
"""PPE detection station — entry point.

    python main.py                     # UI on the cameras in config.yaml
    python main.py --headless          # no UI; prints the latency breakdown
    python main.py --profile           # add a cProfile hotspot report on exit
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from ppe.capture import CameraSet
from ppe.config import Config
from ppe.latency import Profiler
from ppe.runtime import configure

log = logging.getLogger("ppe")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-c", "--config", default="config.yaml", help="path to the YAML config")
    p.add_argument("--headless", action="store_true", help="run without the UI")
    p.add_argument("--profile", action="store_true", help="cProfile the run and report hotspots")
    p.add_argument("--profile-out", default="logs/profile.prof", help="where to write the profile")
    p.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0 = forever)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def run_headless(seconds: float) -> None:
    """Loop until Ctrl+C, then print where the time went."""
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    deadline = time.monotonic() + seconds if seconds else float("inf")
    print("running headless — Ctrl+C to stop")
    while not stop["now"] and time.monotonic() < deadline:
        time.sleep(0.2)
    print("\nstopping…")


def run_ui(cfg, cameras, pipeline) -> int:
    from PySide6.QtWidgets import QApplication

    from ppe.ui import MainWindow

    app = QApplication(sys.argv[:1])
    window = MainWindow(cfg, cameras, pipeline)
    window.show()
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.verbose:
        # cv2's own VIDEOIO errors ("device busy", "no such device", a
        # permission denial) go to its native logger, not Python's — this is
        # usually the actual answer when a camera silently won't open.
        import cv2

        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_DEBUG)

    cfg = Config.load(args.config)
    cfg.validate()

    # Before torch is imported: the maths libraries read their thread counts
    # from the environment as they initialise.
    configure(cfg.model.threads)

    profiler = Profiler(args.profile_out) if args.profile else None
    if profiler and not args.headless:
        # The pipeline thread profiles itself either way; the main thread is
        # only worth sampling when it is drawing the UI, not sleeping.
        profiler.start()

    cameras = CameraSet(cfg.cameras).start()
    if not cameras.wait_ready(timeout=10.0):
        log.warning("not every camera delivered a frame; continuing anyway")

    # Imported here so --help and config errors never wait on torch.
    from ppe.pipeline import Pipeline

    pipeline = Pipeline(cfg, cameras, profiler=profiler)
    pipeline.start()

    try:
        if args.headless:
            run_headless(args.seconds)
            code = 0
        else:
            code = run_ui(cfg, cameras, pipeline)
    finally:
        pipeline.stop()
        cameras.stop()
        print("\n=== latency breakdown (ms) ===")
        print(pipeline.metrics.report())
        if profiler:
            print(profiler.stop())

    return code


if __name__ == "__main__":
    raise SystemExit(main())
