#!/usr/bin/env python3
"""PPE detection station — entry point.

    python main.py                     # UI on the cameras in config.yaml
    python main.py --headless          # no UI; prints the latency breakdown
    python main.py --profile           # add a cProfile hotspot report on exit
    python main.py --capture           # also keep frames for training
    python main.py --selftest          # load the model, run it once, exit 0/1
    python main.py --supervise         # restart the station if it crashes
    python main.py camcheck --scan     # the camera diagnostic (python -m ppe.camcheck)

An installed build (PPEStation.exe) takes the same flags. It reads its
config from %ProgramData%\\PPE Station\\config.yaml and logs to the logs
folder beside it — see ppe/paths.py and docs/DEPLOYMENT.md.
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import logging.handlers
import os
import shutil
import signal
import sys
import time
from pathlib import Path

from ppe import __version__
from ppe.capture import CameraSet
from ppe.config import Config
from ppe.latency import Profiler
from ppe.paths import InstanceLock, app_dir, data_dir, frozen
from ppe.runtime import configure

log = logging.getLogger("ppe")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-c", "--config", default=None,
        help="path to the YAML config (default: config.yaml here, or the installed "
             "machine's %%ProgramData%%\\PPE Station\\config.yaml)",
    )
    p.add_argument("--headless", action="store_true", help="run without the UI")
    p.add_argument("--profile", action="store_true", help="cProfile the run and report hotspots")
    p.add_argument("--profile-out", default="logs/profile.prof", help="where to write the profile")
    p.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0 = forever)")
    p.add_argument(
        "--debug", action="store_true",
        help="debug UI: decision panel, latency HUD, editable checklist, muted audio",
    )
    p.add_argument(
        "--operator", action="store_true", help="force the plain operator UI",
    )
    p.add_argument(
        "--capture", action="store_true",
        help="capture mode: keep frames for training (see the dataset section of the config)",
    )
    p.add_argument(
        "--no-capture", action="store_true", help="force capture mode off for this run",
    )
    p.add_argument(
        "--selftest", action="store_true",
        help="load the model and run it once on a blank frame, then exit 0 (ok) or 1",
    )
    p.add_argument(
        "--supervise", action="store_true",
        help="run the station as a child process and restart it after a crash",
    )
    p.add_argument(
        "--set-relay-ip", metavar="IP",
        help="write this machine's relay module address into the config, then exit "
             "(the installer runs this once per machine)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--version", action="version", version=f"PPE Station {__version__}")
    return p.parse_args(argv)


def prepare_installed() -> None:
    """An installed build runs from its data folder, logs to a file, stays offline.

    Relative paths in config.yaml (logs/, datasets/) then land in
    %ProgramData%\\PPE Station instead of wherever the shortcut happened to
    start; the model and assets are found in the program folder by
    ppe.paths.resolve(). Ultralytics is told never to reach the internet or
    pip-install anything: a station PC has neither, and must not try.
    """
    home = data_dir()
    (home / "logs").mkdir(parents=True, exist_ok=True)
    os.chdir(home)
    # Ultralytics adds its own "Ultralytics" folder and only creates it when
    # the parent already exists, so point it at the data folder itself.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(home))
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def default_config() -> Path:
    """config.yaml in the working directory, or the installed machine's own copy.

    A fresh install that somehow lacks one gets the shipped default rather
    than refusing to start; the installer normally writes it, with this
    machine's relay address filled in.
    """
    if not frozen():
        return Path("config.yaml")
    path = data_dir() / "config.yaml"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(app_dir() / "config.yaml", path)
    return path


def setup_logging(verbose: bool, name: str = "station") -> None:
    handlers: list[logging.Handler] = []
    if sys.stderr is not None:   # a windowed .exe has no console at all
        handlers.append(logging.StreamHandler())
    if frozen():
        logs = data_dir() / "logs"
        handlers.append(logging.handlers.RotatingFileHandler(
            logs / f"{name}.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"))
        # A native crash (a driver, a DLL) kills the process before Python can
        # log anything; faulthandler still gets a traceback onto disk.
        faulthandler.enable(open(logs / f"{name}-crash.log", "a"))  # noqa: SIM115 — process lifetime
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S" if frozen() else "%H:%M:%S",
        handlers=handlers,
    )


def selftest(cfg) -> int:
    """Load the model and run it once — no cameras, no window, no relay board.

    The check that an installed build is complete: a missing DLL, model
    file or hidden import fails here, at install time, instead of on the
    shop floor. Exit code 0 means the station can detect.
    """
    import numpy as np

    from ppe.detector import Detector

    t0 = time.monotonic()
    try:
        detector = Detector(cfg.model, cfg.ppe, base=cfg.base_dir)
        loaded = time.monotonic() - t0
        cam = cfg.cameras[0]
        frame = np.zeros((cam.height or 720, cam.width or 1280, 3), np.uint8)
        detections, timings = detector.detect([frame])
        # The window and the relay link are part of the build too.
        import pymodbus  # noqa: F401
        import PySide6.QtWidgets  # noqa: F401
    except Exception:
        log.exception("selftest FAILED")
        print("selftest FAILED - see the log above", file=sys.stderr if sys.stderr else None)
        return 1
    missing = getattr(detector, "missing", [])
    msg = (f"selftest OK - PPE Station {__version__}, model {cfg.model.weights}, "
           f"{len(detector.names)} classes, loaded in {loaded:.1f} s, "
           f"inference {timings['inference']:.0f} ms, "
           f"{len(detections[0])} detections on a blank frame")
    if missing:
        msg += f"; WARNING: model lacks configured classes {missing}"
    log.info(msg)
    print(msg)
    return 0


def report(text: str) -> None:
    """End-of-run output: the console when there is one, the log when installed."""
    print(text)
    if frozen():
        log.info("\n%s", text)


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
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "camcheck":
        # `PPEStation-console.exe camcheck --scan`: the "no video signal"
        # diagnostic (ppe/camcheck.py), shipped inside the installed program.
        if frozen():
            prepare_installed()
        from ppe.camcheck import main as camcheck

        rest = argv[1:]
        if not {"-c", "--config"} & set(rest):
            rest = ["-c", str(default_config()), *rest]
        return camcheck(rest)

    args = parse_args(argv)
    if frozen():
        prepare_installed()
    # The supervisor and the station are separate processes; on Windows they
    # cannot share one rotating log file.
    setup_logging(args.verbose, "supervisor" if args.supervise else "station")
    if args.supervise:
        from ppe.supervise import supervise

        return supervise(argv)
    if args.verbose:
        # cv2's own VIDEOIO errors ("device busy", "no such device", a
        # permission denial) go to its native logger, not Python's — this is
        # usually the actual answer when a camera silently won't open.
        import cv2

        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_DEBUG)

    config_path = args.config or default_config()
    if args.set_relay_ip:
        from ppe.provision import set_relay_host

        try:
            old = set_relay_host(config_path, args.set_relay_ip)
        except ValueError as exc:
            log.error("cannot set the relay address: %s", exc)
            return 2
        log.info("%s: relay module address %s -> %s", config_path, old, args.set_relay_ip)
        return 0

    cfg = Config.load(config_path)
    if args.debug:
        cfg.ui.mode = "debug"
    if args.operator:
        cfg.ui.mode = "operator"
    if args.capture:
        cfg.dataset.enabled = True
    if args.no_capture:
        cfg.dataset.enabled = False
    cfg.validate()
    if args.selftest:
        configure(cfg.model.threads)
        return selftest(cfg)

    lock = InstanceLock()
    if not lock.acquire():
        # Exit 0, not an error: the copy already running is the station, and
        # a supervisor that started this one should stop, not retry.
        log.error("PPE Station is already running on this PC; not starting a second copy")
        return 0
    log.info("PPE Station %s starting, config %s", __version__, cfg.path)

    if cfg.dataset.enabled:
        # Loud on purpose. This writes photographs of identifiable people to
        # local disk, and the one failure mode that matters is nobody
        # realising it is on.
        log.warning("CAPTURE MODE ON — writing frames to %s (%s)",
                    cfg.dataset.dir, ", ".join(cfg.dataset.triggers))

    # Before torch is imported: the maths libraries read their thread counts
    # from the environment as they initialise.
    configure(cfg.model.threads)

    profiler = Profiler(args.profile_out) if args.profile else None
    if profiler and not args.headless:
        # The pipeline thread profiles itself either way; the main thread is
        # only worth sampling when it is drawing the UI, not sleeping.
        profiler.start()

    cameras = CameraSet(cfg.cameras).start()

    # Imported here so --help and config errors never wait on torch.
    from ppe.pipeline import Pipeline

    # The model loads on the pipeline thread, in the background — building
    # a Pipeline here is cheap, and pipeline.start() returns immediately.
    # Nothing in the UI path waits on it: the window shows straight away,
    # video panes paint as soon as the cameras deliver a frame, and the
    # checklist shows a "loading model" banner in the meantime rather than
    # the whole app appearing to hang until a several-second load finishes.
    pipeline = Pipeline(cfg, cameras, profiler=profiler)
    pipeline.start()

    try:
        if args.headless:
            # No window to show early here, so waiting up front is only
            # ever useful, never something to skip.
            if not cameras.wait_ready(timeout=10.0):
                log.warning("not every camera delivered a frame; continuing anyway")
            run_headless(args.seconds)
            code = 0
        else:
            code = run_ui(cfg, cameras, pipeline)
    finally:
        pipeline.stop()
        cameras.stop()
        report("\n=== latency breakdown (ms) ===\n" + pipeline.metrics.report())
        if profiler:
            report(profiler.stop())
        lock.release()

    return code


if __name__ == "__main__":
    raise SystemExit(main())
