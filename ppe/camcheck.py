"""Diagnose "no video signal" without touching the rest of the station.

    python -m ppe.camcheck                # probe every source in config.yaml
    python -m ppe.camcheck -c other.yaml
    python -m ppe.camcheck --scan          # also list every /dev/video* node
    python -m ppe.camcheck --save frame.jpg

Run this FIRST when a camera won't come up. It opens each configured source
directly — no threads, no detector, no UI — and reports exactly where it
failed: cannot open, opens but never delivers a frame, or works. Add -v to
also print OpenCV's own VIDEOIO error, which is usually the real answer
("device busy", "no such device", a permission denial) and is otherwise
invisible.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .capture import _APIS
from .config import CameraCfg, Config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("--scan", action="store_true", help="also list /dev/video* nodes (Linux)")
    p.add_argument("--save", metavar="FILE", help="save the first working frame of each camera")
    p.add_argument(
        "-v", "--verbose", action="store_true", help="print OpenCV's own VIDEOIO diagnostics"
    )
    p.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for a frame")
    return p.parse_args(argv)


def scan_dev_video() -> list[str]:
    """Every /dev/video* node this machine currently exposes."""
    import glob

    return sorted(glob.glob("/dev/video*"))


def probe(cfg: CameraCfg, timeout: float, save: str | None, tag: str) -> bool:
    src = cfg.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    print(f"\n{cfg.name}  (source={cfg.source!r}, api={cfg.api})")

    api = _APIS.get(cfg.api, cv2.CAP_ANY)
    cap = cv2.VideoCapture(src, api)
    if not cap.isOpened():
        print("  FAILED to open — see the causes below")
        cap.release()
        return False
    print(f"  opened via {cap.getBackendName()}")

    if cfg.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    if cfg.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    if cfg.fps:
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)

    got_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    got_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    got_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  reports {got_w:.0f}x{got_h:.0f} @ {got_fps:.0f} fps"
         + ("" if not cfg.width else
            "  <- driver ignored the requested size" if got_w != cfg.width else ""))

    deadline = time.monotonic() + timeout
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        ok, frame = cap.read()
        if ok and frame is not None:
            blank = frame.max() == frame.min()
            print(f"  frame {attempts}: {frame.shape[1]}x{frame.shape[0]}"
                 + (" — ALL ONE COLOUR (lens cap on? no signal reaching the sensor?)"
                    if blank else " — looks real"))
            if save:
                out = Path(save).with_stem(f"{Path(save).stem}_{tag}")
                cv2.imwrite(str(out), frame)
                print(f"  saved -> {out}")
            cap.release()
            return not blank
        time.sleep(0.1)

    print(f"  opened, but no frame arrived in {timeout:.0f}s ({attempts} attempts)")
    cap.release()
    return False


CAUSES = """
If a camera FAILED to open, in likely order for two USB webcams on an
industrial PC:

  1. Wrong index. Many UVC webcams expose TWO /dev/video nodes (one real,
     one metadata-only) — so index 1 can be the wrong node of camera 0,
     not camera 1 at all. Run --scan and try each node's number directly
     as `source:` in config.yaml, not just 0 and 1.
  2. Permissions. `groups` must include `video`, or the process needs
     root. Fix: `sudo usermod -aG video $USER`, then log out and back in.
  3. USB bandwidth. Two MJPG/YUYV streams at high resolution can exceed
     what a shared USB 2.0 hub/controller can carry — the second camera
     fails to open while the first works. Try separate hubs/controllers,
     or lower `width`/`height` in config.yaml.
  4. Device busy. Another process (or a previous crashed run) still holds
     it. `sudo fuser -v /dev/video0`, or reboot.
  5. Backend mismatch. `api: any` picks whatever OpenCV finds first; set
     `api: v4l2` explicitly on Linux if it is guessing wrong.
  6. Container/VM. If this runs inside Docker, /dev/video0 and /dev/video1
     must be passed through explicitly: `--device=/dev/video0
     --device=/dev/video1` (podman/docker) — they are not visible by
     default even to a privileged-looking container.

If a camera OPENED but no frame arrived, or every frame is one flat
colour: check the lens cap, the cable, and that nothing else (a second
instance of this app, a video call, `cheese`/`guvcview`) is already
streaming from it — most UVC drivers only allow one reader.

Rerun with -v for OpenCV's own VIDEOIO error on the failing camera —
that message ("no such device", "device or resource busy", a permission
denial) usually names the exact cause directly.
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_DEBUG)

    if args.scan:
        nodes = scan_dev_video()
        print("/dev/video* nodes:" if nodes else "no /dev/video* nodes found (not Linux, "
              "or no camera driver has bound yet)")
        for n in nodes:
            print(f"  {n}")

    cfg = Config.load(args.config)
    if not cfg.cameras:
        print(f"\nno cameras configured in {args.config}")
        return 1

    results = [
        probe(cam, args.timeout, args.save, f"cam{i}")
        for i, cam in enumerate(cfg.cameras)
    ]

    if not all(results):
        print(CAUSES)
    print(f"\n{sum(results)}/{len(results)} camera(s) delivering real frames")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
