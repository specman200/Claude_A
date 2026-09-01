"""Diagnose "no video signal" without touching the rest of the station.

    python -m ppe.camcheck                # probe every source in config.yaml
    python -m ppe.camcheck -c other.yaml
    python -m ppe.camcheck --scan          # also enumerate cameras this machine sees
    python -m ppe.camcheck --modes         # what each camera will ACTUALLY deliver
    python -m ppe.camcheck --save frame.jpg

Works the same on Windows and Linux; --scan and the printed causes adapt to
whichever this is running on.

Run this FIRST when a camera won't come up. It opens each configured source
directly — no threads, no detector, no UI — and reports exactly where it
failed: cannot open, opens but never delivers a frame, or works. Add -v to
also print OpenCV's own VIDEOIO error, which is usually the real answer
("device busy", "no such device", a permission denial) and is otherwise
invisible.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from .capture import _APIS
from .config import CameraCfg, Config

ON_WINDOWS = sys.platform.startswith("win")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument(
        "--scan", action="store_true",
        help="also enumerate cameras (/dev/video* on Linux, index probing on Windows)",
    )
    p.add_argument("--save", metavar="FILE", help="save the first working frame of each camera")
    p.add_argument(
        "-v", "--verbose", action="store_true", help="print OpenCV's own VIDEOIO diagnostics"
    )
    p.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for a frame")
    p.add_argument(
        "--modes", action="store_true",
        help="measure what each camera really delivers at each size and format "
             "(opens every combination in turn; takes ~20s per camera)",
    )
    return p.parse_args(argv)


def scan_dev_video() -> list[str]:
    """Every /dev/video* node this machine currently exposes. Linux only."""
    import glob

    return sorted(glob.glob("/dev/video*"))


def scan_windows_indices(max_index: int = 6) -> list[tuple[int, str]]:
    """Which camera indices respond, and to which backend, on Windows.

    OpenCV has no "list camera names" API on Windows, so this is the practical
    equivalent of --scan there: try opening each index against both backends
    and report what answers. Slow (each probe can take ~1s), so kept small.
    """
    found = []
    for i in range(max_index):
        for name, flag in (("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF)):
            cap = cv2.VideoCapture(i, flag)
            ok = cap.isOpened()
            cap.release()
            if ok:
                found.append((i, name))
    return found


# Sizes worth asking a UVC webcam for, smallest first so a bandwidth-starved
# camera shows its working modes before its stalling ones. Both formats are
# tried at every size: which one the camera actually honours is the question.
PROBE_SIZES = ((640, 480), (800, 600), (1280, 720), (1920, 1080))
PROBE_FORMATS = ("MJPG", "YUY2")


def measure_mode(src, api: int, fourcc: str, width: int, height: int, fps: int,
                 warmup: int = 3, timed: int = 10) -> dict | None:
    """Open at one mode and measure what actually comes back, or None.

    Every number here is measured rather than asked for. A driver will report
    a mode it is not delivering — that is the whole reason this exists — so
    the frame size comes from a decoded frame's own shape and the rate from
    timing real reads. CAP_PROP_FOURCC is recorded too, but only as a claim:
    under DirectShow it is frequently wrong, and the measured rate is what
    actually tells you whether compression is in play.
    """
    cap = cv2.VideoCapture(src, api)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        # FOURCC before size, same order the app uses: on DirectShow the pixel
        # format decides which sizes are offered at all.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)

        for _ in range(warmup):  # the first frames after a mode switch settle
            if not cap.read()[0]:
                return None
        start = time.monotonic()
        shape = None
        for _ in range(timed):
            ok, frame = cap.read()
            if not ok:
                return None
            shape = frame.shape
        elapsed = time.monotonic() - start

        tag = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little", signed=False)
        return {
            "width": shape[1],
            "height": shape[0],
            "fps": timed / elapsed if elapsed > 0 else 0.0,
            "claims": tag.decode("ascii", "replace").strip() or "?",
        }
    except Exception:  # noqa: BLE001 — a mode that misbehaves is a result, not a crash
        return None
    finally:
        cap.release()


def probe_modes(cfg: CameraCfg) -> None:
    """Print what this camera really delivers at each size and format.

    OpenCV can say what you got but never what was on offer, which leaves the
    one question that matters unanswerable: is this camera slow because it has
    no compressed mode, or because something else is throttling it? Asking for
    each mode and timing the result answers it directly.
    """
    src = cfg.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    if not isinstance(src, int):
        # A file or stream has no mode table to negotiate: it ignores every
        # size and format request and decodes as fast as the CPU allows, which
        # would print a confident table of numbers that mean nothing.
        print(f"\n{cfg.name}  (source={src!r})")
        print("  not a camera — a file or stream has no modes to probe, skipping")
        return
    api = _APIS.get(cfg.api, cv2.CAP_ANY)
    total = len(PROBE_SIZES) * len(PROBE_FORMATS)
    print(f"\n{cfg.name}  (source={src!r}, api={cfg.api})")
    print(f"  measuring {total} modes — each is opened, set and timed for real\n")
    print("    asked            delivered      claims   measured")

    best: dict[str, tuple[float, int, int]] = {}
    for fourcc in PROBE_FORMATS:
        for width, height in PROBE_SIZES:
            got = measure_mode(src, api, fourcc, width, height, cfg.fps)
            if got is None:
                print(f"    {fourcc} {width:>5}x{height:<5}   —  no frames")
                continue
            print(f"    {fourcc} {width:>5}x{height:<5}   "
                  f"{got['width']:>5}x{got['height']:<5}  {got['claims']:<7}"
                  f"{got['fps']:6.1f} fps")
            rate = got["fps"]
            if fourcc not in best or rate * got["width"] > best[fourcc][0] * best[fourcc][1]:
                best[fourcc] = (rate, got["width"], got["height"])

    if not best:
        print("\n  nothing delivered a frame at any mode — run without --modes first")
        return
    # Whether compression is really in play is answered by the rate, not by the
    # FOURCC the driver reports back: at the same size, an honoured MJPG
    # request beats a raw one by more than measurement noise.
    mjpg, yuy2 = best.get("MJPG"), best.get("YUY2")
    if mjpg and yuy2 and mjpg[0] > yuy2[0] * 1.3:
        print(f"\n  MJPG is honoured — {mjpg[0]:.0f} fps at {mjpg[1]}x{mjpg[2]} against "
              f"{yuy2[0]:.0f} fps raw. Keep fourcc: MJPG.")
    elif mjpg and yuy2:
        print("\n  MJPG buys nothing here — the rates match, so the camera is almost "
              "certainly uncompressed whatever it reports.\n  Resolution is the only "
              "lever on bandwidth; keep fourcc: MJPG anyway, an ignored request is free.")
    fastest = max(best.values(), key=lambda b: b[0])
    print(f"  fastest measured: {fastest[1]}x{fastest[2]} at {fastest[0]:.0f} fps")
    print(f"  currently configured: {cfg.width}x{cfg.height} @ {cfg.fps} fps, "
          f"fourcc {cfg.fourcc or 'driver default'}")


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

    # Same order the app uses: FOURCC before resolution, since on
    # DirectShow/MSMF that decides which resolutions are even offered.
    if cfg.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
    if cfg.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    if cfg.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    if cfg.fps:
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)

    got_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    got_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    got_fps = cap.get(cv2.CAP_PROP_FPS)
    got_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    got_tag = got_fourcc.to_bytes(4, "little").decode("ascii", "replace").strip() or "?"
    print(f"  reports {got_w:.0f}x{got_h:.0f} @ {got_fps:.0f} fps, format {got_tag}"
         + ("" if not cfg.width else
            "  <- driver ignored the requested size" if got_w != cfg.width else ""))
    if cfg.fourcc and got_tag != cfg.fourcc:
        raw_bw = got_w * got_h * 2 * got_fps / 1e6
        print(f"  WARNING: asked for {cfg.fourcc}, driver gave {got_tag} instead — "
              f"if that is a raw format this is ~{raw_bw:.0f} MB/s over USB, "
              "likely why two cameras will not both come up")

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


CAUSES_WINDOWS = """
If a camera FAILED to open, in likely order on a Windows PC:

  1. Camera privacy setting blocks it silently. Settings > Privacy & security
     > Camera > "Let desktop apps access your camera" — if this is OFF, a
     Python/OpenCV process is blocked even though Store apps like the
     Windows Camera app still work fine. This is the single most common
     cause on a locked-down industrial/kiosk image, and it typically fails
     with no error at all rather than a clear one.
  2. Another process already has it open. UVC devices usually allow only one
     reader. Close Teams/Zoom/the Windows Camera app, and check for a
     previous crashed run of this program still holding the device (Task
     Manager > Details, end any stray python.exe).
  3. Wrong backend. `api: any` lets OpenCV choose, which is normally MSMF
     (Media Foundation) on a modern build — but some UVC cameras, especially
     older or industrial ones with a vendor driver, work with one backend
     and not the other. Try `api: dshow` and `api: msmf` explicitly in
     config.yaml; --scan (below) shows which backend answers for which
     index.
  4. Driver problem. Device Manager > Cameras (or "Imaging devices" /
     "Sound, video and game controllers" on older Windows) — a yellow
     warning icon or "Unknown device" means the driver did not install.
  5. USB power management. Device Manager > the camera's USB Root Hub >
     Power Management tab > uncheck "Allow the computer to turn off this
     device to save power". Common on industrial PCs with aggressive power
     profiles, and looks identical to a camera that "randomly" stops.
  6. USB bandwidth. Two cameras at high resolution can exceed what one
     shared hub/controller carries — the second fails to open while the
     first works alone. Move them to separate USB controllers (not just
     separate ports on the same hub), or lower `width`/`height`.
  7. Wrong index. Windows can renumber cameras after a reboot or a replug.
     --scan opens indices 0-5 against both backends and reports what
     answers, so you are not guessing.

If a camera OPENED but no frame arrived, or every frame is one flat
colour: check the lens cap and the cable, and re-check #1 above — a
privacy block sometimes lets the device open but delivers nothing.

Rerun with -v for OpenCV's own diagnostic on the failing camera.
"""

CAUSES_LINUX = """
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

def causes() -> str:
    """The platform-appropriate troubleshooting text, read at call time —
    not baked in at import — so tests (and a frozen ON_WINDOWS override) see
    the branch they actually asked for."""
    return CAUSES_WINDOWS if ON_WINDOWS else CAUSES_LINUX


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_DEBUG)

    if args.scan:
        if ON_WINDOWS:
            print("probing indices 0-5 against dshow and msmf (a few seconds)...")
            hits = scan_windows_indices()
            if not hits:
                print("  nothing answered — see the causes below")
            for index, backend in hits:
                print(f"  index {index}: opens via {backend}")
        else:
            nodes = scan_dev_video()
            print("/dev/video* nodes:" if nodes else "no /dev/video* nodes found (not Linux, "
                  "or no camera driver has bound yet)")
            for n in nodes:
                print(f"  {n}")

    cfg = Config.load(args.config)
    if not cfg.cameras:
        print(f"\nno cameras configured in {args.config}")
        return 1

    if args.modes:
        for cam in cfg.cameras:
            probe_modes(cam)
        return 0

    results = [
        probe(cam, args.timeout, args.save, f"cam{i}")
        for i, cam in enumerate(cfg.cameras)
    ]

    if not all(results):
        print(causes())
    print(f"\n{sum(results)}/{len(results)} camera(s) delivering real frames")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
