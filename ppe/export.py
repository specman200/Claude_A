"""Export the model to a faster CPU runtime.

    python -m ppe.export                      # OpenVINO IR, the CPU default
    python -m ppe.export -w runs/best.pt      # a checkpoint you trained yourself
    python -m ppe.export --format onnx
    python -m ppe.export --int8 --data d.yaml # quantised, needs calibration images

OpenVINO is usually several times faster than PyTorch on an Intel CPU for the
same weights; run ``python -m ppe.bench`` afterwards to see what you actually
got on your own machine.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger("ppe.export")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-c", "--config", default="config.yaml", help="read model.weights from here")
    p.add_argument("-w", "--weights", help="override the .pt to export")
    p.add_argument(
        "-f", "--format", default="openvino", choices=("openvino", "onnx"),
        help="target runtime (default: openvino)",
    )
    p.add_argument("--imgsz", type=int, help="inference size; defaults to the config's")
    p.add_argument(
        "--int8", action="store_true",
        help="quantise to int8 — roughly 2x again, at some accuracy cost",
    )
    p.add_argument(
        "--data", help="dataset yaml used to calibrate --int8; use your own images",
    )
    return p.parse_args(argv)


# What ultralytics names its exports: `<stem>.pt` becomes `<stem>_openvino_model/`,
# `<stem>.onnx`, and so on.
EXPORT_DIRS = ("_openvino_model", "_ncnn_model", "_saved_model", "_paddle_model")


def source_weights(weights: str) -> Path | None:
    """The .pt behind ``weights``, or None if nothing exportable is there.

    Only a PyTorch checkpoint can be exported; ultralytics rejects every other
    format with a TypeError that names the format rather than the fix. But
    ``model.weights`` normally points at the *export*, because that is what the
    station runs — so "regenerate the export" has to mean "go back to the .pt it
    came from". Resolving that here keeps the documented `python -m ppe.export`
    working against a config that points at an export, generated or not.
    """
    path = Path(weights)
    if path.suffix.lower() == ".pt":
        return path
    for suffix in EXPORT_DIRS:
        if path.name.endswith(suffix):
            beside = path.with_name(path.name[: -len(suffix)] + ".pt")
            return beside if beside.is_file() else None
    beside = path.with_suffix(".pt")
    return beside if beside.is_file() else None

def describe(path: Path) -> str:
    """The resolved path and size — which file was read is half of any answer."""
    try:
        size = f"{path.stat().st_size / 1e6:.1f} MB"
    except OSError as exc:
        size = f"unreadable ({exc.strerror})"
    return f"{path.resolve()}  ({size})"


def versions() -> str:
    """The other half: an export that fails on one machine and not another is
    almost always a version difference, so never make anyone go and look."""
    from importlib.metadata import PackageNotFoundError, version

    out = []
    for name in ("ultralytics", "torch", "openvino", "onnx"):
        try:
            out.append(f"{name}={version(name)}")
        except PackageNotFoundError:
            out.append(f"{name}=not installed")
    return "  ".join(out)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from ultralytics import YOLO

    from .config import Config

    cfg = Config.load(args.config)
    weights = args.weights or cfg.model.weights
    imgsz = args.imgsz or cfg.model.imgsz

    source = source_weights(weights)
    if source is None:
        log.error(
            "%s is not a PyTorch model and no .pt sits beside it — only a .pt can "
            "be exported. Point -w at the checkpoint you trained:\n"
            "    python -m ppe.export -w path/to/best.pt",
            weights,
        )
        return 1
    if not source.is_file():
        log.error("no such weights: %s", source)
        return 1
    if source != Path(weights):
        # Say which file is actually being read: exporting the wrong weights is
        # a mistake that stays quiet until the station misbehaves on the floor.
        log.info("%s is an export, not a checkpoint — reading %s instead", weights, source)
    weights = str(source)
    if args.int8 and not args.data:
        # Calibrating on someone else's images is how int8 quietly loses recall.
        log.error(
            "--int8 needs --data pointing at a dataset yaml, so the quantiser "
            "calibrates on images that look like yours"
        )
        return 1

    log.info("exporting %s -> %s at imgsz=%d%s", weights, args.format, imgsz,
             " (int8)" if args.int8 else "")
    # Batch 1: the station serves one camera per cycle on CPU, which is both
    # faster than batching and what these exports support.
    try:
        out = YOLO(weights).export(
            format=args.format, imgsz=imgsz, batch=1, int8=args.int8,
            **({"data": args.data} if args.data else {}),
            **({"simplify": True, "dynamic": False} if args.format == "onnx" else {}),
        )
    except TypeError as exc:
        # Several unrelated conditions all surface as TypeError here, and each
        # message describes the file rather than the file it describes: a
        # TorchScript archive, a bare state_dict, a YOLOv5-era checkpoint, a
        # truncated download. Which file was read, and on which versions, is
        # what actually separates them — so say that, not just the exception.
        log.error("ultralytics refused this file:\n  %s", describe(Path(weights)))
        log.error("  %s", versions())
        log.error("\n%s", exc)
        return 1
    log.info("\nwrote %s", out)
    log.info("point model.weights at it in %s, then: python -m ppe.bench", args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
