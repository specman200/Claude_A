"""Export the model to a faster CPU runtime.

    python -m ppe.export                      # OpenVINO IR, the CPU default
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from ultralytics import YOLO

    from .config import Config

    cfg = Config.load(args.config)
    weights = args.weights or cfg.model.weights
    imgsz = args.imgsz or cfg.model.imgsz

    if not Path(weights).exists():
        log.error("no such weights: %s", weights)
        return 1
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
    out = YOLO(weights).export(
        format=args.format, imgsz=imgsz, batch=1, int8=args.int8,
        **({"data": args.data} if args.data else {}),
        **({"simplify": True, "dynamic": False} if args.format == "onnx" else {}),
    )
    log.info("\nwrote %s", out)
    log.info("point model.weights at it in %s, then: python -m ppe.bench", args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
