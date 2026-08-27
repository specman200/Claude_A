"""Measure inference on this machine, so backend choice is a measurement.

    python -m ppe.bench                        # every model in models/
    python -m ppe.bench -w a.pt b_openvino_model/
    python -m ppe.bench --imgsz 640 512 448    # sweep inference sizes

Reports median and p95 latency per call, and the detections found, so a
backend that is fast because it stopped detecting things is obvious.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from .config import Config, ModelCfg
from .latency import now
from .runtime import configure

log = logging.getLogger("ppe.bench")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("-w", "--weights", nargs="+", help="models to compare (default: models/*)")
    p.add_argument("--imgsz", type=int, nargs="+", help="sizes to sweep")
    p.add_argument("-n", "--runs", type=int, default=15, help="timed calls per model")
    p.add_argument("--threads", type=int, default=0, help="inference threads; 0 = every core")
    p.add_argument("--image", help="image to run on; default is a mid-grey frame")
    return p.parse_args(argv)


def discover(root: Path = Path("models")) -> list[str]:
    """Every model in models/ — .pt, .onnx and OpenVINO IR directories."""
    if not root.is_dir():
        return []
    found = [str(p) for p in sorted(root.glob("*.pt"))]
    found += [str(p) for p in sorted(root.glob("*.onnx"))]
    found += [f"{p}/" for p in sorted(root.glob("*_openvino_model")) if p.is_dir()]
    return found


def time_model(weights: str, cfg: Config, imgsz: int, image, runs: int) -> dict:
    from .detector import Detector

    model = ModelCfg(
        weights=weights, imgsz=imgsz, device="cpu", half=False,
        conf=cfg.model.conf, iou=cfg.model.iou, threads=cfg.model.threads, warmup=False,
    )
    detector = Detector(model, cfg.ppe)
    for _ in range(3):  # let the runtime settle before timing
        detector.detect([image])

    samples, found = [], 0
    for _ in range(runs):
        t = now()
        dets, _ = detector.detect([image])
        samples.append((now() - t) * 1000)
        found = len(dets[0])
    samples.sort()
    return {
        "p50": samples[len(samples) // 2],
        "p95": samples[min(len(samples) - 1, int(len(samples) * 0.95))],
        "min": samples[0],
        "dets": found,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    cfg = Config.load(args.config)
    if args.threads:
        cfg.model.threads = args.threads
    threads = configure(cfg.model.threads)

    weights = args.weights or discover()
    if not weights:
        print("no models found — pass -w, or export one with python -m ppe.export")
        return 1
    sizes = args.imgsz or [cfg.model.imgsz]

    if args.image:
        import cv2

        image = cv2.imread(args.image)
        if image is None:
            print(f"cannot read {args.image}")
            return 1
    else:
        image = np.full((cfg.model.imgsz, cfg.model.imgsz, 3), 114, np.uint8)

    print(f"\n{threads} threads · {args.runs} runs · "
          f"{'grey frame' if not args.image else Path(args.image).name}\n")
    print(f"{'model':<44}{'imgsz':>7}{'p50 ms':>10}{'p95 ms':>10}{'fps':>8}{'dets':>7}")
    print("-" * 86)

    rows = []
    for size in sizes:
        for path in weights:
            try:
                r = time_model(path, cfg, size, image, args.runs)
            except Exception as exc:  # noqa: BLE001 — report and keep going
                print(f"{Path(path).name:<44}{size:>7}   failed: {str(exc)[:40]}")
                continue
            rows.append((path, size, r))
            name = path if len(path) <= 43 else "…" + path[-42:]
            print(f"{name:<44}{size:>7}{r['p50']:>10.1f}{r['p95']:>10.1f}"
                  f"{1000 / r['p50']:>8.1f}{r['dets']:>7}")

    if len(rows) > 1:
        best = min(rows, key=lambda x: x[2]["p50"])
        worst = max(rows, key=lambda x: x[2]["p50"])
        print(f"\nfastest: {best[0]} @ {best[1]} — {best[2]['p50']:.1f} ms, "
              f"{worst[2]['p50'] / best[2]['p50']:.1f}x the slowest here")
        counts = {r[2]["dets"] for r in rows if r[1] == rows[0][1]}
        if len(counts) > 1:
            print(f"NOTE: detection counts differ at the same imgsz ({sorted(counts)}) — "
                  "a backend that finds less is not faster, it is worse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
