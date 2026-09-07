"""RF-DETR inference — an alternative backend to Detector's ultralytics YOLO,
selected by ``model.arch: rfdetr``.

This wraps RF-DETR's own Python API directly (its own ``predict()``), the
same way ``Detector`` wraps ultralytics' ``YOLO(...).predict()`` — not a
hand-rolled ONNX Runtime decode. RF-DETR's export() targets onnx/tflite/
tensorrt/executorch/coreml, not native OpenVINO IR the way ultralytics'
export does; OpenVINO there is an ONNX Runtime execution provider, a
separate conversion step this file does not attempt. Until that exists,
this runs on plain PyTorch — correct, but without the ~5x CPU speedup the
YOLO path gets from OpenVINO. That parity is a follow-up, not a blocker:
this project's own YOLO path started on a bare .pt too (see the README's
"Running on a CPU" section) and got its OpenVINO export later, once the
model itself was proven.

Two differences from the YOLO path are load-bearing, not incidental:

  * RF-DETR expects RGB. Every frame in this pipeline arrives as BGR
    straight from OpenCV, same as the YOLO path — converted once per frame
    here, since nothing upstream of this file knows the difference exists.
  * predict() resizes internally and returns boxes in the ORIGINAL image's
    coordinates. There is deliberately no letterbox/coordinate-remap step
    here the way there is for YOLO (ppe/letterbox.py) — passing the raw
    frame straight through is correct for this backend, not a shortcut.

Verified against the rf-detr repository's documentation and source
(github.com/roboflow/rf-detr, develop branch, checked 2026-09) rather than
against trained weights — none exist yet for this station's classes. The
two points above, and the class-name lookup in ``_read_class_names``, are
what to confirm first once real weights exist; everything else mirrors
Detector's contract exactly, so nothing downstream (Pipeline, bench.py,
the UI) needs to know or care which backend it was handed.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from .config import ModelCfg, PPECfg
from .detector import Detection
from .latency import now

log = logging.getLogger(__name__)


class RFDetrDetector:
    """Same public contract as Detector: .missing, .names, .batches,
    .detect(), .set_classes(), constructed the same way — ``make_detector()``
    is the only place that needs to know this class exists.
    """

    def __init__(self, cfg: ModelCfg, ppe: PPECfg) -> None:
        try:
            from rfdetr import RFDETRBase
        except ImportError as exc:
            raise RuntimeError(
                "model.arch is 'rfdetr' but the rfdetr package is not installed — "
                "pip install rfdetr"
            ) from exc

        self.cfg = cfg
        # predict() takes exactly one image per call — there is no batch API
        # to opt into, unlike the YOLO path where batching is a real (if
        # usually CPU-inadvisable) choice. Pipeline._take() already serves
        # one camera per cycle whenever self.batches is False, so this just
        # always takes that branch; nothing else needs to change for it.
        self.batches = False
        self.batch_size = 1
        self.exact = False

        # A checkpoint is a single .pth file, not a directory — unlike the
        # OpenVINO IR weights the YOLO path normally points at, which *are*
        # directories and, by this config's own convention, written with a
        # trailing slash (see config.yaml). Carrying that habit over to an
        # rfdetr checkpoint tells the OS this path names a directory, and
        # opening a file that way fails with a bare, confusing OSError
        # (errno 22) from deep inside torch.load — normalize it away rather
        # than let that surface.
        weights = os.path.normpath(cfg.weights)
        log.info("loading RF-DETR checkpoint %s", weights)
        self.model = RFDETRBase.from_checkpoint(weights)
        self.names = self._read_class_names()
        self._floors: dict[str, float] = {}
        self._conf = cfg.conf
        self.missing: list[str] = []
        self.set_classes(ppe)

        if cfg.warmup:
            self.warmup()

    def _read_class_names(self) -> dict[int, str]:
        """The model's full class vocabulary, however this build exposes it.

        Tried in order: a class_names attribute on the model wrapper itself,
        then on its inner .model — both documented, at different points in
        rf-detr's own source, as where this lives, and which one it actually
        is was not resolvable without a real checkpoint loaded. If neither
        exists, this raises rather than silently starting with an empty
        vocabulary: every configured class would read as unavailable and
        force the station to DEGRADED, which is a worse failure than
        stopping at load time and saying exactly what went looking for.
        """
        names = getattr(self.model, "class_names", None)
        if names is None:
            names = getattr(getattr(self.model, "model", None), "class_names", None)
        if names is None:
            raise RuntimeError(
                "could not find class_names on the loaded RF-DETR model "
                "(tried model.class_names and model.model.class_names) — "
                "the rfdetr package's API may not match what this integration "
                "was written against; check RFDETRBase.from_checkpoint()'s "
                "return value directly"
            )
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        return dict(enumerate(names))  # a plain list; index is the class id

    def set_classes(self, ppe: PPECfg) -> list[str]:
        """Point the detector at a new class list; returns the unknown names."""
        by_name = {v: k for k, v in self.names.items()}
        self.missing = [c.name for c in ppe.classes if c.name not in by_name]
        if self.missing:
            log.warning(
                "not in %s, will never be detected: %s",
                self.cfg.weights, ", ".join(self.missing),
            )
        self._floors = {
            c.name: (c.conf if c.conf is not None else self.cfg.conf) for c in ppe.classes
        }
        # Ask the model for the lowest floor any configured class needs, then
        # apply the per-class floors on the way out — same reasoning as
        # Detector.set_classes: asking for less than that up front would mean
        # never getting some configured classes' detections back at all.
        self._conf = min([self.cfg.conf, *self._floors.values()]) if self._floors else self.cfg.conf
        return self.missing

    def warmup(self) -> None:
        """Pay the first-call cost up front, same reason as Detector's."""
        blank = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        t = now()
        self._predict_one(blank)
        log.info("warmup %.0f ms", (now() - t) * 1000)

    def _predict_one(self, image_bgr: np.ndarray):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.model.predict(rgb, threshold=self._conf)

    def detect(
        self, images: list[np.ndarray]
    ) -> tuple[list[list[Detection]], dict[str, float]]:
        """Detect on every image; returns per-image detections and stage timings."""
        t0 = now()
        results = [self._predict_one(img) for img in images]
        t1 = now()
        out = [self._decode(r) for r in results]
        t2 = now()
        # predict() does its own resize as part of the same call inference
        # is timed over, so there is no separate preprocess stage the way
        # letterbox() gives the YOLO path one.
        return out, {
            "preprocess": 0.0,
            "inference": (t1 - t0) * 1000,
            "postprocess": (t2 - t1) * 1000,
        }

    def _decode(self, result) -> list[Detection]:
        class_names = result.data.get("class_name") if hasattr(result, "data") else None
        dets = []
        for i in range(len(result.xyxy)):
            name = (
                str(class_names[i]) if class_names is not None
                else self.names.get(int(result.class_id[i]), str(int(result.class_id[i])))
            )
            if name not in self._floors:
                continue  # not a class this station checks for
            conf = float(result.confidence[i])
            if conf < self._floors[name]:
                continue
            x1, y1, x2, y2 = (float(v) for v in result.xyxy[i])
            dets.append(Detection(name, conf, (x1, y1, x2, y2)))
        return dets
