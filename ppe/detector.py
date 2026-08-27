"""YOLOv11 inference, tuned for whichever device it lands on."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import ModelCfg, PPECfg
from .latency import now
from .letterbox import Letterbox, letterbox
from .runtime import apply_torch, configure

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Detection:
    name: str
    conf: float
    xyxy: tuple[float, float, float, float]  # source-frame pixels


def _precision_kwargs(half: bool) -> dict[str, object]:
    """Ask for fp16 in whichever dialect this Ultralytics speaks.

    8.4 replaced ``half=True`` with ``quantize="fp16"``; passing the old name
    still works but logs a deprecation warning on every call.
    """
    if not half:
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT

        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": "fp16"}
    except ImportError:
        pass
    return {"half": True}


def resolve_device(want: str) -> str:
    if want != "auto":
        return want
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    """Wraps a YOLOv11 model and speaks only in source-frame coordinates.

    Both camera frames go through one ``predict`` call: the GPU is launched
    once per cycle instead of once per camera, which roughly halves the
    per-frame overhead of a two-camera station.
    """

    def __init__(self, cfg: ModelCfg, ppe: PPECfg) -> None:
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.half = cfg.half and self.device.startswith("cuda")
        self.threads = configure(cfg.threads)
        if not self.device.startswith("cuda"):
            apply_torch(self.threads)
        # ONNX and OpenVINO exports are fixed at batch 1, and on CPU a batch of
        # two costs about twice a batch of one anyway.
        self.batches = cfg.batches(self.device)

        self.model = YOLO(cfg.weights)
        self.names: dict[int, str] = dict(self.model.names)
        self._precision = _precision_kwargs(self.half)

        self.set_classes(ppe)
        log.info(
            "model %s on %s (fp16=%s, threads=%d, batched=%s)",
            cfg.weights, self.device, self.half, self.threads, self.batches,
        )
        if cfg.warmup:
            self.warmup()

    def set_classes(self, ppe: PPECfg) -> list[str]:
        """Point the detector at a new class list; returns the unknown names."""
        by_name = {v: k for k, v in self.names.items()}
        self.missing = [c.name for c in ppe.classes if c.name not in by_name]
        if self.missing:
            log.warning(
                "not in %s, will never be detected: %s",
                self.cfg.weights,
                ", ".join(self.missing),
            )
        # Filtering inside NMS is cheaper than throwing boxes away afterwards.
        self.class_ids = sorted(by_name[c.name] for c in ppe.classes if c.name in by_name)
        # Run NMS at the lowest threshold any class asks for, then apply the
        # per-class floors on the way out.
        self._floors = {
            c.name: (c.conf if c.conf is not None else self.cfg.conf) for c in ppe.classes
        }
        self._conf = min([self.cfg.conf, *self._floors.values()]) if self._floors else self.cfg.conf
        return self.missing

    def warmup(self, batch: int = 0) -> None:
        """Pay the first-call cost (kernel autotune, cuDNN plans) up front."""
        blank = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        t = now()
        self._predict([blank] * (batch or (2 if self.batches else 1)))
        log.info("warmup %.0f ms", (now() - t) * 1000)

    def _predict(self, canvases: list[np.ndarray]):
        return self.model.predict(
            canvases,
            imgsz=self.cfg.imgsz,
            conf=self._conf,
            iou=self.cfg.iou,
            max_det=self.cfg.max_det,
            classes=self.class_ids or None,
            device=self.device,
            verbose=False,
            **self._precision,
        )

    def detect(
        self, images: list[np.ndarray]
    ) -> tuple[list[list[Detection]], dict[str, float]]:
        """Detect on every image; returns per-image detections and stage timings."""
        t0 = now()
        pairs = [letterbox(img, self.cfg.imgsz) for img in images]
        canvases = [c for c, _ in pairs]
        t1 = now()

        results = self._predict(canvases)
        t2 = now()

        out = [self._decode(r, meta) for r, (_, meta) in zip(results, pairs, strict=True)]
        t3 = now()

        return out, {
            "preprocess": (t1 - t0) * 1000,
            "inference": (t2 - t1) * 1000,
            "postprocess": (t3 - t2) * 1000,
        }

    def _decode(self, result, meta: Letterbox) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        # One host transfer for the whole result, not one per box.
        xyxy = meta.to_source(boxes.xyxy.cpu().numpy())
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        dets = []
        for box, conf, cid in zip(xyxy, confs, clss, strict=True):
            name = self.names.get(int(cid), str(cid))
            if conf < self._floors.get(name, self.cfg.conf):
                continue
            dets.append(Detection(name, float(conf), (float(box[0]), float(box[1]),
                                                      float(box[2]), float(box[3]))))
        return dets
