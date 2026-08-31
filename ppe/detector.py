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
        self._settle_batching()
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

    def _settle_batching(self) -> None:
        """Make ``self.batches`` match what this model will actually accept.

        An export is compiled for exactly one frame count and refuses every
        other; only a .pt takes any. So ``batch`` is a request the model can
        decline in either direction — a batch-1 export asked for two raises,
        and a batch-2 export asked for one raises just as hard. Both surface
        inside the runtime rather than at load: during warmup if warmup is on,
        which reads as "the model failed to load", and otherwise mid-shift on
        the first cycle. Neither is a place to find out, and a station running
        at the wrong batch size is strictly better than one that will not
        start, so ask the model here and take its answer.

        Only a refusal that the *other* count survives is a batch mismatch. If
        both fail the model is broken, and that error is the real one — it goes
        through untouched rather than being reported as a batching problem.
        """
        blank = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        wanted = 2 if self.batches else 1
        refusal = self._refuses(blank, wanted)
        if refusal is not None:
            if self._refuses(blank, 1 if self.batches else 2) is not None:
                raise refusal  # not about the count — the model itself is broken
            self.batches = not self.batches
            log.warning(
                "%s takes %d frame(s) per call and refuses %d — %s. An export only "
                "ever accepts the batch it was exported with; set model.batch: %s to "
                "say so, or re-export with `python -m ppe.export --batch %d`.",
                self.cfg.weights, 2 if self.batches else 1, wanted,
                "serving both cameras in one call" if self.batches
                else "serving one camera per cycle",
                "true" if self.batches else "auto",
                2 if self.batches else 1,
            )

        self.batch_size = 2 if self.batches else 1
        # Two cameras never deliver in lockstep, so a cycle with one fresh frame
        # is routine, not an error — and a fixed-batch export refuses a short
        # call as hard as an oversized one. Find out now whether short calls are
        # allowed; detect() pads them if they are not.
        self.exact = self.batch_size > 1 and self._refuses(blank, 1) is not None
        if self.exact:
            log.info(
                "%s takes exactly %d frames per call, so cycles with fewer fresh "
                "frames are padded to that and the padding discarded",
                self.cfg.weights, self.batch_size,
            )

    def _refuses(self, blank: np.ndarray, count: int) -> Exception | None:
        """The error from asking for ``count`` frames, or None if it ran."""
        try:
            self._predict([blank] * count)
        except Exception as exc:  # noqa: BLE001 — any refusal means the same
            return exc
        return None

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

    def _predict_all(self, canvases: list[np.ndarray]):
        """Run every canvas, in calls this model will actually accept.

        A fixed-batch export takes exactly ``batch_size`` frames — no more and
        no fewer — while the number of cameras with a fresh frame this cycle is
        whatever the cameras happened to deliver. Dropping the odd ones out
        would mean a station that stops judging because one camera is a frame
        behind, so short calls are padded to the compiled size and the padding
        thrown away. A model that takes any count skips all of this.
        """
        if not self.exact:
            return self._predict(canvases)
        out = []
        for start in range(0, len(canvases), self.batch_size):
            chunk = canvases[start : start + self.batch_size]
            padded = chunk + [chunk[-1]] * (self.batch_size - len(chunk))
            out.extend(self._predict(padded)[: len(chunk)])
        return out

    def detect(
        self, images: list[np.ndarray]
    ) -> tuple[list[list[Detection]], dict[str, float]]:
        """Detect on every image; returns per-image detections and stage timings."""
        t0 = now()
        pairs = [letterbox(img, self.cfg.imgsz) for img in images]
        canvases = [c for c, _ in pairs]
        t1 = now()

        results = self._predict_all(canvases)
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
