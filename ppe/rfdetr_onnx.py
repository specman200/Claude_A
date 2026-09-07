"""RF-DETR inference from an exported graph — ONNX or OpenVINO IR.

The third detector backend, and the only one that needs no deep-learning
framework at runtime: it runs the graph ``rfdetr``'s own ``export()``
produces, through OpenVINO (or ONNX Runtime), with the pre- and
post-processing done here in numpy. Selected by pointing ``model.weights``
at a ``.onnx`` or ``.xml`` while ``model.arch`` is ``rfdetr``.

Why it exists: ``RFDetrDetector`` goes through ``RFDETRBase.from_checkpoint``,
which needs torch, the rfdetr package, and a ``.pth`` — a heavy install for
a station that only ever infers, and with no route to the ~5x CPU speedup
the YOLO path gets from OpenVINO IR. This closes that gap. The trade is
that ``predict()``'s conveniences have to be reimplemented, and there are
exactly two of them:

  * **Preprocess** — resize to the graph's own input size (a plain resize,
    not a letterbox, matching what ``predict()`` does), RGB, scale to
    0..1, ImageNet mean/std, NCHW.
  * **Postprocess** — the graph returns *raw* tensors: boxes as normalized
    ``cxcywh`` and logits that have not been through a sigmoid, with no
    NMS anywhere (RF-DETR is a set predictor — there is nothing to
    suppress). So: sigmoid, threshold, ``cxcywh`` -> ``xyxy``, scale to
    source pixels.

Because the boxes come back normalized, scaling them to the source frame
is a plain multiply and the aspect distortion of the resize cancels out —
which is why there is no letterbox metadata to carry around the way
``Detector`` needs for YOLO.

Two things this backend cannot learn from the graph and must be told:

  * **Class names.** An rfdetr export carries no name metadata (unlike an
    ultralytics one, which is why pointing ``arch: yolo`` at an RF-DETR
    ONNX yields ``class0..class998`` and a DEGRADED station). Set
    ``model.class_names`` in the config, in class-id order. The length is
    checked against the logits width at load, so a stale or short list
    fails at startup rather than silently mislabelling every detection.
  * **Which output is which.** Decided from the shapes of the real
    outputs on the first inference — the one whose last axis is 4 is the
    boxes — rather than from output names, which differ between exporter
    versions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from .config import ModelCfg, PPECfg
from .detector import Detection
from .latency import now

log = logging.getLogger(__name__)

# rf-detr's own preprocessing constants: ImageNet statistics over RGB 0..1.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# What OpenVINO will accept as a device. A station configured for cuda:0 or
# mps is asking for something OpenVINO has no plugin for, so it runs on CPU
# rather than failing — the graph is already exported and portable.
OV_DEVICES = frozenset({"CPU", "GPU", "NPU", "AUTO"})

class RFDetrOnnx:
    """Same public contract as Detector and RFDetrDetector: .missing,
    .names, .batches, .detect(), .set_classes(), built the same way.
    """

    def __init__(self, cfg: ModelCfg, ppe: PPECfg) -> None:
        self.cfg = cfg
        # One image per call. The exports are compiled at batch 1 and this
        # backend does not pad short cycles, so Pipeline._take() serves one
        # camera per cycle — the same branch RFDetrDetector takes.
        self.batches = False
        self.batch_size = 1
        self.exact = False

        weights = os.path.normpath(cfg.weights)
        self.runtime = self._open(weights, cfg.device)
        self.height, self.width = self._input_hw()
        log.info(
            "loaded RF-DETR graph %s via %s at %dx%d",
            weights, self.runtime, self.width, self.height,
        )

        self._floors: dict[str, float] = {}
        self._conf = cfg.conf
        self.missing: list[str] = []
        self._boxes_first: bool | None = None  # settled on the first real output
        self.names: dict[int, str] = dict(enumerate(cfg.class_names))
        # Always one inference at load, warmup configured or not: it is what
        # settles which output is which and checks the configured class list
        # against the graph's real logits width. Both are far better found
        # here than mid-shift on the first frame of a real cycle.
        t = now()
        self._probe()
        log.info("warmup %.0f ms", (now() - t) * 1000)
        self.set_classes(ppe)

    # -- loading -----------------------------------------------------------
    def _open(self, weights: str, want_device: str) -> str:
        """Compile the graph, preferring OpenVINO. Returns the runtime name.

        OpenVINO first because it is what makes this backend worth having on
        a CPU, and because it reads plain .onnx as happily as its own IR. It
        can still refuse a graph whose ops it does not implement, and a
        DETR carries a few unusual ones (deformable attention lowers to
        GridSample) — so a .onnx that will not compile falls back to ONNX
        Runtime with a warning rather than leaving the station dead. An .xml
        has no fallback: it is OpenVINO's own format.
        """
        device = self._ov_device(want_device)
        try:
            import openvino as ov

            compiled = ov.Core().compile_model(weights, device)
        except Exception as exc:  # noqa: BLE001 — the fallback is the point
            if Path(weights).suffix.lower() == ".xml":
                raise RuntimeError(
                    f"OpenVINO could not compile {weights!r}, and an .xml is "
                    "its own format so there is nothing else to try. Export "
                    "again, or point model.weights at the .onnx instead."
                ) from exc
            log.warning(
                "OpenVINO could not compile %s (%s) — falling back to ONNX "
                "Runtime, which is slower on CPU but supports more ops. To "
                "get the OpenVINO speedup, re-export at a lower opset or "
                "convert with `ovc` and check what it reports.",
                weights, exc,
            )
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                weights, providers=["CPUExecutionProvider"]
            )
            self._in_name = self._session.get_inputs()[0].name
            return "onnxruntime"
        self._compiled = compiled
        return f"openvino:{device}"

    @staticmethod
    def _ov_device(want: str) -> str:
        """model.device, translated into something OpenVINO recognises."""
        if want.upper() in OV_DEVICES:
            return want.upper()
        if want != "auto":
            log.warning(
                "model.device is %r, which OpenVINO has no plugin for — "
                "running this graph on CPU. Use CPU, GPU, NPU or AUTO to "
                "pick an OpenVINO device explicitly.",
                want,
            )
        return "CPU"

    def _input_hw(self) -> tuple[int, int]:
        """The size the graph was frozen at, or model.imgsz if it is dynamic."""
        shape: list[int | None]
        if self.runtime == "onnxruntime":
            raw = self._session.get_inputs()[0].shape
            shape = [d if isinstance(d, int) else None for d in raw]
        else:
            ps = self._compiled.input(0).partial_shape
            shape = [
                ps[i].get_length() if ps[i].is_static else None
                for i in range(ps.rank.get_length())
            ]
        if len(shape) == 4 and shape[2] and shape[3]:
            return int(shape[2]), int(shape[3])
        log.info(
            "the graph's input size is dynamic — using model.imgsz (%d). It "
            "must be divisible by the variant's patch_size * num_windows or "
            "the graph will reject it.",
            self.cfg.imgsz,
        )
        return self.cfg.imgsz, self.cfg.imgsz

    def _read_class_names(self, num_classes: int) -> dict[int, str]:
        """Names for the ids the graph emits, and a check that they line up."""
        if not self.names:
            raise RuntimeError(
                f"{self.cfg.weights} is an exported graph, which carries no "
                f"class names — set model.class_names in the config to the "
                f"{num_classes} names in class-id order. The training "
                f"checkpoint knows them:\n"
                f"    from rfdetr import RFDETRBase\n"
                f"    m = RFDETRBase.from_checkpoint('best.pth', device='cpu')\n"
                f"    print(m.class_names)"
            )
        if len(self.names) != num_classes:
            raise RuntimeError(
                f"model.class_names has {len(self.names)} entries but "
                f"{self.cfg.weights} predicts {num_classes} classes — every "
                f"detection would be mislabelled. Fix the list rather than "
                f"padding it; the order is the class-id order the model was "
                f"trained with."
            )
        return self.names

    # -- inference ---------------------------------------------------------
    def _run(self, tensor: np.ndarray) -> list[np.ndarray]:
        if self.runtime == "onnxruntime":
            return list(self._session.run(None, {self._in_name: tensor}))
        return [np.asarray(v) for v in self._compiled(tensor).values()]

    def _probe(self) -> None:
        """One inference on a blank frame, to settle the output layout."""
        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        _boxes, logits = self._split(self._run(self._preprocess(blank)))
        self.names = self._read_class_names(int(logits.shape[-1]))

    def warmup(self) -> None:
        """Interface parity — __init__ already paid the first-call cost."""

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
        # Threshold once at the lowest floor any class asks for, then apply
        # the per-class floors on the way out — the same shape as the other
        # two backends, reached without an NMS to push it down into.
        self._conf = min([self.cfg.conf, *self._floors.values()]) if self._floors else self.cfg.conf
        return self.missing

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (self.height, self.width):
            rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def _split(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """(boxes, logits), told apart by shape rather than by name.

        Output names differ between rfdetr versions ("dets"/"labels" in the
        docs, "pred_boxes"/"pred_logits" in the model), but the boxes are
        the only tensor whose last axis is 4, and a station with exactly 4
        classes would otherwise be a coin flip on names alone.
        """
        if len(outputs) != 2:
            raise RuntimeError(
                f"expected 2 outputs from {self.cfg.weights} (boxes and "
                f"logits), got {len(outputs)} with shapes "
                f"{[o.shape for o in outputs]} — this does not look like an "
                f"RF-DETR export."
            )
        if self._boxes_first is None:
            four = [o.shape[-1] == 4 for o in outputs]
            if four.count(True) != 1:
                raise RuntimeError(
                    f"cannot tell the boxes from the logits in "
                    f"{self.cfg.weights}: shapes {[o.shape for o in outputs]}, "
                    f"and exactly one output should end in 4. A 4-class model "
                    f"is the awkward case — re-export, or run the .pth "
                    f"backend instead."
                )
            self._boxes_first = four[0]
        return (outputs[0], outputs[1]) if self._boxes_first else (outputs[1], outputs[0])

    def detect(
        self, images: list[np.ndarray]
    ) -> tuple[list[list[Detection]], dict[str, float]]:
        """Detect on every image; returns per-image detections and stage timings."""
        t0 = now()
        tensors = [self._preprocess(img) for img in images]
        t1 = now()
        raw = [self._run(t) for t in tensors]
        t2 = now()
        out = [
            self._decode(r, img.shape[1], img.shape[0])
            for r, img in zip(raw, images, strict=True)
        ]
        t3 = now()
        return out, {
            "preprocess": (t1 - t0) * 1000,
            "inference": (t2 - t1) * 1000,
            "postprocess": (t3 - t2) * 1000,
        }

    def _decode(
        self, outputs: list[np.ndarray], src_w: int, src_h: int
    ) -> list[Detection]:
        boxes, logits = self._split(outputs)
        scores = 1.0 / (1.0 + np.exp(-logits[0].astype(np.float32)))  # sigmoid
        boxes = boxes[0].astype(np.float32)

        dets = []
        # One threshold pass in numpy: a set predictor emits a fixed number of
        # queries (300) every frame regardless of the scene, so filtering
        # first and looping over what survives keeps this off the hot path.
        for query, cid in np.argwhere(scores >= self._conf):
            name = self.names.get(int(cid))
            if name is None or name not in self._floors:
                continue  # not a class this station checks for
            conf = float(scores[query, cid])
            if conf < self._floors[name]:
                continue
            cx, cy, w, h = boxes[query]
            dets.append(Detection(
                name, conf,
                ((cx - w / 2) * src_w, (cy - h / 2) * src_h,
                 (cx + w / 2) * src_w, (cy + h / 2) * src_h),
            ))
        return dets
