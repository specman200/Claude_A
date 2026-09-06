"""RFDetrDetector against a stubbed rfdetr package — no real weights exist
yet for this station's classes, so what these pin is the contract:
BGR->RGB conversion, class-name resolution, per-class confidence floors,
and the same .missing/.batches/.detect() surface Detector has, so Pipeline,
bench.py and the UI need no changes to run either backend.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from ppe.config import ClassCfg, ModelCfg, PPECfg


class FakeResult:
    """Stands in for the supervision.Detections-shaped object predict() hands
    back: .xyxy, .confidence, .class_id arrays plus .data["class_name"]."""

    def __init__(self, boxes=(), confidences=(), class_ids=(), class_names=()):
        self.xyxy = np.array(boxes, dtype=float).reshape(-1, 4)
        self.confidence = np.array(confidences, dtype=float)
        self.class_id = np.array(class_ids, dtype=int)
        self.data = {"class_name": list(class_names)}


class FakeModel:
    """Stands in for RFDETRBase.from_checkpoint(...)'s return value."""

    def __init__(self, class_names, result=None):
        self.class_names = class_names
        self.result = result if result is not None else FakeResult()
        self.calls: list[np.ndarray] = []
        self.last_threshold: float | None = None
        self.loaded_from: str | None = None

    def predict(self, image, threshold=0.5):
        self.calls.append(image.copy())
        self.last_threshold = threshold
        return self.result


def install_fake_rfdetr(monkeypatch, model: FakeModel) -> None:
    """A stand-in `rfdetr` module whose RFDETRBase.from_checkpoint(...)
    returns ``model`` regardless of the path asked for — no real rfdetr
    install or weights needed."""
    module = types.ModuleType("rfdetr")

    class RFDETRBase:
        @staticmethod
        def from_checkpoint(path):
            model.loaded_from = path
            return model

    module.RFDETRBase = RFDETRBase
    monkeypatch.setitem(sys.modules, "rfdetr", module)


def ppe(*names, conf=None):
    return PPECfg(classes=[ClassCfg(n, conf=conf) for n in names])


def cfg(**kw):
    kw.setdefault("weights", "checkpoint.pth")
    kw.setdefault("arch", "rfdetr")
    kw.setdefault("warmup", False)
    return ModelCfg(**kw)


# -- the one path this sandbox can test for real, not through a stub -------


def test_rfdetr_not_installed_raises_a_clear_error():
    """rfdetr genuinely is not installed here — this is real behaviour, not
    a stubbed one: the error must name the fix, not surface an ImportError
    with no context."""
    from ppe.rfdetr_detector import RFDetrDetector

    with pytest.raises(RuntimeError, match="pip install rfdetr"):
        RFDetrDetector(cfg(), ppe("glove"))


# -- everything else, against the stub --------------------------------------


def test_class_names_from_a_dict_checkpoint(monkeypatch):
    from ppe.rfdetr_detector import RFDetrDetector

    install_fake_rfdetr(monkeypatch, FakeModel({0: "glove", 1: "person"}))
    det = RFDetrDetector(cfg(), ppe("glove"))
    assert det.names == {0: "glove", 1: "person"}


def test_class_names_from_a_list_checkpoint(monkeypatch):
    from ppe.rfdetr_detector import RFDetrDetector

    install_fake_rfdetr(monkeypatch, FakeModel(["glove", "person"]))
    det = RFDetrDetector(cfg(), ppe("glove"))
    assert det.names == {0: "glove", 1: "person"}


def test_no_class_names_attribute_raises_rather_than_silently_empty(monkeypatch):
    """An empty vocabulary would make every configured class read as
    unavailable and force the station to DEGRADED — stopping here, loudly,
    is the safer failure."""
    from ppe.rfdetr_detector import RFDetrDetector

    model = FakeModel(["glove"])
    del model.class_names
    install_fake_rfdetr(monkeypatch, model)
    with pytest.raises(RuntimeError, match="class_names"):
        RFDetrDetector(cfg(), ppe("glove"))


def test_a_configured_class_missing_from_the_checkpoint_is_reported(monkeypatch):
    from ppe.rfdetr_detector import RFDetrDetector

    install_fake_rfdetr(monkeypatch, FakeModel(["glove"]))
    det = RFDetrDetector(cfg(), ppe("glove", "hard_hat"))
    assert det.missing == ["hard_hat"]


def test_the_loaded_path_is_the_configured_weights(monkeypatch):
    from ppe.rfdetr_detector import RFDetrDetector

    model = FakeModel(["glove"])
    install_fake_rfdetr(monkeypatch, model)
    RFDetrDetector(cfg(weights="my_checkpoint.pth"), ppe("glove"))
    assert model.loaded_from == "my_checkpoint.pth"


def test_batches_is_always_false_there_is_no_batch_api(monkeypatch):
    from ppe.rfdetr_detector import RFDetrDetector

    install_fake_rfdetr(monkeypatch, FakeModel(["glove"]))
    det = RFDetrDetector(cfg(), ppe("glove"))
    assert det.batches is False
    assert det.batch_size == 1


def test_warmup_calls_predict_once_when_configured(monkeypatch):
    from ppe.rfdetr_detector import RFDetrDetector

    model = FakeModel(["glove"])
    install_fake_rfdetr(monkeypatch, model)
    RFDetrDetector(cfg(warmup=True), ppe("glove"))
    assert len(model.calls) == 1


def test_no_warmup_call_when_not_configured(monkeypatch):
    from ppe.rfdetr_detector import RFDetrDetector

    model = FakeModel(["glove"])
    install_fake_rfdetr(monkeypatch, model)
    RFDetrDetector(cfg(warmup=False), ppe("glove"))
    assert len(model.calls) == 0


def test_frames_are_converted_from_bgr_to_rgb_before_predicting(monkeypatch):
    """Every frame in this pipeline is BGR straight from OpenCV; rf-detr's
    own examples predict on RGB. Swapped channels would be a silent, subtle
    correctness bug, not a crash — worth pinning directly rather than
    trusting the conversion call is still there next time this file changes.
    """
    from ppe.rfdetr_detector import RFDetrDetector

    model = FakeModel(["glove"])
    install_fake_rfdetr(monkeypatch, model)
    det = RFDetrDetector(cfg(), ppe("glove"))

    bgr = np.zeros((4, 4, 3), np.uint8)
    bgr[..., 0] = 30  # B
    bgr[..., 1] = 20  # G
    bgr[..., 2] = 10  # R
    det.detect([bgr])

    seen = model.calls[-1]
    assert (seen[..., 0] == 10).all()  # R now channel 0
    assert (seen[..., 1] == 20).all()  # G unchanged
    assert (seen[..., 2] == 30).all()  # B now channel 2


def test_detect_applies_per_class_confidence_floors(monkeypatch):
    from ppe.rfdetr_detector import RFDetrDetector

    result = FakeResult(
        boxes=[(0, 0, 10, 10), (0, 0, 20, 20)],
        confidences=[0.4, 0.6],
        class_ids=[0, 0],
        class_names=["glove", "glove"],
    )
    model = FakeModel(["glove"], result=result)
    install_fake_rfdetr(monkeypatch, model)
    det = RFDetrDetector(cfg(), ppe("glove", conf=0.5))

    dets, _timings = det.detect([np.zeros((4, 4, 3), np.uint8)])
    assert [d.conf for d in dets[0]] == [0.6]  # the 0.4 one was below the floor


def test_detect_drops_detections_of_unconfigured_classes(monkeypatch):
    """Same effect as YOLO's upstream NMS class filter, reached a different
    way (rf-detr has no NMS to restrict) — an unconfigured class must not
    leak through as a Detection just because it cleared the global floor."""
    from ppe.rfdetr_detector import RFDetrDetector

    result = FakeResult(
        boxes=[(0, 0, 10, 10), (0, 0, 20, 20)],
        confidences=[0.9, 0.9],
        class_ids=[0, 1],
        class_names=["glove", "wrench"],
    )
    model = FakeModel(["glove", "wrench"], result=result)
    install_fake_rfdetr(monkeypatch, model)
    det = RFDetrDetector(cfg(), ppe("glove"))  # wrench never configured

    dets, _timings = det.detect([np.zeros((4, 4, 3), np.uint8)])
    assert [d.name for d in dets[0]] == ["glove"]


def test_detect_returns_source_coordinates_and_stage_timings(monkeypatch):
    """predict() resizes internally and returns source-image coordinates —
    there is no letterbox remap step for this backend, unlike YOLO's."""
    from ppe.rfdetr_detector import RFDetrDetector

    result = FakeResult(
        boxes=[(5, 6, 50, 60)], confidences=[0.9], class_ids=[0], class_names=["glove"],
    )
    model = FakeModel(["glove"], result=result)
    install_fake_rfdetr(monkeypatch, model)
    det = RFDetrDetector(cfg(), ppe("glove"))

    dets, timings = det.detect([np.zeros((4, 4, 3), np.uint8)])
    assert dets[0][0].xyxy == (5.0, 6.0, 50.0, 60.0)
    assert set(timings) == {"preprocess", "inference", "postprocess"}
