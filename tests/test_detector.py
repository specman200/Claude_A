"""Real-model tests: boxes must come back in source pixels, at any capture size.

These need `ultralytics` plus the weights named by WEIGHTS; they skip if either
is unavailable, so the rest of the suite still runs on a bare checkout.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from ppe.config import ClassCfg, ModelCfg, PPECfg

WEIGHTS = "yolo11s.pt"
pytest.importorskip("ultralytics")


@pytest.fixture(scope="module")
def scene():
    from ultralytics import ASSETS

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    assert image is not None
    return image


@pytest.fixture(scope="module")
def detector():
    from ppe.detector import Detector

    ppe = PPECfg(classes=[ClassCfg("person"), ClassCfg("bus")])
    try:
        return Detector(ModelCfg(weights=WEIGHTS, imgsz=640, conf=0.35, warmup=False), ppe)
    except Exception as exc:  # weights not present / no network
        pytest.skip(f"{WEIGHTS} unavailable: {exc}")


def norm(dets, w, h):
    """Boxes as fractions of the frame, so different capture sizes compare."""
    return sorted(
        (d.name, round(d.xyxy[0] / w, 2), round(d.xyxy[1] / h, 2),
         round(d.xyxy[2] / w, 2), round(d.xyxy[3] / h, 2))
        for d in dets
    )


def test_detects_the_configured_classes(detector, scene):
    dets, timings = detector.detect([scene])
    assert dets[0], "nothing detected in the reference image"
    assert {d.name for d in dets[0]} <= {"person", "bus"}
    assert set(timings) == {"preprocess", "inference", "postprocess"}
    assert all(v > 0 for v in timings.values())


def test_boxes_are_in_source_pixels_not_canvas_pixels(detector, scene):
    """The bus fills most of a 810x1080 frame; a canvas-space box could not."""
    h, w = scene.shape[:2]
    dets = detector.detect([scene])[0][0]
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h
    assert max(d.xyxy[3] for d in dets) > detector.cfg.imgsz  # beyond the 640 canvas


def test_inverse_matches_ultralytics_scale_boxes(detector, scene):
    """Our box inverse must agree with Ultralytics' own, to the pixel."""
    from ultralytics.utils import ops

    from ppe.letterbox import letterbox

    canvas, meta = letterbox(scene, 640)
    raw = detector.model.predict(
        canvas, imgsz=640, conf=detector._conf, iou=detector.cfg.iou,
        classes=detector.class_ids, verbose=False,
    )[0].boxes.xyxy.cpu().numpy()

    mine = meta.to_source(raw)
    theirs = ops.scale_boxes((640, 640), raw.copy(), scene.shape[:2])
    assert mine == pytest.approx(theirs, abs=0.5)


def test_agrees_with_the_native_predict_path(detector, scene):
    """Same objects as Ultralytics' own path.

    We letterbox to a square so both cameras batch into one call; the native
    path pads only to the next stride multiple. Identical geometry either way,
    but a different input tensor shape, so boxes land within a fraction of a
    percent of each other rather than exactly on top.
    """
    ours = detector.detect([scene])[0][0]
    native = detector.model.predict(
        scene, imgsz=640, conf=detector._conf, iou=detector.cfg.iou,
        classes=detector.class_ids, verbose=False,
    )[0]

    h, w = scene.shape[:2]
    theirs = [
        (detector.names[int(c)], *box)
        for box, c in zip(native.boxes.xyxy.cpu().numpy(), native.boxes.cls.cpu().numpy())
    ]
    assert len(ours) == len(theirs), "different number of detections"
    mine = np.array(sorted(d.xyxy for d in ours))
    other = np.array(sorted(tuple(t[1:]) for t in theirs))
    assert mine == pytest.approx(other, abs=0.01 * max(w, h))


@pytest.mark.parametrize("scale", [0.5, 0.75, 1.5])
def test_changing_capture_size_keeps_the_same_detections(detector, scene, scale):
    """Capture resolution may be altered: the objects found must not change."""
    h, w = scene.shape[:2]
    resized = cv2.resize(scene, (int(w * scale), int(h * scale)))

    base = norm(detector.detect([scene])[0][0], w, h)
    other = norm(detector.detect([resized])[0][0], int(w * scale), int(h * scale))

    assert [d[0] for d in base] == [d[0] for d in other], "class set changed with capture size"
    for a, b in zip(base, other):
        assert a[1:] == pytest.approx(b[1:], abs=0.03)  # within 3% of the frame


def test_batching_two_cameras_matches_detecting_them_singly(detector, scene):
    other = cv2.flip(scene, 1)
    batched = detector.detect([scene, other])[0]
    singly = [detector.detect([scene])[0][0], detector.detect([other])[0][0]]

    h, w = scene.shape[:2]
    for got, want in zip(batched, singly):
        assert norm(got, w, h) == norm(want, w, h)


def test_per_class_confidence_floors_are_applied(detector, scene):
    from ppe.detector import Detector

    strict = Detector(
        ModelCfg(weights=WEIGHTS, imgsz=640, conf=0.35, warmup=False),
        PPECfg(classes=[ClassCfg("person", conf=0.9), ClassCfg("bus", conf=0.35)]),
    )
    # NMS still runs at the lowest floor, so the bus is unaffected...
    assert strict._conf == pytest.approx(0.35)
    loose = detector.detect([scene])[0][0]
    tight = strict.detect([scene])[0][0]
    assert [d.name for d in tight].count("bus") == [d.name for d in loose].count("bus")
    # ...while the raised floor drops the low-confidence people.
    assert all(d.conf >= 0.9 for d in tight if d.name == "person")
    assert len(tight) <= len(loose)


def test_classes_absent_from_the_model_are_reported(detector):
    from ppe.detector import Detector

    d = Detector(
        ModelCfg(weights=WEIGHTS, warmup=False),
        PPECfg(classes=[ClassCfg("person"), ClassCfg("hard_hat")]),
    )
    assert d.missing == ["hard_hat"]
    assert d.class_ids == [0]  # only 'person' is filtered for in NMS


def test_a_blank_frame_yields_no_detections(detector):
    blank = np.full((720, 1280, 3), 114, np.uint8)
    assert detector.detect([blank])[0][0] == []


# -- the shipped fine-tuned model ------------------------------------------


def test_the_shipped_config_and_the_shipped_weights_agree():
    """Catches a config that has drifted from the weights it points at."""
    from ppe.config import Config
    from ppe.detector import Detector

    cfg = Config.load("config.yaml")
    if not Path(cfg.model.weights).is_file():
        pytest.skip(f"{cfg.model.weights} not present")

    detector = Detector(ModelCfg(weights=cfg.model.weights, warmup=False), cfg.ppe)
    assert detector.missing == [], (
        f"config lists classes the model does not have: {detector.missing}"
    )
    assert set(detector.names.values()) == {c.name for c in cfg.ppe.classes}, (
        "the model has classes the config never mentions, or vice versa"
    )


def test_the_shipped_model_returns_boxes_in_source_pixels(scene):
    from ppe.config import Config
    from ppe.detector import Detector

    cfg = Config.load("config.yaml")
    if not Path(cfg.model.weights).is_file():
        pytest.skip(f"{cfg.model.weights} not present")

    detector = Detector(ModelCfg(weights=cfg.model.weights, conf=0.25, warmup=False), cfg.ppe)
    dets = detector.detect([scene])[0][0]
    h, w = scene.shape[:2]
    assert dets, "the model found nothing at all in the reference image"
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
