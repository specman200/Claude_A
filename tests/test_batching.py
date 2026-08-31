"""What happens when a model and the config disagree about batch size.

An export is compiled for exactly one frame count and refuses every other,
while a .pt takes any. `model.batch` is therefore a request the model can
decline in either direction, and it declines it inside the runtime — during
warmup, or mid-shift if warmup is off. These pin that the detector settles the
disagreement at construction instead.
"""

import pytest

from ppe.config import ClassCfg, ModelCfg, PPECfg


class FakeBoxes:
    xyxy = conf = cls = None

    def __len__(self):
        return 0


class FakeResult:
    boxes = FakeBoxes()


class FakeYOLO:
    """A model that accepts exactly `takes` frames per call, as an export does."""

    takes = 1

    def __init__(self, *_args, **_kwargs):
        self.names = {0: "person"}
        self.calls = []

    def predict(self, canvases, **_kwargs):
        self.calls.append(len(canvases))
        if len(canvases) != self.takes:
            raise RuntimeError(
                f"model input (shape=[{self.takes},3,640,640]) and the tensor "
                f"(shape=({len(canvases)},3,640,640)) are incompatible"
            )
        return [FakeResult() for _ in canvases]


def detector(monkeypatch, takes, batch):
    from ppe.detector import Detector

    monkeypatch.setattr("ultralytics.YOLO", type("Y", (FakeYOLO,), {"takes": takes}))
    cfg = ModelCfg(weights="fake_openvino_model/", imgsz=640, batch=batch, warmup=False)
    return Detector(cfg, PPECfg(classes=[ClassCfg("person")]))


def test_a_batch_1_model_asked_for_two_serves_one_camera_per_cycle(monkeypatch, caplog):
    det = detector(monkeypatch, takes=1, batch=True)
    assert det.batches is False
    assert "refuses 2" in caplog.text


def test_a_batch_2_model_asked_for_one_serves_both_cameras_in_one_call(monkeypatch, caplog):
    """The mirror case: a --batch 2 export refuses a single frame just as hard."""
    det = detector(monkeypatch, takes=2, batch=False)
    assert det.batches is True
    assert "refuses 1" in caplog.text


def test_a_model_that_takes_what_was_asked_is_left_alone(monkeypatch, caplog):
    assert detector(monkeypatch, takes=1, batch=False).batches is False
    assert detector(monkeypatch, takes=2, batch=True).batches is True
    assert "refuses" not in caplog.text


def test_a_model_that_fails_at_every_size_reports_its_own_error(monkeypatch):
    """Both counts failing is a broken model, not a batching problem — that
    error must reach the caller instead of being retold as a size mismatch."""
    from ppe.detector import Detector

    class Broken(FakeYOLO):
        def predict(self, canvases, **_kwargs):
            raise RuntimeError("the weights are rubble")

    monkeypatch.setattr("ultralytics.YOLO", Broken)
    with pytest.raises(RuntimeError, match="rubble"):
        Detector(
            ModelCfg(weights="broken/", imgsz=640, batch=True, warmup=False),
            PPECfg(classes=[ClassCfg("person")]),
        )


def test_settling_costs_one_call_when_the_model_agrees(monkeypatch):
    """The probe doubles as the first warm call; it must not become two."""
    from ppe.detector import Detector

    made = {}

    class Counting(FakeYOLO):
        takes = 1

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made["model"] = self

    monkeypatch.setattr("ultralytics.YOLO", Counting)
    Detector(
        ModelCfg(weights="x_openvino_model/", imgsz=640, batch=False, warmup=False),
        PPECfg(classes=[ClassCfg("person")]),
    )
    assert made["model"].calls == [1]


# -- exporting at a batch the station can then use -------------------------


def test_export_passes_the_requested_batch_through(tmp_path, monkeypatch):
    from ppe import export as export_mod

    seen = {}

    class RecordingYOLO:
        def __init__(self, weights):
            seen["weights"] = weights

        def export(self, **kwargs):
            seen.update(kwargs)
            return "out_openvino_model/"

    monkeypatch.setattr("ultralytics.YOLO", RecordingYOLO)
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"")
    cfg = tmp_path / "config.yaml"
    import yaml

    with open("config.yaml") as fh:
        raw = yaml.safe_load(fh)
    raw["model"]["weights"] = str(pt)
    cfg.write_text(yaml.safe_dump(raw))

    assert export_mod.main(["-c", str(cfg), "--batch", "2"]) == 0
    assert seen["batch"] == 2
    # Staged under its own stem, so a batch-2 export never lands on top of the
    # batch-1 one the shipped config runs.
    assert seen["weights"].endswith("best_b2.pt")
    assert not (tmp_path / "best_b2.pt").exists(), "the staging copy must be cleaned up"


def test_export_rejects_a_nonsense_batch(tmp_path, caplog):
    from ppe import export as export_mod

    pt = tmp_path / "best.pt"
    pt.write_bytes(b"")
    import yaml

    with open("config.yaml") as fh:
        raw = yaml.safe_load(fh)
    raw["model"]["weights"] = str(pt)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(raw))

    assert export_mod.main(["-c", str(cfg), "--batch", "0"]) == 1
    assert "--batch must be >= 1" in caplog.text
