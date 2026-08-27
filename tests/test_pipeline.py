"""End-to-end wiring: real capture threads, a stubbed model, a fake bus."""

import time

import pytest

import ppe.pipeline as pipeline_mod
from ppe.capture import CameraSet
from ppe.config import CameraCfg, ClassCfg, Config, ModelCfg, PPECfg, TelemetryCfg, TowerCfg
from ppe.tower import Status

from .conftest import StubDetector


def make_config(clip, tmp_path, **telemetry):
    return Config(
        model=ModelCfg(imgsz=320, warmup=False),
        cameras=[CameraCfg("Line A", clip, 320, 240), CameraCfg("Line B", clip, 320, 240)],
        ppe=PPECfg(classes=[ClassCfg("helmet"), ClassCfg("vest")], hold_ms=1000, confirm_frames=1),
        tower=TowerCfg(enabled=False),
        telemetry=TelemetryCfg(csv=str(tmp_path / "latency.csv"), **telemetry),
    )


def run(cfg, monkeypatch, cycles=6, timeout=15.0):
    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    cameras = CameraSet(cfg.cameras).start()
    assert cameras.wait_ready(timeout=10.0), "cameras never delivered a frame"

    seen = []
    pipe = pipeline_mod.Pipeline(cfg, cameras, on_result=seen.append)
    pipe.start()
    deadline = time.monotonic() + timeout
    while len(seen) < cycles and time.monotonic() < deadline:
        time.sleep(0.01)
    pipe.stop()
    cameras.stop()
    assert len(seen) >= cycles, f"only {len(seen)} cycles in {timeout}s"
    return pipe, seen


def test_pipeline_publishes_detections_for_both_cameras(clip, tmp_path, monkeypatch):
    cfg = make_config(clip, tmp_path)
    pipe, seen = run(cfg, monkeypatch)

    last = seen[-1]
    assert len(last.detections) == 2
    assert all(d[0].name == "helmet" for d in last.detections)
    assert last.seqs[0] > 0 and last.seqs[1] > 0
    assert pipe.infer_fps > 0


def test_missing_required_ppe_reaches_the_status_and_the_ui_rows(clip, tmp_path, monkeypatch):
    cfg = make_config(clip, tmp_path)
    pipe, seen = run(cfg, monkeypatch)

    last = seen[-1]
    assert last.status is Status.VIOLATION  # the stub never reports a vest
    assert last.missing == ["Vest"]
    rows = {c.name: c for c in last.classes}
    assert rows["helmet"].present and not rows["vest"].present
    assert rows["helmet"].conf == pytest.approx(0.9)


def test_both_cameras_share_one_batched_inference(clip, tmp_path, monkeypatch):
    cfg = make_config(clip, tmp_path)
    pipe, _ = run(cfg, monkeypatch, cycles=8)
    # Two live cameras should mostly be served by one call each cycle.
    assert max(pipe.detector.batch_sizes) == 2


def test_frames_are_dropped_not_queued(clip, tmp_path, monkeypatch):
    """The detector must never process a frame it has already seen."""
    cfg = make_config(clip, tmp_path)
    pipe, seen = run(cfg, monkeypatch, cycles=8)
    seqs = [r.seqs[0] for r in seen]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


def test_every_cycle_is_logged_with_a_full_stage_breakdown(clip, tmp_path, monkeypatch):
    cfg = make_config(clip, tmp_path)
    pipe, seen = run(cfg, monkeypatch)

    snap = pipe.metrics.snapshot()
    for stage in ("wait", "preprocess", "inference", "postprocess", "logic", "relay"):
        assert snap[stage]["n"] > 0, f"{stage} never recorded"
    assert snap["inference"]["p50"] == pytest.approx(8.0)
    # End-to-end starts at the camera grab, so it covers every stage.
    assert snap["end_to_end"]["p50"] >= snap["inference"]["p50"]
    assert seen[-1].latency_ms > 0

    lines = (tmp_path / "latency.csv").read_text().splitlines()
    assert len(lines) > len(seen)  # header + one row per camera per cycle
    assert "Line A" in lines[1] or "Line B" in lines[1]


def test_reconfigure_adopts_an_edited_class_list_live(clip, tmp_path, monkeypatch):
    cfg = make_config(clip, tmp_path)
    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    cameras = CameraSet(cfg.cameras).start()
    cameras.wait_ready(timeout=10.0)
    pipe = pipeline_mod.Pipeline(cfg, cameras)
    pipe.start()
    try:
        cfg.ppe.classes = [ClassCfg("helmet")]  # the user unticks "vest"
        pipe.reconfigure()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if pipe.result is not None and pipe.result.status is Status.OK:
                break
            time.sleep(0.01)
        assert pipe.result.status is Status.OK
        assert [c.name for c in pipe.result.classes] == ["helmet"]
    finally:
        pipe.stop()
        cameras.stop()


def test_no_camera_leaves_the_station_degraded(tmp_path, monkeypatch):
    cfg = make_config("no-such-device.avi", tmp_path)
    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    monkeypatch.setattr(pipeline_mod, "STALE_AFTER", 0.05)
    cameras = CameraSet(cfg.cameras).start()
    pipe = pipeline_mod.Pipeline(cfg, cameras)
    pipe.start()
    time.sleep(0.5)
    pipe.stop()
    cameras.stop()
    assert pipe.result is not None and pipe.result.status is Status.DEGRADED


def test_going_offline_clears_the_overlays_and_does_not_spin(clip, tmp_path, monkeypatch):
    """A dead feed must not keep showing the boxes from its last good frame."""
    cfg = make_config(clip, tmp_path)
    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    monkeypatch.setattr(pipeline_mod, "STALE_AFTER", 0.05)
    cameras = CameraSet(cfg.cameras).start()
    assert cameras.wait_ready(timeout=10.0)

    published = []
    pipe = pipeline_mod.Pipeline(cfg, cameras, on_result=published.append)
    pipe.start()
    while not published:
        time.sleep(0.01)
    assert any(published[-1].detections), "expected boxes while the feed is live"

    cameras.stop()  # pull the cameras out from under the pipeline
    time.sleep(1.0)
    before = len(published)
    time.sleep(1.0)
    after = len(published)
    pipe.stop()

    assert published[-1].status is Status.DEGRADED
    assert not any(published[-1].detections), "stale boxes left on a dead feed"
    # Throttled to ~1/OFFLINE_PERIOD, not once per poll of the 1 ms loop.
    assert (after - before) <= 2 / pipeline_mod.OFFLINE_PERIOD


# -- subject gating end to end ---------------------------------------------


def gated_config(clip, tmp_path, dets):
    """A pipeline whose stub reports a fixed scene, with person gating on."""
    cfg = make_config(clip, tmp_path)
    cfg.ppe.classes = [
        ClassCfg("helmet"), ClassCfg("vest"), ClassCfg("person", required=False)
    ]
    cfg.ppe.subject = "person"
    cfg.ppe.containment = 0.5
    cfg.ppe.confirm_frames = 1

    class Scene(StubDetector):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.dets = dets

    return cfg, Scene


def run_scene(cfg, scene_cls, monkeypatch, cycles=4):
    monkeypatch.setattr(pipeline_mod, "Detector", scene_cls)
    cameras = CameraSet(cfg.cameras).start()
    assert cameras.wait_ready(timeout=10.0)
    seen = []
    pipe = pipeline_mod.Pipeline(cfg, cameras, on_result=seen.append)
    pipe.start()
    deadline = time.monotonic() + 15.0
    while len(seen) < cycles and time.monotonic() < deadline:
        time.sleep(0.01)
    pipe.stop()
    cameras.stop()
    assert len(seen) >= cycles
    return seen[-1]


def box(name, xyxy, conf=0.9):
    from ppe.detector import Detection

    return Detection(name, conf, xyxy)


NEAR = box("person", (100.0, 100.0, 500.0, 900.0))
FAR = box("person", (600.0, 300.0, 700.0, 500.0))


def test_an_empty_cell_puts_the_station_on_standby(clip, tmp_path, monkeypatch):
    cfg, scene = gated_config(clip, tmp_path, [box("helmet", (10.0, 10.0, 40.0, 40.0))])
    result = run_scene(cfg, scene, monkeypatch)
    assert result.status is Status.STANDBY
    assert result.missing == []
    assert all(s is None for s in result.subjects)
    # The helmet is still reported, just not counted.
    assert result.detections == [[], []]
    assert all(any(d.name == "helmet" for d in ig) for ig in result.ignored)


def test_ppe_on_the_subject_passes(clip, tmp_path, monkeypatch):
    cfg, scene = gated_config(
        clip, tmp_path,
        [NEAR, box("helmet", (200.0, 150.0, 300.0, 250.0)),
         box("vest", (150.0, 300.0, 450.0, 600.0))],
    )
    result = run_scene(cfg, scene, monkeypatch)
    assert result.status is Status.OK
    assert all(s is not None for s in result.subjects)


def test_gear_on_a_bystander_does_not_dress_the_subject(clip, tmp_path, monkeypatch):
    """The far person's vest must not satisfy the near person's requirement."""
    cfg, scene = gated_config(
        clip, tmp_path,
        [NEAR, FAR,
         box("helmet", (200.0, 150.0, 300.0, 250.0)),   # on the subject
         box("vest", (610.0, 320.0, 690.0, 480.0))],    # on the bystander
    )
    result = run_scene(cfg, scene, monkeypatch)
    assert result.status is Status.VIOLATION
    assert result.missing == ["Vest"]


def test_only_the_largest_person_is_taken_as_the_subject(clip, tmp_path, monkeypatch):
    cfg, scene = gated_config(clip, tmp_path, [FAR, NEAR])
    result = run_scene(cfg, scene, monkeypatch)
    for subject in result.subjects:
        assert subject is not None and subject.xyxy == NEAR.xyxy
    assert all(any(d.xyxy == FAR.xyxy for d in ig) for ig in result.ignored)


def test_gating_can_be_turned_off(clip, tmp_path, monkeypatch):
    """With no subject class the station checks everything, as it used to."""
    cfg, scene = gated_config(
        clip, tmp_path,
        [box("helmet", (10.0, 10.0, 40.0, 40.0)), box("vest", (50.0, 50.0, 90.0, 90.0))],
    )
    cfg.ppe.subject = ""
    result = run_scene(cfg, scene, monkeypatch)
    assert result.status is Status.OK
    assert result.ignored == [[], []]
