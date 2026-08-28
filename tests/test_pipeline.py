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


def test_batching_puts_both_cameras_in_one_call(clip, tmp_path, monkeypatch):
    """When batching is on (a GPU), both cameras share a single call."""
    cfg = make_config(clip, tmp_path)
    monkeypatch.setattr(StubDetector, "batches", True)
    pipe, _ = run(cfg, monkeypatch, cycles=8)
    assert max(pipe.detector.batch_sizes) == 2


def test_frames_are_dropped_not_queued(clip, tmp_path, monkeypatch):
    """The detector must never go back to a frame it has already passed."""
    cfg = make_config(clip, tmp_path)
    pipe, seen = run(cfg, monkeypatch, cycles=8)
    seqs = [r.seqs[0] for r in seen]
    # Round-robin repeats a camera's last sequence on the cycles it is not
    # served, so the invariant is non-decreasing, never re-processed.
    assert seqs == sorted(seqs)
    assert seqs[-1] > seqs[0], "camera 0 never advanced"
    assert pipe.detector.calls <= sum(1 for _ in seen) + 1


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


# -- CPU scheduling --------------------------------------------------------
# On a CPU a batch of two costs about twice a batch of one and makes each frame
# wait for the other's result, and the ONNX/OpenVINO exports are batch-1 only.


def frames(*indices):
    import numpy as np

    from ppe.capture import Frame

    return [Frame(i, i + 1, np.zeros((4, 4, 3), np.uint8), 0.0) for i in indices]


def bare_pipeline(cfg, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    cameras = CameraSet(cfg.cameras)  # never started; we only exercise _take
    pipe = pipeline_mod.Pipeline(cfg, cameras)
    assert pipe._load()  # load synchronously; the thread never runs here
    return pipe


def test_one_camera_per_cycle_when_not_batching(clip, tmp_path, monkeypatch):
    pipe = bare_pipeline(make_config(clip, tmp_path), monkeypatch)
    assert pipe.detector.batches is False
    assert len(pipe._take(frames(0, 1))) == 1


def test_cameras_take_turns_so_neither_starves(clip, tmp_path, monkeypatch):
    pipe = bare_pipeline(make_config(clip, tmp_path), monkeypatch)
    served = [pipe._take(frames(0, 1))[0].index for _ in range(6)]
    assert served == [0, 1, 0, 1, 0, 1]


def test_a_lone_fresh_camera_is_served_immediately(clip, tmp_path, monkeypatch):
    """The other camera having no new frame must not stall this one."""
    pipe = bare_pipeline(make_config(clip, tmp_path), monkeypatch)
    for _ in range(3):
        assert pipe._take(frames(1))[0].index == 1


def test_batching_still_takes_every_fresh_frame(clip, tmp_path, monkeypatch):
    pipe = bare_pipeline(make_config(clip, tmp_path), monkeypatch)
    pipe.detector.batches = True
    assert len(pipe._take(frames(0, 1))) == 2


def test_round_robin_still_updates_both_cameras(clip, tmp_path, monkeypatch):
    """Halving the per-cycle cost must not cost a camera its detections."""
    cfg = make_config(clip, tmp_path)
    pipe, seen = run(cfg, monkeypatch, cycles=10)
    assert max(r.seqs[0] for r in seen) > 0
    assert max(r.seqs[1] for r in seen) > 0
    assert max(pipe.detector.batch_sizes) == 1  # one image per call


# -- deferred model loading --------------------------------------------
# Loading the model can take real, human-noticeable time. It must not block
# whoever constructs the Pipeline, so a UI can show a window and live video
# immediately instead of appearing to hang.


def test_construction_does_not_load_the_model(clip, tmp_path, monkeypatch):
    cfg = make_config(clip, tmp_path)
    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    pipe = pipeline_mod.Pipeline(cfg, CameraSet(cfg.cameras))
    assert pipe.detector is None
    assert not pipe.ready.is_set()


def test_starting_the_thread_loads_it_and_fires_on_ready(clip, tmp_path, monkeypatch):
    cfg = make_config(clip, tmp_path)
    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    cameras = CameraSet(cfg.cameras).start()
    ready = []
    pipe = pipeline_mod.Pipeline(cfg, cameras, on_ready=lambda: ready.append(True))
    pipe.start()
    deadline = time.monotonic() + 10.0
    while not ready and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready and pipe.ready.is_set()
    assert pipe.detector is not None
    pipe.stop()
    cameras.stop()


def test_a_model_that_fails_to_load_reports_the_error_and_stops_cleanly(
    clip, tmp_path, monkeypatch
):
    """A bad config must fail loudly, not hang forever looking like it is
    still busy — and must not crash the process either."""
    cfg = make_config(clip, tmp_path)

    class Broken(StubDetector):
        def __init__(self, *a, **k):
            raise RuntimeError("no such weights file")

    monkeypatch.setattr(pipeline_mod, "Detector", Broken)
    cameras = CameraSet(cfg.cameras).start()
    errors = []
    pipe = pipeline_mod.Pipeline(cfg, cameras, on_error=errors.append)
    pipe.start()
    pipe.join(timeout=10.0)

    assert not pipe.is_alive()
    assert len(errors) == 1 and "no such weights file" in str(errors[0])
    assert pipe.detector is None and not pipe.ready.is_set()
    assert isinstance(pipe.error, RuntimeError)
    cameras.stop()


def test_reconfigure_before_the_model_loads_is_a_harmless_no_op(clip, tmp_path, monkeypatch):
    cfg = make_config(clip, tmp_path)
    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    pipe = pipeline_mod.Pipeline(cfg, CameraSet(cfg.cameras))
    pipe.reconfigure()  # must not raise even though nothing has loaded yet


def test_shutdown_before_the_model_loads_does_not_crash_on_a_missing_tower(
    clip, tmp_path, monkeypatch
):
    """stop() can land while the thread is still inside _load(); _shutdown()
    must not assume self.tower exists yet."""
    cfg = make_config(clip, tmp_path)
    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    cameras = CameraSet(cfg.cameras).start()
    pipe = pipeline_mod.Pipeline(cfg, cameras)
    pipe.start()
    pipe.stop()  # stop immediately; loading may or may not have finished
    cameras.stop()
