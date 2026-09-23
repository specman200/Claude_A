"""Capture mode: does it keep the right frames, and stay out of the loop?"""

import csv

import numpy as np
import pytest

from ppe.capture import Frame
from ppe.config import DatasetCfg
from ppe.dataset import TRIGGERS, DatasetRecorder
from ppe.detector import Detection
from ppe.tower import ClassState, Status

CLASSES = ["helmet", "vest", "person"]


def frame(index=0, seq=1, w=64, h=48):
    return Frame(index=index, seq=seq, ts=0.0,
                 image=np.full((h, w, 3), 128, dtype=np.uint8))


def rec(tmp_path, triggers, **kw):
    cfg = DatasetCfg(enabled=True, dir=str(tmp_path), triggers=list(triggers),
                     min_gap_sec=0.0, **kw)
    return DatasetRecorder(cfg, CLASSES)


def states(**seen):
    """Class states with the given per-class sighting counts."""
    return [ClassState(n, n.title(), required=n != "person", need=1,
                       seen=seen.get(n, 1)) for n in CLASSES]


def drain(r):
    r.close()
    return sorted(p.name for p in (r.root / "images").glob("*.jpg"))


def index_rows(r):
    with (r.root / "captures.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_the_interval_trigger_keeps_a_frame_on_its_own_clock(tmp_path):
    r = rec(tmp_path, ["interval"], interval_sec=10.0)
    for t in (0.0, 4.0, 9.9, 10.0, 14.0, 20.1):
        r.observe([(frame(), [])], Status.OK, states(), False, t)
    assert len(drain(r)) == 3, "t=0, t=10, t=20.1"


def test_a_violation_is_kept_once_not_once_a_cycle(tmp_path):
    """The edge, not the state. A violation that stands for a minute is one
    frame's worth of information."""
    r = rec(tmp_path, ["violation"])
    r.observe([(frame(), [])], Status.OK, states(), True, 0.0)
    for t in (1.0, 1.1, 1.2, 1.3):
        r.observe([(frame(), [])], Status.VIOLATION, states(), True, t)
    r.observe([(frame(), [])], Status.OK, states(), True, 2.0)
    r.observe([(frame(), [])], Status.VIOLATION, states(), True, 3.0)
    names = drain(r)
    assert len(names) == 2, "two violations, two frames"
    assert all("violation" in n for n in names)


def test_the_motor_starting_and_stopping_are_both_edges(tmp_path):
    r = rec(tmp_path, ["motor_start", "motor_stop"])
    r.observe([(frame(), [])], Status.OK, states(), False, 0.0)
    r.observe([(frame(), [])], Status.OK, states(), True, 1.0)      # start
    r.observe([(frame(), [])], Status.OK, states(), True, 2.0)
    r.observe([(frame(), [])], Status.VIOLATION, states(), False, 3.0)  # stop
    names = drain(r)
    assert [n.split("_")[1] for n in names] == ["motor", "motor"]
    assert "motor_start" in names[0] and "motor_stop" in names[1]


def test_the_miss_trigger_catches_the_frame_the_model_failed_on(tmp_path):
    """The one that matters for chasing nuisance stops: the moment a required
    item went out of sight, not the minute afterwards."""
    r = rec(tmp_path, ["miss"])
    r.observe([(frame(), [])], Status.OK, states(), True, 0.0)
    r.observe([(frame(), [])], Status.OK, states(vest=0), True, 1.0)   # gone
    r.observe([(frame(), [])], Status.OK, states(vest=0), True, 1.1)   # still gone
    r.observe([(frame(), [])], Status.OK, states(), True, 2.0)         # back
    r.observe([(frame(), [])], Status.OK, states(vest=0), True, 3.0)   # gone again
    names = drain(r)
    assert len(names) == 2
    assert all("miss" in n for n in names)


def test_an_optional_class_going_missing_is_not_a_miss(tmp_path):
    r = rec(tmp_path, ["miss"])
    r.observe([(frame(), [])], Status.OK, states(), True, 0.0)
    r.observe([(frame(), [])], Status.OK, states(person=0), True, 1.0)
    assert drain(r) == []


def test_a_class_the_model_lacks_is_not_a_miss(tmp_path):
    """That is a DEGRADED problem, and it would otherwise fire on every
    single cycle for as long as the model stayed wrong."""
    r = rec(tmp_path, ["miss"])
    missing = states()
    missing[1].available = False
    missing[1].seen = 0
    r.observe([(frame(), [])], Status.OK, states(), True, 0.0)
    r.observe([(frame(), [])], Status.DEGRADED, missing, True, 1.0)
    assert drain(r) == []


def test_detections_are_written_beside_the_image_as_yolo_labels(tmp_path):
    r = rec(tmp_path, ["violation"])
    dets = [
        Detection("helmet", 0.9, (0.0, 0.0, 32.0, 24.0)),   # top-left quarter
        Detection("badger", 0.9, (0.0, 0.0, 10.0, 10.0)),   # not in the class list
    ]
    r.observe([(frame(w=64, h=48), dets)], Status.VIOLATION, states(), True, 0.0)
    r.close()

    label = (r.root / "labels").glob("*.txt").__next__().read_text().split()
    assert label[0] == "0", "helmet is index 0 in the dataset's class list"
    assert [float(v) for v in label[1:]] == pytest.approx([0.25, 0.25, 0.5, 0.5])
    assert "badger" not in " ".join(label), "a class the dataset has no index for"

    yaml = (r.root / "data.yaml").read_text()
    assert "nc: 3" in yaml and "0: helmet" in yaml
    assert "not ground truth" in yaml, "the labels are the model's opinion, and say so"


def test_both_camera_views_are_kept_for_one_event(tmp_path):
    r = rec(tmp_path, ["violation"])
    r.observe([(frame(index=0), []), (frame(index=1), [])],
              Status.VIOLATION, states(), True, 0.0)
    names = drain(r)
    assert [n[-8:] for n in names] == ["cam0.jpg", "cam1.jpg"]


def test_the_rate_limit_refuses_a_second_capture_too_soon(tmp_path):
    cfg = DatasetCfg(enabled=True, dir=str(tmp_path), triggers=["violation"],
                     min_gap_sec=5.0)
    r = DatasetRecorder(cfg, CLASSES)
    r.observe([(frame(), [])], Status.VIOLATION, states(), True, 0.0)
    r.observe([(frame(), [])], Status.OK, states(), True, 1.0)
    r.observe([(frame(), [])], Status.VIOLATION, states(), True, 2.0)   # too soon
    r.observe([(frame(), [])], Status.OK, states(), True, 3.0)
    r.observe([(frame(), [])], Status.VIOLATION, states(), True, 8.0)   # far enough
    assert len(drain(r)) == 2


def test_a_refused_capture_still_moves_the_edge_on(tmp_path):
    """Otherwise the next cycle thinks the violation only just started, and
    the rate limit turns into a delay rather than a limit."""
    cfg = DatasetCfg(enabled=True, dir=str(tmp_path), triggers=["violation"],
                     min_gap_sec=100.0)
    r = DatasetRecorder(cfg, CLASSES)
    r.observe([(frame(), [])], Status.VIOLATION, states(), True, 0.0)   # kept
    for t in (1.0, 2.0, 3.0):                                           # refused
        r.observe([(frame(), [])], Status.VIOLATION, states(), True, t)
    assert len(drain(r)) == 1


def test_the_image_cap_is_a_hard_stop(tmp_path):
    cfg = DatasetCfg(enabled=True, dir=str(tmp_path), triggers=["interval"],
                     interval_sec=0.0, min_gap_sec=0.0, max_images=3)
    r = DatasetRecorder(cfg, CLASSES)
    for t in range(20):
        r.observe([(frame(), [])], Status.OK, states(), False, float(t))
    assert r.full
    assert len(drain(r)) == 3


def test_the_index_records_why_each_frame_was_kept(tmp_path):
    r = rec(tmp_path, ["violation", "motor_start"])
    r.observe([(frame(), [])], Status.OK, states(), False, 0.0)
    r.observe([(frame(), [])], Status.OK, states(), True, 1.0)
    r.observe([(frame(), [])], Status.VIOLATION, states(), True, 2.0)
    r.close()
    rows = index_rows(r)
    assert [row["trigger"] for row in rows] == ["motor_start", "violation"]
    assert [row["status"] for row in rows] == ["ok", "violation"]
    assert [row["motor"] for row in rows] == ["1", "1"]


def test_an_unknown_trigger_in_the_config_is_simply_not_armed(tmp_path):
    r = rec(tmp_path, ["violation", "teleport"])
    assert r.triggers == {"violation"}
    assert set(TRIGGERS) >= r.triggers


def test_a_write_that_fails_does_not_take_the_station_down(tmp_path, monkeypatch):
    """A full disk is a bad day, not a stopped machine."""
    import ppe.dataset as dataset_mod

    r = rec(tmp_path, ["violation"])
    monkeypatch.setattr(dataset_mod.cv2, "imwrite",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    r.observe([(frame(), [])], Status.VIOLATION, states(), True, 0.0)
    r.close()                       # must not raise
    assert r.saved == 1, "it was queued; the write is what failed"
