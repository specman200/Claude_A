"""Tests for the parts that do not need a camera or a screen."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import pytest

import capture
from capture import (
    Countdown,
    CaptureError,
    Frame,
    clamp_delay,
    collect,
    countdown_text,
    parse_args,
    shot_paths,
    skew_ms,
)


class FakeStream:
    def __init__(self, label, frame=None):
        self.label = label
        self._frame = frame

    def latest(self):
        return self._frame


def test_delay_defaults_to_two_seconds():
    assert parse_args([]).delay == 2.0


def test_clamp_delay_reads_the_timer_box():
    assert clamp_delay("3.5") == 3.5
    assert clamp_delay("") == capture.DEFAULT_DELAY
    assert clamp_delay("later") == capture.DEFAULT_DELAY
    assert clamp_delay(None) == capture.DEFAULT_DELAY
    assert clamp_delay(float("nan")) == capture.DEFAULT_DELAY


def test_clamp_delay_stays_in_range():
    assert clamp_delay(-4) == 0.0
    assert clamp_delay(10_000) == capture.MAX_DELAY


def test_countdown_runs_down_and_expires():
    timer = Countdown.start(2.0, now=100.0)
    assert timer.remaining(100.0) == 2.0
    assert timer.remaining(101.5) == 0.5
    assert not timer.expired(101.999)
    assert timer.expired(102.0)
    assert timer.remaining(105.0) == 0.0  # never runs negative


def test_countdown_text_counts_whole_seconds():
    assert countdown_text(2.0) == "2"
    assert countdown_text(1.01) == "2"
    assert countdown_text(1.0) == "1"
    assert countdown_text(0.01) == "1"  # the last tick still reads 1, not 0


def test_shot_paths_pair_shares_a_stem(tmp_path):
    when = datetime(2026, 9, 1, 15, 4, 5)
    left, right = shot_paths(tmp_path, ["cam0", "cam1"], when)
    assert left.name == "20260901-150405_cam0.jpg"
    assert right.name == "20260901-150405_cam1.jpg"


def test_shot_paths_do_not_overwrite_within_a_second(tmp_path):
    when = datetime(2026, 9, 1, 15, 4, 5)
    first = shot_paths(tmp_path, ["cam0", "cam1"], when)
    for path in first:
        path.write_bytes(b"")
    second = shot_paths(tmp_path, ["cam0", "cam1"], when)
    assert [p.name for p in second] == [
        "20260901-150405-2_cam0.jpg",
        "20260901-150405-2_cam1.jpg",
    ]
    # a half-written pair counts as taken too
    second[0].write_bytes(b"")
    third = shot_paths(tmp_path, ["cam0", "cam1"], when)
    assert third[0].name == "20260901-150405-3_cam0.jpg"


def test_shot_paths_honour_the_format(tmp_path):
    paths = shot_paths(tmp_path, ["a", "b"], datetime(2026, 9, 1), ".png")
    assert all(path.suffix == ".png" for path in paths)


def test_collect_returns_the_newest_frame_from_each():
    streams = [FakeStream("cam0", Frame("L", 9.8)), FakeStream("cam1", Frame("R", 9.9))]
    frames = collect(streams, clock=lambda: 10.0)
    assert [frame.image for frame in frames] == ["L", "R"]
    assert skew_ms(frames) == pytest.approx(100.0)


def test_collect_names_the_camera_that_never_started():
    streams = [FakeStream("cam0", Frame("L", 10.0)), FakeStream("cam1")]
    with pytest.raises(CaptureError, match="cam1"):
        collect(streams, clock=lambda: 10.0)


def test_collect_refuses_a_stale_frame():
    streams = [FakeStream("cam0", Frame("L", 10.0)), FakeStream("cam1", Frame("R", 1.0))]
    with pytest.raises(CaptureError, match="cam1 stopped"):
        collect(streams, clock=lambda: 10.0, stale_after=2.0)


def test_labels_follow_the_camera_indices():
    assert parse_args(["--cameras", "0", "2"]).labels == ["cam0", "cam2"]
    assert parse_args(["--labels", "left", "right"]).labels == ["left", "right"]
    assert parse_args(["--demo"]).labels == ["demoA", "demoB"]


def test_two_cameras_cannot_share_a_label():
    with pytest.raises(SystemExit):
        parse_args(["--cameras", "1", "1"])


def test_out_of_range_delay_is_clamped_at_the_command_line():
    assert parse_args(["--delay", "-1"]).delay == 0.0
    assert parse_args(["--delay", "1e9"]).delay == capture.MAX_DELAY
