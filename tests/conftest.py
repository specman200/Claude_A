"""Shared fixtures: a synthetic camera and a model stub."""

import os
import time

import cv2
import numpy as np
import pytest

from ppe.detector import Detection

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def clip(tmp_path):
    """A 320x240 video file standing in for a camera."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    assert writer.isOpened(), "no MJPG encoder available"
    for i in range(600):
        writer.write(np.full((240, 320, 3), i * 2 % 255, np.uint8))
    writer.release()
    return str(path)


class StubDetector:
    """Reports whatever the test tells it to, and spends the time it claims."""

    names = {0: "helmet", 1: "vest"}
    missing: list[str] = []
    batches = False  # CPU default: one camera per cycle

    def __init__(self, *_args, **_kwargs):
        self.dets = [Detection("helmet", 0.9, (10.0, 10.0, 100.0, 100.0))]
        self.calls = 0
        self.batch_sizes = []
        self.floors: dict[str, float | None] = {}

    def detect(self, images):
        self.calls += 1
        self.batch_sizes.append(len(images))
        time.sleep(0.008)  # spend the time we claim, so end-to-end stays honest
        return (
            [list(self.dets) for _ in images],
            {"preprocess": 1.0, "inference": 8.0, "postprocess": 0.5},
        )

    def set_classes(self, ppe):
        # Record what the detector was reconfigured with, so tests can prove
        # a UI edit actually reached it.
        self.floors = {c.name: c.conf for c in ppe.classes}
        return list(self.missing)
