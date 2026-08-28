"""FOURCC must be set before resolution, and before the driver locks in a
raw format neither USB port can carry for two cameras at once.
"""

import cv2

from ppe.config import CameraCfg


class FakeCap:
    """Records every .set() call, in order, without touching real hardware."""

    def __init__(self, *_a, **_k):
        self.calls: list[tuple[int, float]] = []
        self.props = {cv2.CAP_PROP_FOURCC: 0.0}

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.calls.append((prop, value))
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def read(self):
        return False, None

    def release(self):
        pass

    def getBackendName(self):  # noqa: N802 — matches cv2's own method name
        return "FAKE"


def test_fourcc_is_set_before_resolution(monkeypatch):
    """DirectShow/MSMF pick which resolutions exist based on pixel format;
    setting it after width/height can silently lock in the wrong mode."""
    from ppe.capture import Camera

    fake = FakeCap()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: fake)
    cam = Camera(0, CameraCfg("Cam", 0, 1280, 720, 30, fourcc="MJPG"))
    cam._open()

    order = [p for p, _ in fake.calls]
    assert order.index(cv2.CAP_PROP_FOURCC) < order.index(cv2.CAP_PROP_FRAME_WIDTH)
    assert order.index(cv2.CAP_PROP_FOURCC) < order.index(cv2.CAP_PROP_FRAME_HEIGHT)


def test_fourcc_of_empty_string_leaves_the_driver_default(monkeypatch):
    from ppe.capture import Camera

    fake = FakeCap()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: fake)
    cam = Camera(0, CameraCfg("Cam", 0, 1280, 720, 30, fourcc=""))
    cam._open()

    assert cv2.CAP_PROP_FOURCC not in [p for p, _ in fake.calls]


def test_the_requested_fourcc_is_the_mjpg_code(monkeypatch):
    from ppe.capture import Camera

    fake = FakeCap()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: fake)
    cam = Camera(0, CameraCfg("Cam", 0, fourcc="MJPG"))
    cam._open()

    value = dict(fake.calls)[cv2.CAP_PROP_FOURCC]
    assert int(value) == cv2.VideoWriter_fourcc(*"MJPG")


def test_camera_default_config_requests_mjpg():
    """The shipped default must not be the raw format that overruns USB."""
    assert CameraCfg().fourcc == "MJPG"


def test_camcheck_warns_when_the_driver_ignores_the_request(capsys):
    from ppe.camcheck import probe

    class RawCap(FakeCap):
        """Pretends the driver accepted MJPG but silently kept YUY2."""

        def get(self, prop):
            if prop == cv2.CAP_PROP_FOURCC:
                return float(cv2.VideoWriter_fourcc(*"YUY2"))
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 1280.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 720.0
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            return 0.0

    fake = RawCap()
    orig = cv2.VideoCapture
    cv2.VideoCapture = lambda *a, **k: fake
    try:
        probe(CameraCfg("Cam", 0, 1280, 720, 30, fourcc="MJPG"), timeout=0.1, save=None, tag="t")
    finally:
        cv2.VideoCapture = orig

    out = capsys.readouterr().out
    assert "driver gave YUY2 instead" in out
    assert "MB/s" in out  # names the bandwidth cost, not just the mismatch
