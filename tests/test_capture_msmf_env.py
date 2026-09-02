"""MSMF's hardware-transform negotiation can stall opening a UVC webcam for
minutes. The fix is one env var, read once by OpenCV on first MSMF use — so
it has to be set before that happens, and importing ppe.capture is the
earliest point every entry point (main.py, camcheck.py) shares.
"""

import importlib
import os

VAR = "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"


def test_importing_capture_disables_msmf_hardware_transforms(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)
    import ppe.capture

    importlib.reload(ppe.capture)
    assert os.environ[VAR] == "0"


def test_an_operators_own_setting_is_never_overridden(monkeypatch):
    """Someone who set this deliberately — including turning it back on —
    must not have that silently undone on the next import."""
    monkeypatch.setenv(VAR, "1")
    import ppe.capture

    importlib.reload(ppe.capture)
    assert os.environ[VAR] == "1"
