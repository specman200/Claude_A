"""The camera diagnostic tool: it must tell the truth about what it finds."""

import cv2
import numpy as np

from ppe.camcheck import probe, scan_dev_video
from ppe.config import CameraCfg


def clip(tmp_path, name, frame, n=6, fps=30):
    path = tmp_path / name
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, frame.shape[1::-1])
    for _ in range(n):
        writer.write(frame)
    writer.release()
    return str(path)


def test_a_real_source_probes_true(tmp_path, capsys):
    frame = np.full((240, 320, 3), 200, np.uint8)
    frame[50:150, 50:150] = 10  # not flat, so it reads as real content
    path = clip(tmp_path, "ok.avi", frame)
    assert probe(CameraCfg("Cam", path, 320, 240, 30), timeout=3.0, save=None, tag="t") is True
    assert "looks real" in capsys.readouterr().out


def test_a_missing_source_probes_false(capsys):
    cfg = CameraCfg("Cam", 9999, 640, 480, 30, api="v4l2")
    assert probe(cfg, timeout=0.5, save=None, tag="t") is False
    assert "FAILED to open" in capsys.readouterr().out


def test_a_flat_frame_is_flagged_not_silently_passed(tmp_path, capsys):
    """A lens cap or a dead sensor still opens and 'works' — the flatness is the tell."""
    path = clip(tmp_path, "blank.avi", np.zeros((240, 320, 3), np.uint8))
    assert probe(CameraCfg("Cam", path, 320, 240, 30), timeout=3.0, save=None, tag="t") is False
    assert "ALL ONE COLOUR" in capsys.readouterr().out


def test_a_source_that_opens_but_never_delivers_a_frame_times_out(tmp_path, capsys):
    """Opens fine but read() never succeeds — a stalled driver, not a missing device."""
    path = tmp_path / "empty.avi"
    cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240)).release()
    cfg = CameraCfg("Cam", str(path), 320, 240, 30)
    assert probe(cfg, timeout=0.3, save=None, tag="t") is False
    assert "no frame arrived" in capsys.readouterr().out


def test_save_writes_one_file_per_camera_tag(tmp_path):
    frame = np.full((240, 320, 3), 200, np.uint8)
    frame[0:10, 0:10] = 5
    path = clip(tmp_path, "ok.avi", frame)
    out = tmp_path / "probe.jpg"
    probe(CameraCfg("Cam", path, 320, 240, 30), timeout=3.0, save=str(out), tag="cam0")
    assert (tmp_path / "probe_cam0.jpg").exists()


def test_scan_never_raises_even_with_no_video_devices():
    # Whatever this host has, the call must return a list, not throw.
    assert isinstance(scan_dev_video(), list)


def test_main_exits_nonzero_when_a_camera_fails(tmp_path, monkeypatch):
    import yaml

    from ppe.camcheck import main

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"cameras": [{"name": "Dead", "source": 9999, "api": "v4l2"}]})
    )
    monkeypatch.chdir(tmp_path)
    assert main(["-c", str(cfg_path), "--timeout", "0.2"]) == 1


def test_main_exits_zero_when_every_camera_works(tmp_path, monkeypatch):
    import yaml

    from ppe.camcheck import main

    frame = np.full((240, 320, 3), 200, np.uint8)
    frame[0:20, 0:20] = 5
    path = clip(tmp_path, "ok.avi", frame)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"cameras": [{"name": "Cam", "source": path, "width": 320, "height": 240}]})
    )
    monkeypatch.chdir(tmp_path)
    assert main(["-c", str(cfg_path), "--timeout", "3.0"]) == 0


def test_main_reports_a_useful_error_with_no_cameras_configured(tmp_path, monkeypatch):
    import yaml

    from ppe.camcheck import main

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"cameras": []}))
    monkeypatch.chdir(tmp_path)
    assert main(["-c", str(cfg_path)]) == 1


# -- platform dispatch -------------------------------------------------
# This suite runs on Linux, so the Windows branch is exercised by forcing
# ON_WINDOWS rather than by an actual Windows machine — the real backend
# calls (cv2.CAP_DSHOW / CAP_MSMF) are still made, just against no hardware.


def test_windows_index_scan_never_raises_with_no_cameras_present(monkeypatch):
    import ppe.camcheck as camcheck

    # No real camera responds in this environment; the call must still
    # return cleanly rather than raising or hanging.
    assert camcheck.scan_windows_indices(max_index=2) == []


def test_causes_text_content_is_platform_specific():
    from ppe.camcheck import CAUSES_LINUX, CAUSES_WINDOWS

    assert "camera privacy" in CAUSES_WINDOWS.lower()
    assert "/dev/video" not in CAUSES_WINDOWS

    assert "/dev/video" in CAUSES_LINUX
    assert "privacy" not in CAUSES_LINUX.lower()


def test_causes_reads_on_windows_live_not_at_import_time(monkeypatch):
    """A frozen CAUSES computed at import would keep printing Linux advice
    forever once ON_WINDOWS is set — causes() must read it fresh."""
    import ppe.camcheck as camcheck

    monkeypatch.setattr(camcheck, "ON_WINDOWS", True)
    assert camcheck.causes() is camcheck.CAUSES_WINDOWS

    monkeypatch.setattr(camcheck, "ON_WINDOWS", False)
    assert camcheck.causes() is camcheck.CAUSES_LINUX


def test_main_scan_dispatches_to_windows_probing_when_forced(monkeypatch, tmp_path, capsys):
    import yaml

    import ppe.camcheck as camcheck

    monkeypatch.setattr(camcheck, "ON_WINDOWS", True)
    monkeypatch.setattr(camcheck, "scan_windows_indices", lambda max_index=6: [(0, "msmf")])

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"cameras": [{"name": "Dead", "source": 9999, "api": "msmf"}]})
    )
    monkeypatch.chdir(tmp_path)
    camcheck.main(["-c", str(cfg_path), "--scan", "--timeout", "0.2"])

    out = capsys.readouterr().out
    assert "index 0: opens via msmf" in out
    assert "probing indices" in out


def test_main_scan_uses_dev_video_glob_when_not_windows(monkeypatch, tmp_path, capsys):
    import yaml

    import ppe.camcheck as camcheck

    monkeypatch.setattr(camcheck, "ON_WINDOWS", False)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"cameras": [{"name": "Dead", "source": 9999, "api": "v4l2"}]})
    )
    monkeypatch.chdir(tmp_path)
    camcheck.main(["-c", str(cfg_path), "--scan", "--timeout", "0.2"])

    out = capsys.readouterr().out
    assert "/dev/video" in out
    assert "probing indices" not in out


def test_msmf_and_dshow_are_valid_backend_names():
    from ppe.capture import _APIS

    assert "msmf" in _APIS and "dshow" in _APIS


# -- mode probing ------------------------------------------------------
# OpenCV reports what you got, never what was on offer. --modes asks for each
# size and format in turn and measures the result, so "is this camera slow
# because it has no compressed mode?" becomes answerable.


class FakeCap:
    """A camera that delivers one fixed mode whatever it is asked for."""

    def __init__(self, width=640, height=480, fps=10.0, opens=True, claims="YUY2"):
        self.width, self.height, self.fps = width, height, fps
        self.opens, self.claims = opens, claims
        self.asked = []
        self.released = False

    def isOpened(self):  # noqa: N802 — cv2 naming
        return self.opens

    def set(self, prop, value):
        self.asked.append((prop, value))
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FOURCC:
            return float(int.from_bytes(self.claims.encode(), "little"))
        return 0.0

    def read(self):
        import time as _t

        _t.sleep(1.0 / self.fps)
        return True, np.zeros((self.height, self.width, 3), np.uint8)

    def release(self):
        self.released = True


def test_mode_probing_reports_the_frame_it_got_not_the_size_it_asked_for(monkeypatch):
    """A driver will happily report a mode it is not delivering — that is the
    whole reason this exists — so the size must come from a decoded frame."""
    import ppe.camcheck as camcheck

    cap = FakeCap(width=640, height=480, fps=60.0)
    monkeypatch.setattr(camcheck.cv2, "VideoCapture", lambda *a, **k: cap)
    got = camcheck.measure_mode(0, cv2.CAP_ANY, "MJPG", 1920, 1080, 30, warmup=1, timed=3)
    assert got["width"] == 640 and got["height"] == 480, "reported the request, not the frame"
    assert got["fps"] > 0
    assert cap.released, "every probe must release its handle"


def test_a_mode_that_will_not_open_is_a_result_not_a_crash(monkeypatch):
    import ppe.camcheck as camcheck

    monkeypatch.setattr(camcheck.cv2, "VideoCapture", lambda *a, **k: FakeCap(opens=False))
    assert camcheck.measure_mode(0, cv2.CAP_ANY, "MJPG", 640, 480, 30) is None


def test_mode_probing_skips_files_rather_than_printing_meaningless_numbers(tmp_path, capsys):
    """A file ignores every mode request and decodes as fast as the CPU allows,
    which would print a confident table that means nothing."""
    from ppe.camcheck import probe_modes

    path = clip(tmp_path, "vid.avi", np.full((240, 320, 3), 128, np.uint8))
    probe_modes(CameraCfg("Cam", path, 320, 240, 30))
    out = capsys.readouterr().out
    assert "not a camera" in out
    assert "fps" not in out, "no measurements should be printed for a file"


def test_mode_probing_measures_every_size_and_format(monkeypatch, capsys):
    import ppe.camcheck as camcheck

    monkeypatch.setattr(camcheck.cv2, "VideoCapture", lambda *a, **k: FakeCap(fps=60.0))
    monkeypatch.setattr(camcheck, "PROBE_SIZES", ((640, 480), (1280, 720)))
    camcheck.probe_modes(CameraCfg("Cam", 0, 1280, 720, 30))
    out = capsys.readouterr().out
    assert out.count("fps") >= 4, out          # 2 sizes x 2 formats
    assert "MJPG" in out and "YUY2" in out
    assert "currently configured" in out
