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
