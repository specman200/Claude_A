"""Finding files in a source checkout and in an installed build, and the
one-station-per-PC lock."""

import os
import sys

import pytest

from ppe import paths
from ppe.config import BrandingCfg
from ppe.paths import InstanceLock, resolve


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """Pretend to be a PyInstaller build: program files in one folder,
    ProgramData in another."""
    bundle = tmp_path / "Program Files" / "PPE Station" / "_internal"
    (bundle / "models" / "m_openvino_model").mkdir(parents=True)
    (bundle / "assets").mkdir()
    (bundle / "assets" / "logo.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    programdata = tmp_path / "ProgramData"
    programdata.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("PROGRAMDATA", str(programdata))
    monkeypatch.delenv("PPE_STATION_DATA", raising=False)
    monkeypatch.chdir(programdata)
    return bundle, programdata


def test_from_source_the_program_dir_is_the_checkout_and_data_is_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not paths.frozen()
    assert (paths.app_dir() / "ppe" / "paths.py").is_file()
    assert paths.data_dir() == tmp_path


def test_installed_the_two_folders_split(installed):
    bundle, programdata = installed
    assert paths.frozen()
    assert paths.app_dir() == bundle
    assert paths.data_dir() == programdata / "PPE Station"


def test_the_data_folder_can_be_redirected(installed, tmp_path, monkeypatch):
    monkeypatch.setenv("PPE_STATION_DATA", str(tmp_path / "scratch"))
    assert paths.data_dir() == tmp_path / "scratch"


def test_a_path_that_exists_as_given_is_returned_untouched(tmp_path, monkeypatch):
    """The behaviour from before resolve() existed, byte for byte — including
    the trailing slash ultralytics uses to spot an OpenVINO directory."""
    (tmp_path / "models" / "x_openvino_model").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert resolve("models/x_openvino_model/", tmp_path / "elsewhere") == "models/x_openvino_model/"


def test_installed_the_model_is_found_in_the_program_folder(installed):
    bundle, programdata = installed
    got = resolve("models/m_openvino_model/", programdata / "PPE Station")
    assert got == str(bundle / "models" / "m_openvino_model") + os.sep
    assert got.endswith(os.sep), "the trailing separator marks an export directory"


def test_a_file_beside_the_config_wins_over_the_shipped_one(installed, tmp_path):
    bundle, _ = installed
    local = tmp_path / "cfgdir"
    (local / "models").mkdir(parents=True)
    (local / "models" / "custom.pt").write_bytes(b"x")
    (bundle / "models" / "custom.pt").write_bytes(b"y")
    assert resolve("models/custom.pt", local) == str(local / "models" / "custom.pt")


def test_a_bare_model_name_found_nowhere_reaches_ultralytics_as_written(installed):
    assert resolve("yolo11s.pt", None) == "yolo11s.pt"
    assert resolve("", None) == ""


def test_installed_the_logo_falls_back_to_the_shipped_asset(installed):
    bundle, programdata = installed
    got = BrandingCfg(logo="assets/logo.svg").logo_path(programdata / "PPE Station")
    assert got == bundle / "assets" / "logo.svg"


def test_the_fallback_still_refuses_a_directory_as_a_logo(installed):
    _, programdata = installed
    assert BrandingCfg(logo="assets").logo_path(programdata / "PPE Station") is None


# -- one station per PC ------------------------------------------------------


def test_a_second_station_cannot_take_the_lock(tmp_path):
    first, second = InstanceLock(tmp_path / "s.lock"), InstanceLock(tmp_path / "s.lock")
    assert first.acquire()
    try:
        assert not second.acquire()
    finally:
        first.release()


def test_the_lock_is_free_again_once_released(tmp_path):
    first, second = InstanceLock(tmp_path / "s.lock"), InstanceLock(tmp_path / "s.lock")
    assert first.acquire()
    first.release()
    assert second.acquire()
    second.release()


def test_the_lock_creates_its_folder(tmp_path):
    lock = InstanceLock(tmp_path / "logs" / "deep" / "s.lock")
    assert lock.acquire()
    lock.release()
