"""Writing each machine's relay address into its config at install time."""

import shutil

import pytest

from ppe.config import Config
from ppe.provision import set_relay_host


@pytest.fixture
def cfg_file(tmp_path):
    path = tmp_path / "config.yaml"
    shutil.copyfile("config.yaml", path)
    return path


def test_the_relay_address_is_set_and_the_old_one_returned(cfg_file):
    before = Config.load(cfg_file).tower.host
    assert set_relay_host(cfg_file, "192.168.1.56") == before
    assert Config.load(cfg_file).tower.host == "192.168.1.56"


def test_only_that_one_value_changes_and_every_comment_survives(cfg_file):
    original = cfg_file.read_text(encoding="utf-8").splitlines()
    set_relay_host(cfg_file, "10.0.0.7")
    edited = cfg_file.read_text(encoding="utf-8").splitlines()
    changed = [(a, b) for a, b in zip(original, edited, strict=True) if a != b]
    assert len(changed) == 1
    old, new = changed[0]
    assert "10.0.0.7" in new and new.split("#")[1:] == old.split("#")[1:]


def test_windows_line_endings_are_kept(cfg_file):
    cfg_file.write_bytes(cfg_file.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    set_relay_host(cfg_file, "192.168.1.60")
    raw = cfg_file.read_bytes()
    assert b"\r\r" not in raw and raw.count(b"\n") == raw.count(b"\r\n")
    assert Config.load(cfg_file).tower.host == "192.168.1.60"


@pytest.mark.parametrize("bad", ["192.168.1", "relay.local", "192.168.1.300", "", "1.2.3.4 # x"])
def test_anything_but_a_plain_ipv4_address_is_refused(cfg_file, bad):
    before = cfg_file.read_bytes()
    with pytest.raises(ValueError):
        set_relay_host(cfg_file, bad)
    assert cfg_file.read_bytes() == before


def test_a_host_line_outside_the_tower_block_is_never_touched(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("camera:\n  host: 1.1.1.1\ntower:\n  enabled: true\nui:\n  host: 2.2.2.2\n")
    with pytest.raises(ValueError, match="no 'host:' line"):
        set_relay_host(path, "192.168.1.51")
    assert "1.1.1.1" in path.read_text() and "2.2.2.2" in path.read_text()


def test_the_cli_flag_edits_the_given_config_and_exits(cfg_file):
    import main

    assert main.main(["--set-relay-ip", "192.168.1.77", "-c", str(cfg_file)]) == 0
    assert Config.load(cfg_file).tower.host == "192.168.1.77"
    assert main.main(["--set-relay-ip", "nonsense", "-c", str(cfg_file)]) == 2
