"""The supervisor restarts a crashed station and leaves a closed one closed."""

import sys
from pathlib import Path

import pytest

from ppe import supervise as sup


class FakeProc:
    def __init__(self, code):
        self.code = code
        self.terminated = False

    def wait(self):
        if isinstance(self.code, BaseException):
            raise self.code
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = 0


class Harness:
    """Feeds exit codes and uptimes to supervise() and records what it did."""

    def __init__(self, runs):
        self.runs = list(runs)      # (exit code, seconds it ran)
        self.t = 0.0
        self.started = []
        self.sleeps = []
        self.procs = []

    def popen(self, cmd):
        self.started.append(cmd)
        code, ran = self.runs.pop(0)
        self.t += ran
        proc = FakeProc(code)
        self.procs.append(proc)
        return proc

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s

    def clock(self):
        return self.t

    def run(self, argv=("--supervise",)):
        return sup.supervise(list(argv), popen=self.popen, sleep=self.sleep, clock=self.clock)


def test_a_clean_exit_is_respected_and_not_restarted():
    h = Harness([(0, 50.0)])
    assert h.run() == 0
    assert len(h.started) == 1 and h.sleeps == []


def test_a_crash_is_restarted_until_the_station_closes_cleanly():
    h = Harness([(1, 5.0), (3221225477, 5.0), (0, 10.0)])   # exception, access violation, closed
    assert h.run() == 0
    assert len(h.started) == 3


def test_back_off_doubles_to_a_ceiling_while_the_station_keeps_failing():
    h = Harness([(1, 1.0)] * 8 + [(0, 1.0)])
    h.run()
    assert h.sleeps == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0]


def test_a_long_healthy_run_resets_the_back_off():
    h = Harness([(1, 1.0), (1, 1.0), (1, 1.0), (1, 600.0), (0, 1.0)])
    h.run()
    assert h.sleeps == [2.0, 4.0, 8.0, 2.0]


def test_the_child_gets_the_same_arguments_without_supervise():
    h = Harness([(0, 1.0)])
    h.run(["--supervise", "--debug", "-c", "x.yaml"])
    cmd = h.started[0]
    assert "--supervise" not in cmd
    assert cmd[-3:] == ["--debug", "-c", "x.yaml"]
    assert cmd[0] == sys.executable and Path(cmd[1]).name == "main.py"


def test_installed_the_child_is_the_exe_itself(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert sup.child_command(["--supervise", "--debug"]) == [sys.executable, "--debug"]


def test_ctrl_c_stops_the_child_and_the_supervisor():
    h = Harness([(KeyboardInterrupt(), 1.0)])
    assert h.run() == 0
    assert h.procs[0].terminated


@pytest.mark.parametrize("code", [0])
def test_the_main_entry_point_dispatches_to_the_supervisor(monkeypatch, code):
    import main

    seen = {}

    def fake(argv):
        seen["argv"] = list(argv)
        return code

    monkeypatch.setattr(sup, "supervise", fake)
    assert main.main(["--supervise", "--debug"]) == code
    assert seen["argv"] == ["--supervise", "--debug"]


def test_camcheck_runs_as_a_subcommand_against_the_default_config(monkeypatch):
    import main
    import ppe.camcheck

    seen = {}
    monkeypatch.setattr(ppe.camcheck, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    assert main.main(["camcheck", "--scan"]) == 0
    assert seen["argv"] == ["-c", "config.yaml", "--scan"]


def test_camcheck_keeps_an_explicit_config(monkeypatch):
    import main
    import ppe.camcheck

    seen = {}
    monkeypatch.setattr(ppe.camcheck, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    main.main(["camcheck", "-c", "other.yaml"])
    assert seen["argv"] == ["-c", "other.yaml"]
