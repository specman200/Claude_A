"""Operator and debug are one window with two faces — never two windows.

A separate debug window would drift from the operator one, and then debug
would no longer be showing you what production does.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

import ppe.pipeline as pipeline_mod  # noqa: E402
from ppe.capture import CameraSet  # noqa: E402
from ppe.config import (  # noqa: E402
    BrandingCfg,
    CameraCfg,
    ClassCfg,
    Config,
    ModelCfg,
    PPECfg,
    TowerCfg,
    UICfg,
)
from ppe.tower import Status  # noqa: E402
from ppe.ui import MainWindow  # noqa: E402

from .conftest import StubDetector  # noqa: E402


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


def build(mode, clip, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    cfg = Config(
        model=ModelCfg(conf=0.3),
        ppe=PPECfg(classes=[ClassCfg("helmet"), ClassCfg("vest")]),
        cameras=[CameraCfg("Line A", clip, 320, 240)],
        tower=TowerCfg(enabled=False),
        ui=UICfg(mode=mode),
        # Real branding, so the logo assertions actually run instead of
        # skipping and quietly proving nothing.
        branding=BrandingCfg(
            name="A. Engineer",
            tagline="Vision & Automation",
            logo=str(Path("assets/logo.svg").resolve()),
        ),
    )
    cameras = CameraSet(cfg.cameras)
    pipe = pipeline_mod.Pipeline(cfg, cameras)
    return MainWindow(cfg, cameras, pipe), pipe, cfg


# -- what each mode shows -------------------------------------------------


def test_operator_hides_the_diagnostics(app, clip, monkeypatch):
    window, _, _ = build("operator", clip, monkeypatch)
    assert not window.debug
    assert window.hud.isHidden()
    assert window.decisions.isHidden()


def test_debug_shows_the_diagnostics(app, clip, monkeypatch):
    window, _, _ = build("debug", clip, monkeypatch)
    assert window.debug
    assert not window.hud.isHidden()
    assert not window.decisions.isHidden()


def test_operator_cannot_retune_the_station_mid_shift(app, clip, monkeypatch):
    """The rows still show everything; they just cannot be changed."""
    window, _, _ = build("operator", clip, monkeypatch)
    assert not window.panel.editable
    row = window.panel.rows["helmet"]
    assert row.conf.isHidden()        # no confidence floor to nudge
    assert row.required.isHidden()    # no requirement to untick
    assert window.panel.picker.isHidden()
    assert set(window.panel.rows) == {"helmet", "vest"}   # but all of it is visible


def test_debug_can_edit_the_checklist(app, clip, monkeypatch):
    window, _, _ = build("debug", clip, monkeypatch)
    assert window.panel.editable
    assert not window.panel.rows["helmet"].conf.isHidden()
    assert not window.panel.picker.isHidden()


def test_both_modes_show_the_same_status(app, clip, monkeypatch):
    """The whole point of one window: debug must not flatter production."""
    for mode in ("operator", "debug"):
        window, _, _ = build(mode, clip, monkeypatch)
        window.banner.apply(Status.VIOLATION, ["Vest"], tower_ok=True)
        assert "PPE MISSING" in window.banner.text()
        assert "Vest" in window.banner.text()


def test_operator_reads_words_debug_reads_numbers(app, clip, monkeypatch):
    from ppe.tower import ClassState

    state = ClassState("helmet", "Hard Hat", True, "present", need=1, count=1, conf=0.87)

    operator, _, _ = build("operator", clip, monkeypatch)
    operator.panel.apply([state])
    assert operator.panel.rows["helmet"].score.text() == "OK"

    debug, _, _ = build("debug", clip, monkeypatch)
    debug.panel.apply([state])
    assert debug.panel.rows["helmet"].score.text() == "0.87"


# -- the annunciator ------------------------------------------------------


def test_debug_mutes_the_annunciator(app, clip, monkeypatch):
    """Tuning a station should not mean listening to it nag."""
    _, pipe, _ = build("debug", clip, monkeypatch)
    assert pipe.annunciator.mute is True


def test_operator_does_not_mute_the_annunciator(app, clip, monkeypatch):
    _, pipe, _ = build("operator", clip, monkeypatch)
    assert pipe.annunciator.mute is False


def test_a_muted_annunciator_still_runs_its_timing():
    """Muting by skipping the evaluation would hide the behaviour you opened
    the debug view to look at."""
    from ppe.annunciator import Annunciator

    a = Annunciator(None, grace_sec=1.0, repeat_sec=2.0, mute=True)
    a.update(Status.VIOLATION, t=0.0)
    assert a.due_in(t=0.0) == pytest.approx(1.0)   # countdown is live
    a.update(Status.VIOLATION, t=1.0)
    assert a.suppressed == 1                       # would have spoken
    assert a.plays == 0                            # but did not


def test_the_debug_badge_says_the_audio_is_muted(app, clip, monkeypatch):
    """An operator walking past a debug screen should not report a dead speaker."""
    from PySide6.QtWidgets import QLabel

    window, _, _ = build("debug", clip, monkeypatch)
    texts = " ".join(w.text() for w in window.findChildren(QLabel))
    assert "DEBUG" in texts and "muted" in texts.lower()


# -- config and CLI -------------------------------------------------------


def test_the_cli_flags_override_the_configured_mode():
    from main import parse_args

    assert parse_args(["--debug"]).debug is True
    assert parse_args(["--operator"]).operator is True


@pytest.mark.parametrize("mode", ["operator", "debug"])
def test_valid_modes_pass_validation(mode):
    cfg = Config.load("config.yaml")
    cfg.ui.mode = mode
    cfg.validate()


def test_an_unknown_mode_is_rejected():
    cfg = Config.load("config.yaml")
    cfg.ui.mode = "kiosk"
    with pytest.raises(ValueError, match="ui.mode must be"):
        cfg.validate()


def test_the_shipped_config_is_operator_mode():
    """A station that boots into debug on the floor is a mistake."""
    assert Config.load("config.yaml").ui.mode == "operator"


# -- the privacy notice ---------------------------------------------------
# People are filmed here all shift. The notice is the thing that says why,
# so it must not quietly shrink back into a footnote.


def notice_of(window):
    from PySide6.QtWidgets import QLabel

    for label in window.findChildren(QLabel):
        if "SURVEILLANCE" in label.text().upper():
            return label
    raise AssertionError("no privacy notice found in the window")


@pytest.mark.parametrize("mode", ["operator", "debug"])
def test_the_privacy_notice_is_shown_in_both_modes(app, clip, monkeypatch, mode):
    window, _, _ = build(mode, clip, monkeypatch)
    text = notice_of(window).text().upper()
    assert "SAFETY" in text
    assert "NOT RECORDED" in text and "NOT SURVEILLANCE" in text


def test_the_privacy_notice_is_prominent_not_a_footnote(app, clip, monkeypatch):
    window, _, _ = build("operator", clip, monkeypatch)
    notice = notice_of(window)
    style = notice.styleSheet()

    # Bigger than the muted 11px caption style used for section headings.
    size = int(style.split("font-size:")[1].split("px")[0])
    assert size >= 15, f"notice font is {size}px — that is footnote treatment"
    assert "font-weight:700" in style
    assert notice.minimumHeight() >= 40
    # Not the muted grey used for de-emphasised text.
    assert notice.objectName() != "h"


def test_the_logo_gets_real_estate(app, clip, monkeypatch):
    """'Allow more space for the logo' — pin it so a later tidy-up cannot
    silently shrink it back."""
    from PySide6.QtWidgets import QLabel

    window, _, cfg = build("operator", clip, monkeypatch)
    marks = [
        w for w in window.brand.findChildren(QLabel)
        if w.pixmap() is not None and not w.pixmap().isNull()
    ]
    if not marks:
        pytest.skip("no logo configured in this fixture")
    assert marks[0].width() >= 64
