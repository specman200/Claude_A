"""UI smoke tests — the window builds, paints, and reflects live state.

Run headless via Qt's offscreen platform, so they work in CI.
"""

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from ppe.capture import Frame  # noqa: E402
from ppe.config import ClassCfg, Config, ModelCfg, PPECfg  # noqa: E402
from ppe.detector import Detection  # noqa: E402
from ppe.tower import ClassState, Status  # noqa: E402
from ppe.ui import (  # noqa: E402
    ABSENT,
    BANNER,
    IDLE,
    PRESENT,
    UNAVAILABLE,
    MainWindow,
    PPEPanel,
    StatusBanner,
    StatusCard,
    VideoPane,
)

from .conftest import StubDetector  # noqa: E402


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


def paint(widget, w=800, h=600):
    """Render a widget offscreen; raises if paintEvent throws."""
    widget.resize(w, h)
    image = QImage(w, h, QImage.Format_RGB32)
    widget.render(image)
    return image


def frame(w=1280, h=720, seq=1):
    return Frame(0, seq, np.full((h, w, 3), 90, np.uint8), 0.0)


def test_pane_paints_before_any_video_arrives(app):
    paint(VideoPane("Line A"))


def test_pane_paints_frames_and_boxes(app):
    pane = VideoPane("Line A")
    pane.show_frame(frame(), 30.0, True)
    pane.show_detections(
        [Detection("helmet", 0.91, (100.0, 50.0, 300.0, 260.0))], [], None, {"helmet": PRESENT}
    )
    paint(pane)
    assert pane.seq == 1


@pytest.mark.parametrize("size", [(1920, 1080), (640, 480), (480, 640)])
@pytest.mark.parametrize("widget", [(800, 600), (300, 700), (1000, 200)])
def test_boxes_track_the_image_through_every_aspect_combination(app, size, widget):
    """A box on the image's centre must land on the widget's drawn centre."""
    from ppe.letterbox import fit

    pane = VideoPane("cam")
    pane.show_frame(frame(*size), 30.0, True)
    pane.resize(*widget)

    scale, ox, oy = fit(size[0], size[1], *widget)
    # The drawn image must sit inside the widget, centred, aspect intact.
    assert ox >= -1e-6 and oy >= -1e-6
    assert size[0] * scale <= widget[0] + 1e-6
    assert size[1] * scale <= widget[1] + 1e-6
    assert ox == pytest.approx((widget[0] - size[0] * scale) / 2)
    assert oy == pytest.approx((widget[1] - size[1] * scale) / 2)

    cx = ox + (size[0] / 2) * scale
    assert cx == pytest.approx(widget[0] / 2)
    paint(pane, *widget)


def config():
    return Config(
        model=ModelCfg(conf=0.3),
        ppe=PPECfg(
            classes=[ClassCfg("helmet"), ClassCfg("vest"), ClassCfg("mask", required=False)]
        ),
    )


def dot_color(row):
    return row.dot.styleSheet()


def test_panel_lists_every_configured_class(app):
    panel = PPEPanel(config(), ["helmet", "vest", "mask", "gloves"])
    assert list(panel.rows) == ["helmet", "vest", "mask"]
    assert panel.rows["helmet"].required.isChecked()
    assert not panel.rows["mask"].required.isChecked()


def test_row_colour_follows_detection(app):
    panel = PPEPanel(config(), ["helmet", "vest", "mask"])
    panel.apply(
        [
            ClassState("helmet", "Hard Hat", True, count=1, conf=0.87),
            ClassState("vest", "Vest", True, count=0),
            ClassState("mask", "Mask", False, count=0),
        ]
    )
    assert PRESENT in dot_color(panel.rows["helmet"])   # detected -> green
    assert ABSENT in dot_color(panel.rows["vest"])      # required, missing -> red
    assert IDLE in dot_color(panel.rows["mask"])        # optional, missing -> grey
    assert panel.rows["helmet"].score.text() == "0.87"


def test_a_class_the_model_lacks_is_flagged_amber(app):
    panel = PPEPanel(config(), ["helmet"])
    panel.apply([ClassState("vest", "Vest", True, available=False)])
    assert UNAVAILABLE in dot_color(panel.rows["vest"])
    assert panel.rows["vest"].score.text() == "n/a"


def test_editing_a_row_updates_the_config_and_signals(app):
    cfg = config()
    panel = PPEPanel(cfg, ["helmet", "vest", "mask"])
    edits = []
    panel.edited.connect(lambda: edits.append(1))

    panel.rows["mask"].required.setChecked(True)
    panel.rows["helmet"].conf.setValue(0.55)
    assert cfg.ppe.classes[2].required is True
    assert cfg.ppe.classes[0].conf == 0.55
    assert len(edits) == 2


def test_adding_and_removing_classes_rewrites_the_list(app):
    cfg = config()
    panel = PPEPanel(cfg, ["helmet", "vest", "mask", "gloves"])
    panel.picker.setCurrentText("gloves")
    panel._add_from_picker()
    assert [c.name for c in cfg.ppe.classes][-1] == "gloves"
    assert "gloves" in panel.rows

    panel._remove("helmet")
    assert "helmet" not in panel.rows
    assert [c.name for c in cfg.ppe.classes] == ["vest", "mask", "gloves"]


def test_duplicate_and_blank_additions_are_ignored(app):
    cfg = config()
    panel = PPEPanel(cfg, ["helmet"])
    for text in ("helmet", "", "   "):
        panel.picker.setCurrentText(text)
        panel._add_from_picker()
    assert len(cfg.ppe.classes) == 3


def test_save_writes_the_edited_config(app, tmp_path):
    cfg = config()
    cfg.path = tmp_path / "config.yaml"
    panel = PPEPanel(cfg, ["helmet", "vest", "mask"])
    panel.rows["mask"].required.setChecked(True)
    panel._save()

    reloaded = Config.load(cfg.path)
    assert [c.name for c in reloaded.ppe.required] == ["helmet", "vest", "mask"]


def test_each_class_keeps_a_distinct_box_colour(app):
    panel = PPEPanel(config(), ["helmet", "vest", "mask"])
    colors = panel.colors()
    assert set(colors) == {"helmet", "vest", "mask"}
    assert len(set(colors.values())) == 3


@pytest.mark.parametrize("status", list(Status))
def test_banner_renders_every_status(app, status):
    banner = StatusBanner()
    banner.apply(status, ["Vest"], tower_ok=True)
    assert banner.text()
    paint(banner, 360, 64)


# The real MainWindow pins the side column to these widths — operator mode
# at 400px (StatusCard(compact=False)), debug mode at 380px (compact=True).
STATUS_CARD_WIDTHS = [(False, 400), (True, 380)]


@pytest.mark.parametrize("compact,width", STATUS_CARD_WIDTHS)
@pytest.mark.parametrize("status", list(Status))
def test_status_card_renders_every_status_at_production_width(app, status, compact, width):
    card = StatusCard(compact=compact)
    card.apply(status, ["Vest"], tower_ok=True)
    paint(card, width, card.minimumHeight())


@pytest.mark.parametrize("compact,width", STATUS_CARD_WIDTHS)
def test_the_standby_headline_needs_wrapping_at_production_width(app, compact, width):
    """STANDBY's headline is the longest text BANNER carries. Pin that it
    genuinely overflows a single line at the real column width — if a
    future edit shrinks the font or widens the column enough that this
    stops being true, the wrap-vs-clip code path this exercises would go
    untested."""
    from PySide6.QtGui import QFont, QFontMetrics

    text = BANNER[Status.STANDBY][0]
    font = QFont("Segoe UI", 13 if compact else 19, QFont.Bold)
    # Same left margin the paint code reserves before the headline column.
    margin = (16 + 40 + 14) if compact else 24
    available = width - 2 - margin
    assert QFontMetrics(font).horizontalAdvance(text) > available


# -- the thumbs-up / thumbs-down glyphs -------------------------------------


def test_ok_and_violation_ship_a_real_icon_not_the_vector_fallback(app):
    """Pin that the shipped assets actually resolve — if a rename or a
    corrupt SVG ever broke this, the silent fallback to the drawn tick/cross
    would hide it; a passing paint() alone would not catch that."""
    from ppe.ui import _glyph_icon

    assert _glyph_icon("thumbs_up", 40, 1.0) is not None
    assert _glyph_icon("thumbs_down", 40, 1.0) is not None


@pytest.mark.parametrize("compact,width", STATUS_CARD_WIDTHS)
@pytest.mark.parametrize("status", [Status.OK, Status.VIOLATION])
def test_ok_and_violation_paint_with_the_icon(app, status, compact, width):
    card = StatusCard(compact=compact)
    card.apply(status, ["Vest"], tower_ok=True)
    paint(card, width, card.minimumHeight())


def test_a_missing_icon_falls_back_to_the_drawn_glyph_rather_than_going_blank(
    app, monkeypatch
):
    """The verdict glyph is safety-critical: a bad or missing icon file must
    never leave the card with nothing drawn where the tick/cross used to be."""
    import ppe.ui as ui_mod

    monkeypatch.setattr(ui_mod, "_glyph_icon", lambda *a, **k: None)
    card = StatusCard(compact=False)
    card.apply(Status.OK, [], tower_ok=True)
    paint(card, 400, card.minimumHeight())  # must not raise


@pytest.mark.parametrize("status", [Status.STANDBY, Status.DEGRADED])
def test_standby_and_degraded_are_unaffected_by_the_icon_change(app, status):
    """Only OK and VIOLATION got icons; the dash and the bang must still be
    the drawn vector glyphs, not an icon lookup that silently no-ops."""
    from ppe.ui import StatusCard

    assert status not in StatusCard._ICON_NAMES


def test_banner_names_what_is_missing_and_flags_a_dead_bus(app):
    banner = StatusBanner()
    banner.apply(Status.VIOLATION, ["Vest", "Gloves"], tower_ok=False)
    assert "Vest, Gloves" in banner.text()
    assert "tower offline" in banner.text()


def test_degraded_banner_distinguishes_no_video_from_an_unusable_model(app):
    """Both faults show amber; the operator still needs to know which it is."""
    banner = StatusBanner()

    banner.apply(Status.DEGRADED, [], tower_ok=True)
    assert "NO VIDEO SIGNAL" in banner.text()

    banner.apply(Status.DEGRADED, [], tower_ok=True, unavailable=["Hard Hat"])
    assert "Hard Hat" in banner.text()
    assert "NO VIDEO" not in banner.text()


# -- whole window ----------------------------------------------------------


@pytest.fixture
def station(clip, tmp_path, monkeypatch, app):
    """A live window over synthetic cameras and a stubbed model."""
    import ppe.pipeline as pipeline_mod
    from ppe.capture import CameraSet
    from ppe.config import BrandingCfg, CameraCfg, TelemetryCfg, TowerCfg
    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    cfg = config()
    cfg.path = tmp_path / "config.yaml"
    cfg.cameras = [CameraCfg("Line A", clip, 320, 240), CameraCfg("Line B", clip, 320, 240)]
    cfg.tower = TowerCfg(enabled=False)
    cfg.telemetry = TelemetryCfg(csv=str(tmp_path / "latency.csv"))
    cfg.branding = BrandingCfg(
        name="A. Engineer",
        tagline="Vision & Automation",
        logo=str(Path("assets/logo.svg").resolve()),
    )

    cameras = CameraSet(cfg.cameras).start()
    assert cameras.wait_ready(timeout=10.0)
    pipe = pipeline_mod.Pipeline(cfg, cameras)
    window = MainWindow(cfg, cameras, pipe)
    pipe.start()
    try:
        yield window, pipe
    finally:
        window.close()


def pump(app, until, timeout=10.0):
    """Spin the Qt event loop until ``until()`` holds."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if until():
            return True
        time.sleep(0.01)
    return False


def test_window_shows_a_pane_per_camera(app, station):
    window, _ = station
    assert len(window.panes) == 2
    assert [p.name for p in window.panes] == ["Line A", "Line B"]


def test_live_frames_and_detections_reach_the_screen(app, station):
    window, pipe = station
    assert pump(app, lambda: all(p.seq > 0 for p in window.panes)), "no frames drawn"
    assert pump(app, lambda: any(p._dets for p in window.panes)), "no boxes drawn"
    assert pump(app, lambda: PRESENT in window.panel.rows["helmet"].dot.styleSheet())
    # Rows track detections immediately, but the banner follows the debounced
    # status, so it can lag the rows by a few cycles — wait for it rather than
    # assuming they update together.
    assert pump(app, lambda: "PPE MISSING" in window.banner.text()), window.banner.text()
    paint(window, 1200, 800)


def test_the_hud_fills_in_from_real_measurements(app, station):
    window, pipe = station
    assert pump(app, lambda: pipe.metrics.stats("end_to_end")["n"] > 0)
    window._draw_stats()
    assert window.hud.cells["inference"][0].text() == "8.0"
    assert float(window.hud.cells["end_to_end"][0].text()) >= 8.0
    assert "inferences/s" in window.hud.rate.text()


def test_render_time_is_measured_too(app, station):
    window, pipe = station
    assert pump(app, lambda: pipe.metrics.stats("render")["n"] > 0), "render not profiled"


def test_editing_the_checklist_takes_effect_live(app, station):
    window, pipe = station
    assert pump(app, lambda: window.banner.text().startswith("PPE MISSING"))
    window.panel._remove("vest")  # the user drops the vest requirement
    assert pump(app, lambda: "ALL PPE PRESENT" in window.banner.text())


# -- forbidden classes in the UI -------------------------------------------


def forbidden_config():
    from ppe.config import ClassCfg, Config, ModelCfg, PPECfg

    return Config(
        model=ModelCfg(conf=0.3),
        ppe=PPECfg(
            classes=[
                ClassCfg("sleeves", "Sleeves"),
                ClassCfg("Wrong Sleeve", "Wrong Sleeve", expect="absent"),
            ]
        ),
    )


def test_a_forbidden_row_is_marked_so_it_cannot_be_misread(app):
    panel = PPEPanel(forbidden_config(), ["sleeves", "Wrong Sleeve"])
    assert "⊘" in panel.rows["Wrong Sleeve"].label.text()
    assert "⊘" not in panel.rows["sleeves"].label.text()
    assert "must NOT appear" in panel.rows["Wrong Sleeve"].label.toolTip()


def test_green_means_compliant_even_when_that_means_absent(app):
    """A forbidden class reads green while missing and red once detected."""
    panel = PPEPanel(forbidden_config(), ["sleeves", "Wrong Sleeve"])
    row = panel.rows["Wrong Sleeve"]

    panel.apply([ClassState("Wrong Sleeve", "Wrong Sleeve", True, "absent", count=0)])
    assert PRESENT in row.dot.styleSheet()

    panel.apply(
        [ClassState("Wrong Sleeve", "Wrong Sleeve", True, "absent", count=1, conf=0.71)]
    )
    assert ABSENT in row.dot.styleSheet()
    assert row.score.text() == "0.71"


def test_a_normal_row_is_unaffected_by_the_inverted_rule(app):
    panel = PPEPanel(forbidden_config(), ["sleeves", "Wrong Sleeve"])
    row = panel.rows["sleeves"]
    panel.apply([ClassState("sleeves", "Sleeves", True, "present", count=1, conf=0.9)])
    assert PRESENT in row.dot.styleSheet()
    panel.apply([ClassState("sleeves", "Sleeves", True, "present", count=0)])
    assert ABSENT in row.dot.styleSheet()


def test_the_banner_names_each_kind_of_fault_in_its_own_terms(app):
    banner = StatusBanner()

    banner.apply(Status.VIOLATION, ["Gloves"], tower_ok=True, banned=["Wrong Sleeve"])
    assert "MISSING: Gloves" in banner.text()
    assert "NOT ALLOWED: Wrong Sleeve" in banner.text()

    banner.apply(Status.VIOLATION, [], tower_ok=True, banned=["Wrong Sleeve"])
    assert "MISSING" not in banner.text()
    assert "NOT ALLOWED: Wrong Sleeve" in banner.text()

    banner.apply(Status.VIOLATION, ["Gloves"], tower_ok=True)
    assert "MISSING: Gloves" in banner.text()
    assert "NOT ALLOWED" not in banner.text()


# -- subject gating in the UI ----------------------------------------------


def test_the_pane_draws_ignored_detections_and_rings_the_subject(app):
    """Off-subject boxes stay visible, faintly, so nothing looks undetected."""
    pane = VideoPane("Line A")
    pane.show_frame(frame(), 30.0, True)
    subject = Detection("person", 0.9, (100.0, 100.0, 500.0, 900.0))
    pane.show_detections(
        [subject, Detection("Gloves", 0.8, (200.0, 400.0, 260.0, 460.0))],
        [Detection("person", 0.5, (600.0, 300.0, 700.0, 500.0))],
        subject,
        {"person": PRESENT, "Gloves": PRESENT},
    )
    paint(pane, 800, 600)
    assert pane._subject is subject and len(pane._ignored) == 1


def test_the_pane_survives_having_no_subject(app):
    pane = VideoPane("Line A")
    pane.show_frame(frame(), 30.0, True)
    pane.show_detections([], [Detection("Gloves", 0.8, (10.0, 10.0, 50.0, 50.0))], None, {})
    paint(pane, 800, 600)


@pytest.mark.parametrize("status", [Status.STANDBY, Status.DEGRADED])
def test_rows_go_neutral_when_the_station_is_not_judging(app, status):
    """Red rows under a STANDBY banner would read as violations nobody caused."""
    panel = PPEPanel(config(), ["helmet", "vest", "mask"])
    states = [
        ClassState("helmet", "Hard Hat", True, "present", count=0),
        ClassState("vest", "Vest", True, "present", count=0),
    ]
    panel.apply(states, judging=False)
    for name in ("helmet", "vest"):
        assert IDLE in panel.rows[name].dot.styleSheet()
        assert panel.rows[name].score.text() == "--"

    panel.apply(states, judging=True)
    assert ABSENT in panel.rows["helmet"].dot.styleSheet()


def test_the_standby_banner_says_why_it_is_idle(app):
    banner = StatusBanner()
    banner.apply(Status.STANDBY, [], tower_ok=True)
    assert "STANDBY" in banner.text() and "NO PERSON" in banner.text()
    assert "MISSING" not in banner.text()
    paint(banner, 360, 64)


def test_changing_a_class_confidence_reaches_the_detector_live(app, station):
    """The spinbox is not cosmetic: the edit is pushed into the running model."""
    window, pipe = station
    assert pump(app, lambda: pipe.cycles > 0)

    window.panel.rows["helmet"].conf.setValue(0.75)

    assert window.cfg.ppe.classes[0].conf == 0.75
    assert pipe.detector.floors["helmet"] == 0.75
    assert pipe.detector.floors["vest"] is None  # untouched classes keep the default


def test_each_class_carries_its_own_confidence(app, tmp_path):
    from ppe.config import Config

    cfg = config()
    cfg.path = tmp_path / "config.yaml"
    panel = PPEPanel(cfg, ["helmet", "vest", "mask"])
    panel.rows["helmet"].conf.setValue(0.70)
    panel.rows["mask"].conf.setValue(0.20)
    panel._save()

    saved = {c.name: c.conf for c in Config.load(cfg.path).ppe.classes}
    assert saved == {"helmet": 0.70, "vest": None, "mask": 0.20}


def test_a_count_rule_shows_the_tally_not_the_confidence(app):
    """'1/2' is the fault; a confidence score would hide it."""
    from ppe.config import ClassCfg, Config, ModelCfg, PPECfg

    cfg = Config(
        model=ModelCfg(conf=0.3),
        ppe=PPECfg(classes=[ClassCfg("Gloves", "Gloves", count=2), ClassCfg("Mask", "Mask")]),
    )
    panel = PPEPanel(cfg, ["Gloves", "Mask"])
    assert "×2" in panel.rows["Gloves"].label.text()
    assert "×" not in panel.rows["Mask"].label.text()

    panel.apply([
        ClassState("Gloves", "Gloves", True, "present", need=2, count=1, conf=0.8),
        ClassState("Mask", "Mask", True, "present", need=1, count=1, conf=0.8),
    ])
    assert panel.rows["Gloves"].score.text() == "1/2"
    assert ABSENT in panel.rows["Gloves"].dot.styleSheet()   # short by one
    assert panel.rows["Mask"].score.text() == "0.80"
    assert PRESENT in panel.rows["Mask"].dot.styleSheet()

    panel.apply([ClassState("Gloves", "Gloves", True, "present", need=2, count=2, conf=0.8)])
    assert panel.rows["Gloves"].score.text() == "2/2"
    assert PRESENT in panel.rows["Gloves"].dot.styleSheet()

# -- branding --------------------------------------------------------------


SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'>"
    "<rect width='10' height='10' fill='red'/></svg>"
)


def test_loads_an_svg_logo_at_the_requested_size(app, tmp_path):
    from ppe.ui import load_logo

    path = tmp_path / "logo.svg"
    path.write_text(SVG)
    pixmap = load_logo(path, 40)
    assert pixmap is not None and pixmap.size().width() == 40


def test_a_rectangular_svg_logo_keeps_its_own_aspect_ratio(app, tmp_path):
    """A wide mark must come back wide, not squished into a height-square box."""
    from ppe.ui import load_logo

    wide = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 10'>"
        "<rect width='40' height='10' fill='red'/></svg>"
    )
    path = tmp_path / "wide.svg"
    path.write_text(wide)
    pixmap = load_logo(path, 40)
    assert pixmap is not None
    assert pixmap.height() == 40
    assert pixmap.width() == 160  # 4:1 source aspect, preserved


def test_a_rectangular_raster_logo_keeps_its_own_aspect_ratio(app, tmp_path):
    from ppe.ui import load_logo

    path = tmp_path / "wide.png"
    QImage(80, 20, QImage.Format_RGB32).save(str(path))
    pixmap = load_logo(path, 40)
    assert pixmap is not None
    assert pixmap.height() == 40
    assert pixmap.width() == 160  # 4:1 source aspect, preserved


def test_loads_a_raster_logo_too(app, tmp_path):
    from ppe.ui import load_logo

    path = tmp_path / "logo.png"
    QImage(20, 20, QImage.Format_RGB32).save(str(path))
    assert load_logo(path, 40) is not None


def test_a_high_dpi_logo_is_rendered_at_device_resolution(app, tmp_path):
    from ppe.ui import load_logo

    path = tmp_path / "logo.svg"
    path.write_text(SVG)
    pixmap = load_logo(path, 40, ratio=2.0)
    # Twice the pixels, same layout size — the mark stays crisp when scaled.
    assert pixmap.width() == 80
    assert pixmap.devicePixelRatio() == 2.0
    assert pixmap.deviceIndependentSize().width() == 40


@pytest.mark.parametrize(
    "name,content",
    [
        ("missing.svg", None),               # never created
        ("broken.svg", "not xml at all <<<"),
        ("empty.svg", ""),
        ("fake.png", "this is not a png"),
        ("mystery.xyz", "unsupported format"),
    ],
)
def test_a_bad_logo_is_ignored_rather_than_fatal(app, tmp_path, name, content):
    """The logo is meant to be swapped, so a bad file must never crash the app."""
    from ppe.ui import load_logo

    path = tmp_path / name
    if content is not None:
        path.write_text(content)
    assert load_logo(path, 40) is None


def test_no_logo_configured_is_not_an_error(app):
    from ppe.ui import load_logo

    assert load_logo(None, 40) is None


def branding(**kw):
    from ppe.config import BrandingCfg

    return BrandingCfg(**{"name": "A. Engineer", "tagline": "Vision & Automation", **kw})


def test_brand_strip_shows_only_the_logo(app, tmp_path):
    """No name or tagline text — just the mark, so it can't be squished by a
    layout built to share space with a label next to it."""
    from ppe.ui import BrandStrip

    (tmp_path / "logo.svg").write_text(SVG)
    strip = BrandStrip(branding(logo="logo.svg"), tmp_path)
    assert not [w.text() for w in strip.findChildren(QLabel) if w.text()]
    assert any(w.pixmap() and not w.pixmap().isNull() for w in strip.findChildren(QLabel))
    paint(strip, 340, 60)


def test_brand_strip_hides_itself_when_the_logo_is_missing(app, tmp_path):
    from ppe.ui import BrandStrip

    assert BrandStrip(branding(logo="vanished.svg"), tmp_path).isHidden()


def test_brand_strip_hides_itself_when_no_logo_is_configured(app, tmp_path):
    from ppe.ui import BrandStrip

    assert BrandStrip(branding(name="", tagline="", logo=""), tmp_path).isHidden()


def test_the_window_wears_the_logo_as_its_icon(app, station):
    window, _ = station
    assert not window.windowIcon().isNull()
    assert not window.brand.isHidden()


# -- deferred model loading in the window --------------------------------


def test_window_shows_immediately_without_the_model_loaded(app, clip, tmp_path, monkeypatch):
    """Construction must not touch pipeline.detector — it may not exist yet."""
    import ppe.pipeline as pipeline_mod
    from ppe.capture import CameraSet
    from ppe.config import CameraCfg, TowerCfg

    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    cfg = config()
    cfg.cameras = [CameraCfg("Line A", clip, 320, 240)]
    cfg.tower = TowerCfg(enabled=False)
    cameras = CameraSet(cfg.cameras)  # deliberately never started
    pipe = pipeline_mod.Pipeline(cfg, cameras)  # deliberately never started

    window = MainWindow(cfg, cameras, pipe)  # must not raise
    assert "LOADING MODEL" in window.banner.text()
    assert window.panel.picker.count() == 0
    # Not closed: the pipeline was deliberately never started, and close()
    # would try to join a thread that never ran — a real app always starts
    # the pipeline before the window exists, so this ordering never occurs.


def test_the_checklist_appears_before_the_model_does(app, clip, tmp_path, monkeypatch):
    """The row list comes from config, not the model — it needs no wait."""
    import ppe.pipeline as pipeline_mod
    from ppe.capture import CameraSet
    from ppe.config import CameraCfg, TowerCfg

    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    cfg = config()
    cfg.cameras = [CameraCfg("Line A", clip, 320, 240)]
    cfg.tower = TowerCfg(enabled=False)
    cameras = CameraSet(cfg.cameras)
    pipe = pipeline_mod.Pipeline(cfg, cameras)

    window = MainWindow(cfg, cameras, pipe)
    assert set(window.panel.rows) == {c.name for c in cfg.ppe.classes}


def test_model_ready_after_construction_populates_the_picker(app, station):
    window, pipe = station
    assert pump(app, lambda: window._model_loaded), "model_ready never reached the window"
    assert window.panel.picker.count() == len(pipe.detector.names)


def test_model_already_ready_before_the_window_exists_is_not_missed(
    app, clip, monkeypatch, tmp_path
):
    """The race this guards: the pipeline could finish loading, and even
    fire on_ready, before MainWindow exists to hear it."""
    import ppe.pipeline as pipeline_mod
    from ppe.capture import CameraSet
    from ppe.config import CameraCfg, TowerCfg

    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    cfg = config()
    cfg.cameras = [CameraCfg("Line A", clip, 320, 240)]
    cfg.tower = TowerCfg(enabled=False)
    cameras = CameraSet(cfg.cameras).start()
    pipe = pipeline_mod.Pipeline(cfg, cameras)
    assert pipe._load()  # loaded synchronously, before any window exists

    window = MainWindow(cfg, cameras, pipe)
    assert window._model_loaded
    assert window.panel.picker.count() > 0
    # Model is loaded, but no detection cycle has run yet (the thread was
    # never started) — the banner must say so distinctly, not keep blaming
    # a model load that already finished.
    assert "LOADING MODEL" not in window.banner.text()
    assert "MODEL READY" in window.banner.text()
    cameras.stop()  # the pipeline itself was never started; nothing to stop there


def test_a_model_that_already_failed_before_the_window_exists_shows_the_error(
    app, clip, monkeypatch, tmp_path
):
    import ppe.pipeline as pipeline_mod
    from ppe.capture import CameraSet
    from ppe.config import CameraCfg, TowerCfg

    class Broken(StubDetector):
        def __init__(self, *a, **k):
            raise RuntimeError("bad weights path")

    monkeypatch.setattr(pipeline_mod, "Detector", Broken)
    cfg = config()
    cfg.cameras = [CameraCfg("Line A", clip, 320, 240)]
    cfg.tower = TowerCfg(enabled=False)
    cameras = CameraSet(cfg.cameras)
    pipe = pipeline_mod.Pipeline(cfg, cameras)
    assert not pipe._load()

    window = MainWindow(cfg, cameras, pipe)
    assert "FAILED TO LOAD" in window.banner.text()
    assert "bad weights path" in window.banner.text()


def test_a_model_that_fails_after_the_window_exists_shows_the_error_live(
    app, clip, monkeypatch, tmp_path
):
    import ppe.pipeline as pipeline_mod
    from ppe.capture import CameraSet
    from ppe.config import CameraCfg, TowerCfg

    class Broken(StubDetector):
        def __init__(self, *a, **k):
            raise RuntimeError("camera-adjacent config error")

    monkeypatch.setattr(pipeline_mod, "Detector", Broken)
    cfg = config()
    cfg.cameras = [CameraCfg("Line A", clip, 320, 240)]
    cfg.tower = TowerCfg(enabled=False)
    cameras = CameraSet(cfg.cameras).start()
    pipe = pipeline_mod.Pipeline(cfg, cameras)
    window = MainWindow(cfg, cameras, pipe)
    pipe.start()

    assert pump(app, lambda: "FAILED TO LOAD" in window.banner.text())
    assert "camera-adjacent config error" in window.banner.text()
    window.close()
    pipe.stop()
    cameras.stop()


def test_the_banner_blames_the_camera_not_the_model_when_only_the_camera_is_slow(
    app, monkeypatch, tmp_path
):
    """The scenario this feature exists for: a camera that never opens must
    read as a video problem, never as a stuck model load — the model in
    this case loads (and a first, camera-less cycle publishes) in well
    under a millisecond, faster than the GUI thread can even be scheduled."""
    import ppe.pipeline as pipeline_mod
    from ppe.capture import CameraSet
    from ppe.config import CameraCfg, TowerCfg

    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    cfg = config()
    cfg.cameras = [CameraCfg("Line A", str(tmp_path / "never-appears.avi"), 320, 240)]
    cfg.tower = TowerCfg(enabled=False)
    cameras = CameraSet(cfg.cameras).start()
    pipe = pipeline_mod.Pipeline(cfg, cameras)
    window = MainWindow(cfg, cameras, pipe)
    pipe.start()

    assert pump(app, lambda: window._model_loaded)
    assert "LOADING MODEL" not in window.banner.text()
    # The card shows a headline and a detail line; the cause lives in the
    # detail, so match without caring which half it landed in.
    assert "no video signal" in window.banner.text().lower()

    window.close()
    pipe.stop()
    cameras.stop()


def test_model_ready_with_nothing_published_yet_shows_a_distinct_waiting_state(
    app, clip, monkeypatch, tmp_path
):
    """Deterministic version of the race above: the model is loaded (not via
    the thread, so no cycle has run) and pipeline.result is still None."""
    import ppe.pipeline as pipeline_mod
    from ppe.capture import CameraSet
    from ppe.config import CameraCfg, TowerCfg

    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    cfg = config()
    cfg.cameras = [CameraCfg("Line A", clip, 320, 240)]
    cfg.tower = TowerCfg(enabled=False)
    cameras = CameraSet(cfg.cameras)
    pipe = pipeline_mod.Pipeline(cfg, cameras)
    assert pipe._load()
    assert pipe.result is None

    window = MainWindow(cfg, cameras, pipe)
    assert "MODEL READY" in window.banner.text()
    assert "LOADING MODEL" not in window.banner.text()


def test_model_ready_with_a_result_already_published_applies_it_immediately(
    app, clip, monkeypatch, tmp_path
):
    """Deterministic version: loading finished AND a result already exists
    (set directly, not via the thread) before the window is constructed —
    _on_model_ready must apply it rather than showing "waiting"."""
    import ppe.pipeline as pipeline_mod
    from ppe.capture import CameraSet
    from ppe.config import CameraCfg, TowerCfg
    from ppe.tower import ClassState, Status

    monkeypatch.setattr(pipeline_mod, "Detector", StubDetector)
    cfg = config()
    cfg.cameras = [CameraCfg("Line A", clip, 320, 240)]
    cfg.tower = TowerCfg(enabled=False)
    cameras = CameraSet(cfg.cameras)
    pipe = pipeline_mod.Pipeline(cfg, cameras)
    assert pipe._load()
    pipe.result = pipeline_mod.Result(
        status=Status.OK,
        classes=[ClassState("helmet", "Hard Hat", True, "present", need=1, count=1, conf=0.9)],
        detections=[[]],
        ignored=[[]],
        subjects=[None],
        seqs=[1],
        tower_ok=True,
    )

    window = MainWindow(cfg, cameras, pipe)
    assert "ALL PPE PRESENT" in window.banner.text()
    assert "MODEL READY" not in window.banner.text()
    assert "LOADING" not in window.banner.text()
