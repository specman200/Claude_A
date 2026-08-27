"""Qt front-end: two live panes, an editable PPE checklist, and a latency HUD."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .capture import CameraSet, Frame
from .config import BrandingCfg, ClassCfg, Config
from .detector import Detection
from .latency import STAGES
from .letterbox import fit
from .pipeline import Pipeline, Result
from .tower import ClassState, Status

# One accent per class, so a box's colour matches its row in the checklist.
PALETTE = ["#38bdf8", "#a78bfa", "#f472b6", "#fbbf24", "#34d399", "#fb923c", "#60a5fa", "#f87171"]
PRESENT = "#22c55e"
ABSENT = "#ef4444"
IDLE = "#4b5563"
UNAVAILABLE = "#f59e0b"

BANNER = {
    Status.OK: ("ALL PPE PRESENT", "#16a34a"),
    Status.VIOLATION: ("PPE MISSING", "#dc2626"),
    Status.DEGRADED: ("NOT READY", "#d97706"),
}

STYLE = """
QWidget       { background:#0f1216; color:#e5e7eb;
                font-family:"Segoe UI","Helvetica Neue",sans-serif; font-size:13px; }
QFrame#card   { background:#161b22; border:1px solid #232a33; border-radius:10px; }
QLabel#h      { color:#9aa4b2; font-size:11px; font-weight:600; letter-spacing:1.2px; }
QPushButton   { background:#232a33; border:1px solid #2f3742; border-radius:6px; padding:6px 12px; }
QPushButton:hover { background:#2b333d; }
QCheckBox::indicator { width:15px; height:15px; }
QComboBox, QDoubleSpinBox { background:#1c222b; border:1px solid #2f3742;
                            border-radius:6px; padding:3px 6px; }
"""


def card(title: str) -> tuple[QFrame, QVBoxLayout]:
    """A titled panel — the only container style this UI uses."""
    frame = QFrame()
    frame.setObjectName("card")
    box = QVBoxLayout(frame)
    box.setContentsMargins(14, 12, 14, 12)
    box.setSpacing(8)
    heading = QLabel(title)
    heading.setObjectName("h")
    box.addWidget(heading)
    return frame, box


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


class VideoPane(QWidget):
    """Draws the newest frame and the newest boxes with one shared transform.

    Frames are painted as they arrive, independent of the detector, so the feed
    stays smooth even when inference runs slower than the camera.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._buf = None          # keeps the numpy buffer alive behind QImage
        self._image: QImage | None = None
        self._dets: list[Detection] = []
        self._colors: dict[str, str] = {}
        self.seq = -1
        self.fps = 0.0
        self.online = False

    def show_frame(self, frame: Frame, fps: float, online: bool) -> None:
        self._buf = frame.image
        h, w = self._buf.shape[:2]
        # Format_BGR888 consumes OpenCV's byte order directly — no colour convert.
        self._image = QImage(self._buf.data, w, h, self._buf.strides[0], QImage.Format_BGR888)
        self.seq, self.fps, self.online = frame.seq, fps, online
        self.update()

    def show_detections(self, dets: list[Detection], colors: dict[str, str]) -> None:
        self._dets, self._colors = dets, colors
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 — Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b0e11"))
        if self._image is None:
            self._banner(painter, f"{self.name} — waiting for video")
            return

        src_w, src_h = self._image.width(), self._image.height()
        scale, ox, oy = fit(src_w, src_h, self.width(), self.height())
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawImage(QRectF(ox, oy, src_w * scale, src_h * scale), self._image)

        # Boxes go through the very same scale/offset as the pixels they mark.
        painter.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
        for det in self._dets:
            x1, y1, x2, y2 = det.xyxy
            rect = QRectF(ox + x1 * scale, oy + y1 * scale,
                          (x2 - x1) * scale, (y2 - y1) * scale)
            color = QColor(self._colors.get(det.name, "#38bdf8"))
            painter.setPen(QPen(color, 2))
            painter.drawRect(rect)

            text = f"{det.name} {det.conf:.2f}"
            metrics = painter.fontMetrics()
            tw, th = metrics.horizontalAdvance(text) + 8, metrics.height() + 2
            ty = rect.top() - th if rect.top() > th else rect.top()
            painter.fillRect(QRectF(rect.left(), ty, tw, th), color)
            painter.setPen(QColor("#0b0e11"))
            painter.drawText(QRectF(rect.left() + 4, ty, tw, th), Qt.AlignVCenter, text)

        self._banner(painter, f"{self.name}   {src_w}x{src_h}   {self.fps:4.1f} fps"
                     + ("" if self.online else "   OFFLINE"))

    def _banner(self, painter: QPainter, text: str) -> None:
        painter.setPen(QColor("#0b0e11"))
        painter.fillRect(0, 0, self.width(), 24, QColor(0, 0, 0, 150))
        painter.setPen(QColor("#e5e7eb" if self.online else UNAVAILABLE))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(QRectF(8, 0, self.width() - 16, 24), Qt.AlignVCenter, text)


# ---------------------------------------------------------------------------
# PPE checklist
# ---------------------------------------------------------------------------


class ClassRow(QWidget):
    """One PPE item: required toggle, name, confidence floor, live colour."""

    changed = Signal()
    removed = Signal(str)

    def __init__(self, cfg: ClassCfg, color: str, default_conf: float) -> None:
        super().__init__()
        self.cfg = cfg
        self.color = color

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(8)

        self.dot = QLabel()
        self.dot.setFixedSize(12, 12)

        self.required = QCheckBox()
        self.required.setChecked(cfg.required)
        self.required.setToolTip("Required — drives the tower light")
        self.required.toggled.connect(self._on_required)

        self.label = QLabel(cfg.label)
        self.label.setToolTip(f"model class: {cfg.name}")
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.05, 0.95)
        self.conf.setSingleStep(0.05)
        self.conf.setDecimals(2)
        self.conf.setFixedWidth(66)
        self.conf.setValue(cfg.conf if cfg.conf is not None else default_conf)
        self.conf.setToolTip("Confidence floor for this class")
        self.conf.valueChanged.connect(self._on_conf)

        self.score = QLabel("--")
        self.score.setFixedWidth(34)
        self.score.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        remove = QPushButton("\u00d7")  # ×
        remove.setFixedSize(24, 24)
        # The shared button padding would leave no room for the glyph here.
        remove.setStyleSheet("padding:0; color:#9aa4b2; font-size:16px; font-weight:700;")
        remove.setToolTip("Remove this class")
        remove.clicked.connect(lambda: self.removed.emit(self.cfg.name))

        for widget in (self.dot, self.required, self.label, self.score, self.conf, remove):
            row.addWidget(widget)

        # Start in the "not seen yet" look rather than an inherited default.
        self.apply(ClassState(cfg.name, cfg.label, cfg.required))

    def _paint_dot(self, color: str) -> None:
        self.dot.setStyleSheet(f"background:{color}; border-radius:6px;")

    def _on_required(self, on: bool) -> None:
        self.cfg.required = on
        self.changed.emit()

    def _on_conf(self, value: float) -> None:
        self.cfg.conf = round(value, 2)
        self.changed.emit()

    def apply(self, state) -> None:
        """Recolour from the live class state — the app's main visual feedback."""
        if not state.available:
            self._paint_dot(UNAVAILABLE)
            self.label.setStyleSheet(f"color:{UNAVAILABLE};")
            self.label.setToolTip(f"'{self.cfg.name}' is not a class in the loaded model")
            self.score.setText("n/a")
            return
        if state.present:
            color, text = PRESENT, f"{state.conf:.2f}"
        elif state.required:
            color, text = ABSENT, "--"
        else:
            color, text = IDLE, "--"
        self._paint_dot(color)
        self.label.setStyleSheet(f"color:{color}; font-weight:{600 if state.present else 400};")
        self.score.setText(text)


class PPEPanel(QFrame):
    """The editable checklist. Edits apply live; Save writes them to YAML."""

    edited = Signal()

    def __init__(self, cfg: Config, model_names: list[str]) -> None:
        super().__init__()
        self.cfg = cfg
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)
        heading = QLabel("REQUIRED PPE")
        heading.setObjectName("h")
        outer.addWidget(heading)

        self.rows: dict[str, ClassRow] = {}
        self._list = QVBoxLayout()
        self._list.setSpacing(2)
        outer.addLayout(self._list)
        outer.addSpacing(6)

        self.picker = QComboBox()
        self.picker.setEditable(True)
        self.picker.addItems(sorted(model_names))
        self.picker.setCurrentText("")
        self.picker.lineEdit().setPlaceholderText("add a model class…")
        add = QPushButton("Add")
        add.clicked.connect(self._add_from_picker)
        save = QPushButton("Save to config")
        save.clicked.connect(self._save)
        self.saved = QLabel("")
        self.saved.setObjectName("h")

        controls = QHBoxLayout()
        controls.addWidget(self.picker, 1)
        controls.addWidget(add)
        outer.addLayout(controls)
        footer = QHBoxLayout()
        footer.addWidget(save)
        footer.addWidget(self.saved, 1)
        outer.addLayout(footer)

        self.rebuild()

    def colors(self) -> dict[str, str]:
        return {name: row.color for name, row in self.rows.items()}

    def rebuild(self) -> None:
        while self._list.count():
            item = self._list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.rows.clear()
        for i, klass in enumerate(self.cfg.ppe.classes):
            row = ClassRow(klass, PALETTE[i % len(PALETTE)], self.cfg.model.conf)
            row.changed.connect(self.edited)
            row.removed.connect(self._remove)
            self.rows[klass.name] = row
            self._list.addWidget(row)

    def apply(self, states) -> None:
        for state in states:
            row = self.rows.get(state.name)
            if row is not None:
                row.apply(state)

    def _add_from_picker(self) -> None:
        name = self.picker.currentText().strip()
        if not name or name in self.rows:
            return
        self.cfg.ppe.classes.append(ClassCfg(name=name))
        self.picker.setCurrentText("")
        self.rebuild()
        self.edited.emit()

    def _remove(self, name: str) -> None:
        self.cfg.ppe.classes = [c for c in self.cfg.ppe.classes if c.name != name]
        self.rebuild()
        self.edited.emit()

    def _save(self) -> None:
        path = self.cfg.save()
        self.saved.setText(f"saved → {path}")
        QTimer.singleShot(2500, lambda: self.saved.setText(""))


# ---------------------------------------------------------------------------
# Status + latency
# ---------------------------------------------------------------------------


class StatusBanner(QLabel):
    """Mirrors the tower light so the screen and the lamp never disagree."""

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(64)
        self.setWordWrap(True)
        self.setFont(QFont("Segoe UI", 17, QFont.Bold))
        self.apply(Status.DEGRADED, [], False)

    def apply(
        self,
        status: Status,
        missing: list[str],
        tower_ok: bool,
        unavailable: list[str] | None = None,
    ) -> None:
        text, color = BANNER[status]
        if status is Status.VIOLATION and missing:
            text = f"{text}: {', '.join(missing)}"
        elif status is Status.DEGRADED:
            # Two very different faults share this lamp — name the right one.
            text = (
                f"MODEL HAS NO CLASS FOR: {', '.join(unavailable)}"
                if unavailable
                else "NO VIDEO SIGNAL"
            )
        if not tower_ok:
            text += "   (tower offline)"
        self.setText(text)
        self.setStyleSheet(
            f"background:{color}; color:#0b0e11; border-radius:10px; padding:10px;"
        )


class LatencyHUD(QFrame):
    """Where the end-to-end budget goes, refreshed from the rolling metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        heading = QLabel("END-TO-END LATENCY")
        heading.setObjectName("h")
        outer.addWidget(heading)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(3)
        mono = QFont()
        mono.setStyleHint(QFont.Monospace)
        mono.setFamily(mono.defaultFamily())
        for col, name in enumerate(("stage", "p50", "p95", "max")):
            head = QLabel(name)
            head.setObjectName("h")
            grid.addWidget(head, 0, col, alignment=Qt.AlignRight if col else Qt.AlignLeft)

        self.cells: dict[str, list[QLabel]] = {}
        for r, stage in enumerate((*STAGES, "render", "end_to_end"), start=1):
            name = QLabel("total" if stage == "end_to_end" else stage)
            if stage == "end_to_end":
                name.setStyleSheet("font-weight:700;")
            grid.addWidget(name, r, 0)
            cells = []
            for col in (1, 2, 3):
                value = QLabel("-")
                value.setFont(mono)
                value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                grid.addWidget(value, r, col)
                cells.append(value)
            self.cells[stage] = cells
        outer.addLayout(grid)

        self.rate = QLabel("-")
        self.rate.setObjectName("h")
        outer.addWidget(self.rate)

    def apply(self, snapshot: dict, infer_fps: float) -> None:
        for stage, cells in self.cells.items():
            stats = snapshot.get(stage)
            for cell, key in zip(cells, ("p50", "p95", "max"), strict=True):
                cell.setText(f"{stats[key]:.1f}" if stats and stats["n"] else "-")
        self.rate.setText(f"{infer_fps:.1f} inferences/s   (ms per stage)")


# ---------------------------------------------------------------------------
# Branding
# ---------------------------------------------------------------------------


def load_logo(path: Path | None, size: int, ratio: float = 1.0) -> QPixmap | None:
    """Render a logo file to a crisp ``size``-square pixmap, or None.

    This file is meant to be swapped, so every failure — missing, unreadable,
    an unsupported format — returns None and the UI simply omits the mark.
    A logo must never be able to take the station down.
    """
    if path is None:
        return None
    px = max(1, int(size * ratio))
    try:
        if path.suffix.lower() in (".svg", ".svgz"):
            from PySide6.QtSvg import QSvgRenderer

            renderer = QSvgRenderer(str(path))
            if not renderer.isValid():
                return None
            # Render at device resolution rather than scaling a small raster.
            pixmap = QPixmap(px, px)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing, True)
            renderer.render(painter, QRectF(0, 0, px, px))
            painter.end()
        else:
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                return None
            pixmap = pixmap.scaled(px, px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    except Exception:  # noqa: BLE001 — a bad logo is cosmetic, never fatal
        return None
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


class BrandStrip(QFrame):
    """Logo, name and tagline. Hidden entirely when no name is configured."""

    LOGO = 38

    def __init__(self, cfg: BrandingCfg, base: Path, ratio: float = 1.0) -> None:
        super().__init__()
        self.setObjectName("card")
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(12)

        pixmap = load_logo(cfg.logo_path(base), self.LOGO, ratio)
        if pixmap is not None:
            mark = QLabel()
            mark.setPixmap(pixmap)
            mark.setFixedSize(self.LOGO, self.LOGO)
            mark.setScaledContents(True)
            row.addWidget(mark)

        text = QVBoxLayout()
        text.setSpacing(1)
        name = QLabel(cfg.name)
        name.setStyleSheet("font-size:14px; font-weight:700;")
        text.addWidget(name)
        if cfg.tagline:
            tagline = QLabel(cfg.tagline)
            tagline.setObjectName("h")
            tagline.setWordWrap(True)
            text.addWidget(tagline)
        row.addLayout(text, 1)

        if not cfg.name:
            self.hide()


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    result_ready = Signal(object)

    def __init__(self, cfg: Config, cameras: CameraSet, pipeline: Pipeline) -> None:
        super().__init__()
        self.cfg = cfg
        self.cameras = cameras
        self.pipeline = pipeline
        self.setWindowTitle("PPE Detection Station")
        self.resize(1500, 880)
        self.setStyleSheet(STYLE)

        self.panes = [VideoPane(c.cfg.name) for c in cameras.cameras]
        video = QSplitter(Qt.Horizontal if len(self.panes) <= 2 else Qt.Vertical)
        for pane in self.panes:
            video.addWidget(pane)

        base = cfg.base_dir
        ratio = self.devicePixelRatioF() or 1.0
        icon = load_logo(cfg.branding.logo_path(base), 256, 1.0)
        if icon is not None:
            self.setWindowIcon(QIcon(icon))
        self.brand = BrandStrip(cfg.branding, base, ratio)

        self.banner = StatusBanner()
        self.panel = PPEPanel(cfg, list(pipeline.detector.names.values()))
        self.panel.edited.connect(self._on_edit)
        self.hud = LatencyHUD()

        side = QWidget()
        side.setFixedWidth(360)
        column = QVBoxLayout(side)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(12)
        column.addWidget(self.banner)
        column.addWidget(self.panel)
        column.addWidget(self.hud)
        column.addStretch(1)
        column.addWidget(self.brand)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(video, 1)
        layout.addWidget(side)
        self.setCentralWidget(root)

        # Results arrive on the pipeline thread; the signal hops them to the GUI.
        self.result_ready.connect(self._on_result)
        pipeline.on_result = self.result_ready.emit

        self._video = QTimer(self, interval=16, timeout=self._draw_frames)
        self._video.start()
        self._stats = QTimer(self, interval=500, timeout=self._draw_stats)
        self._stats.start()

    # -- refresh -----------------------------------------------------------
    def _draw_frames(self) -> None:
        """Repaint only panes with a genuinely new frame."""
        with self.pipeline.metrics.measure("render"):
            for pane, cam in zip(self.panes, self.cameras.cameras, strict=True):
                frame = cam.latest()
                if frame is not None and frame.seq != pane.seq:
                    pane.show_frame(frame, cam.fps, cam.connected)

    def _on_result(self, result: Result) -> None:
        colors = self.panel.colors()
        for pane, dets in zip(self.panes, result.detections, strict=True):
            pane.show_detections(dets, colors)
        self.panel.apply(result.classes)
        self.banner.apply(
            result.status, result.missing, result.tower_ok, result.unavailable
        )

    def _draw_stats(self) -> None:
        self.hud.apply(self.pipeline.metrics.snapshot(), self.pipeline.infer_fps)

    def _on_edit(self) -> None:
        self.pipeline.reconfigure()

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt naming
        self._video.stop()
        self._stats.stop()
        self.pipeline.stop()
        self.cameras.stop()
        super().closeEvent(event)
