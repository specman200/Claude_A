"""Typed configuration, loaded from and saved back to a single YAML file."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _build(cls, data: dict[str, Any]):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ModelCfg:
    weights: str = "yolo11s.pt"
    imgsz: int = 640
    device: str = "auto"
    half: bool = True
    conf: float = 0.35
    iou: float = 0.50
    max_det: int = 100
    warmup: bool = True


@dataclass
class CameraCfg:
    name: str = "Camera"
    source: Any = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    api: str = "any"


@dataclass
class ClassCfg:
    name: str                      # must match the model's class name
    label: str = ""                # shown in the UI
    required: bool = True
    conf: float | None = None      # per-class confidence override

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name.replace("_", " ").title()


@dataclass
class PPECfg:
    classes: list[ClassCfg] = field(default_factory=list)
    hold_ms: int = 1500
    confirm_frames: int = 3

    @property
    def required(self) -> list[ClassCfg]:
        return [c for c in self.classes if c.required]


@dataclass
class TowerCfg:
    enabled: bool = True
    transport: str = "tcp"
    host: str = "192.168.1.50"
    port: int = 502
    serial_port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    unit: int = 1
    timeout: float = 0.5
    coils: dict[str, int] = field(
        default_factory=lambda: {"green": 0, "amber": 1, "red": 2, "buzzer": 3}
    )
    buzzer_on_violation: bool = False


@dataclass
class TelemetryCfg:
    csv: str = "logs/latency.csv"
    window: int = 300
    print_every: float = 0.0


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    cameras: list[CameraCfg] = field(default_factory=list)
    ppe: PPECfg = field(default_factory=PPECfg)
    tower: TowerCfg = field(default_factory=TowerCfg)
    telemetry: TelemetryCfg = field(default_factory=TelemetryCfg)
    path: Path | None = None

    # -- io ----------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        ppe_raw = dict(raw.get("ppe") or {})
        ppe = _build(PPECfg, ppe_raw)
        ppe.classes = [_build(ClassCfg, c) for c in ppe_raw.get("classes", [])]
        return cls(
            model=_build(ModelCfg, raw.get("model") or {}),
            cameras=[_build(CameraCfg, c) for c in raw.get("cameras") or []],
            ppe=ppe,
            tower=_build(TowerCfg, raw.get("tower") or {}),
            telemetry=_build(TelemetryCfg, raw.get("telemetry") or {}),
            path=path,
        )

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or self.path or "config.yaml")
        data = {k: v for k, v in asdict(self).items() if k != "path"}
        data["ppe"]["classes"] = [
            {k: v for k, v in c.items() if v is not None} for c in data["ppe"]["classes"]
        ]
        path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
        self.path = path
        return path

    # -- validation --------------------------------------------------------
    def validate(self) -> None:
        if not self.cameras:
            raise ValueError("config: at least one camera is required")
        if self.model.imgsz % 32:
            raise ValueError(
                f"config: model.imgsz must be a multiple of 32, got {self.model.imgsz}"
            )
        if not self.ppe.classes:
            raise ValueError("config: ppe.classes is empty — nothing to detect")
        names = [c.name for c in self.ppe.classes]
        if len(names) != len(set(names)):
            raise ValueError("config: duplicate class names in ppe.classes")
        missing = set(self.tower.coils) - {"green", "amber", "red", "buzzer"}
        if missing:
            raise ValueError(f"config: unknown tower coils {sorted(missing)}")
