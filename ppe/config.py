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
    threads: int = 0            # inference threads; 0 = every core
    batch: bool | str = "auto"  # "auto" | true | false — batch both cameras?

    def batches(self, device: str) -> bool:
        """Should both cameras go through one call?

        Only on a GPU. On a CPU a batch of two costs about twice a batch of
        one *and* doubles how long each frame waits for its own result, and
        ONNX/OpenVINO exports are fixed at batch 1 anyway.
        """
        if isinstance(self.batch, bool):
            return self.batch
        return device.startswith("cuda")


@dataclass
class CameraCfg:
    name: str = "Camera"
    source: Any = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    api: str = "any"
    fourcc: str = "MJPG"  # "" leaves the driver's default (usually raw, and much bigger)


# What a class's presence means for compliance.
EXPECT_PRESENT = "present"  # the item must be worn
EXPECT_ABSENT = "absent"    # the item must NOT appear (a violation class)
EXPECTATIONS = (EXPECT_PRESENT, EXPECT_ABSENT)


@dataclass
class ClassCfg:
    name: str                      # must match the model's class name
    label: str = ""                # shown in the UI
    required: bool = True          # does this class gate the tower light?
    expect: str = EXPECT_PRESENT   # "present" to require it, "absent" to forbid it
    count: int = 1                 # how many must be on the subject (2 gloves, 2 sleeves)
    conf: float | None = None      # per-class confidence override
    hold_ms: int | None = None     # per-class hold override; None uses ppe.hold_ms

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name.replace("_", " ").title()

    @property
    def forbidden(self) -> bool:
        """True when detecting this class is itself the violation."""
        return self.expect == EXPECT_ABSENT


@dataclass
class PPECfg:
    classes: list[ClassCfg] = field(default_factory=list)
    hold_ms: int = 1500        # default hold; a class's own hold_ms overrides it
    subject: str = ""          # class that gates the checks; empty = check always
    containment: float = 0.5   # fraction of a PPE box that must lie on the subject

    # How long a status must hold before the lamp follows it, in SECONDS —
    # not frames. Inference rate moves with CPU load, so a frame count is a
    # different amount of real time from one minute to the next.
    #
    # These are deliberately asymmetric. Going green is a safety claim and
    # should be slow; going red is an alarm and should be quick.
    confirm_sec: dict[str, float] = field(
        default_factory=lambda: {
            "ok": 1.0,        # slow to declare compliant
            "violation": 0.4,  # quick to alarm
            "standby": 1.0,    # slow to conclude nobody is there
            "degraded": 0.5,
        }
    )

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
    reconnect_sec: float = 5.0  # wait between reconnect attempts
    channels: int = 8           # channels on the board; all blanked at connect
    coils: dict[str, int] = field(
        default_factory=lambda: {"green": 0, "amber": 1, "red": 2, "buzzer": 3}
    )
    buzzer_on_violation: bool = False


@dataclass
class AudioCfg:
    """Spoken prompt while a violation stands. Empty `file` = silent."""

    file: str = ""          # .mp3/.wav; relative paths resolve from the config
    grace_sec: float = 3.0   # wait before the first prompt — time to comply
    repeat_sec: float = 3.0  # gap between repeats while the violation stands


@dataclass
class BrandingCfg:
    """Who built this station. Swap the logo by pointing `logo` at your file."""

    name: str = ""
    tagline: str = ""
    logo: str = ""      # .svg, .png or .jpg; relative paths resolve from the config

    def logo_path(self, base: Path | None = None) -> Path | None:
        """Absolute path to the logo, or None if unset or missing on disk."""
        if not self.logo:
            return None
        path = Path(self.logo)
        if not path.is_absolute() and base is not None:
            path = base / path
        return path if path.is_file() else None


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
    branding: BrandingCfg = field(default_factory=BrandingCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
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
            branding=_build(BrandingCfg, raw.get("branding") or {}),
            audio=_build(AudioCfg, raw.get("audio") or {}),
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

    @property
    def base_dir(self) -> Path:
        """Directory relative paths in the config resolve against."""
        return self.path.parent if self.path is not None else Path.cwd()

    # -- validation --------------------------------------------------------
    def validate(self) -> None:
        if not self.cameras:
            raise ValueError("config: at least one camera is required")
        if self.model.imgsz % 32:
            raise ValueError(
                f"config: model.imgsz must be a multiple of 32, got {self.model.imgsz}"
            )
        if self.audio.grace_sec < 0 or self.audio.repeat_sec <= 0:
            raise ValueError(
                "config: audio.grace_sec must be >= 0 and audio.repeat_sec > 0"
            )
        if self.model.threads < 0:
            raise ValueError(f"config: model.threads must be >= 0, got {self.model.threads}")
        if not isinstance(self.model.batch, bool) and self.model.batch != "auto":
            raise ValueError(
                f"config: model.batch must be true, false or 'auto', got {self.model.batch!r}"
            )
        if not self.ppe.classes:
            raise ValueError("config: ppe.classes is empty — nothing to detect")
        names = [c.name for c in self.ppe.classes]
        if len(names) != len(set(names)):
            raise ValueError("config: duplicate class names in ppe.classes")
        if self.ppe.subject and self.ppe.subject not in names:
            # The detector filters NMS down to the configured classes, so a
            # subject that is not listed would never be detected at all.
            raise ValueError(
                f"config: ppe.subject {self.ppe.subject!r} must also appear in ppe.classes"
            )
        valid_status = {"ok", "violation", "standby", "degraded"}
        unknown = set(self.ppe.confirm_sec) - valid_status
        if unknown:
            raise ValueError(
                f"config: ppe.confirm_sec has unknown status {sorted(unknown)}; "
                f"expected any of {sorted(valid_status)}"
            )
        for status, secs in self.ppe.confirm_sec.items():
            if secs < 0:
                raise ValueError(f"config: ppe.confirm_sec[{status}] must be >= 0, got {secs}")
        for klass in self.ppe.classes:
            if klass.hold_ms is not None and klass.hold_ms < 0:
                raise ValueError(
                    f"config: {klass.name}.hold_ms must be >= 0, got {klass.hold_ms}"
                )
        if not 0.0 <= self.ppe.containment <= 1.0:
            raise ValueError(
                f"config: ppe.containment must be between 0 and 1, got {self.ppe.containment}"
            )
        for klass in self.ppe.classes:
            if klass.expect not in EXPECTATIONS:
                raise ValueError(
                    f"config: {klass.name}.expect must be one of {list(EXPECTATIONS)}, "
                    f"got {klass.expect!r}"
                )
            if klass.count < 1:
                raise ValueError(
                    f"config: {klass.name}.count must be at least 1, got {klass.count}"
                )
            if klass.forbidden and klass.count != 1:
                # "two of them are a violation but one is fine" is not a rule
                # anyone means; a forbidden class is a fault the moment it appears.
                raise ValueError(
                    f"config: {klass.name} is expect:absent, so count must be 1, "
                    f"got {klass.count}"
                )
        missing = set(self.tower.coils) - {"green", "amber", "red", "buzzer"}
        if missing:
            raise ValueError(f"config: unknown tower coils {sorted(missing)}")
