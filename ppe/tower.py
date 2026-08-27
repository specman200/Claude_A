"""PPE compliance state machine and the Modbus tower light it drives."""

from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import dataclass
from enum import Enum

from .config import PPECfg, TowerCfg
from .detector import Detection
from .latency import now

log = logging.getLogger(__name__)


class Status(Enum):
    OK = "ok"                # every required item present
    VIOLATION = "violation"  # at least one required item missing
    DEGRADED = "degraded"    # can't judge: no camera, or classes the model lacks


@dataclass(slots=True)
class ClassState:
    """What the UI renders for one row of the required-PPE list."""

    name: str
    label: str
    required: bool
    present: bool = False
    conf: float = 0.0
    last_seen: float = 0.0
    available: bool = True   # False when the model has no such class


class ComplianceMonitor:
    """Turns per-frame detections into a stable station status.

    Two things keep the light steady rather than strobing: a *hold* window, so
    an item flickering out for a frame stays lit, and a *confirm* count, so the
    lamp only changes after several evaluations agree.
    """

    def __init__(self, cfg: PPECfg, unavailable: list[str] | None = None) -> None:
        self.hold = cfg.hold_ms / 1000.0
        self.confirm = max(1, cfg.confirm_frames)
        missing = set(unavailable or ())
        self.classes = [
            ClassState(c.name, c.label, c.required, available=c.name not in missing)
            for c in cfg.classes
        ]
        self._by_name = {c.name: c for c in self.classes}
        self.status = Status.DEGRADED
        self._pending = Status.DEGRADED
        self._agree = 0

    def update(self, detections: list[Detection], t: float | None = None) -> Status:
        """Fold one cycle's detections (all cameras) into the station status."""
        t = now() if t is None else t
        for det in detections:
            state = self._by_name.get(det.name)
            if state is None:
                continue
            # Keep the best confidence seen this cycle, across both cameras.
            state.conf = det.conf if state.last_seen < t else max(state.conf, det.conf)
            state.last_seen = t

        for state in self.classes:
            state.present = state.last_seen > 0 and (t - state.last_seen) <= self.hold

        return self._debounce(self._evaluate())

    def _evaluate(self) -> Status:
        required = [c for c in self.classes if c.required]
        if not required or any(not c.available for c in required):
            return Status.DEGRADED
        return Status.OK if all(c.present for c in required) else Status.VIOLATION

    def _debounce(self, candidate: Status) -> Status:
        if candidate == self.status:
            self._agree = 0
            self._pending = candidate
            return self.status
        if candidate != self._pending:
            self._pending, self._agree = candidate, 0
        self._agree += 1
        if self._agree >= self.confirm:
            log.info("status %s -> %s", self.status.value, candidate.value)
            self.status = candidate
            self._agree = 0
        return self.status

    def missing(self) -> list[str]:
        """Required items not currently seen."""
        return [c.label for c in self.classes if c.required and not c.present]

    def unavailable(self) -> list[str]:
        """Required items the loaded model has no class for."""
        return [c.label for c in self.classes if c.required and not c.available]

    def degrade(self) -> Status:
        """Force DEGRADED — used when no camera is delivering frames."""
        return self._debounce(Status.DEGRADED)


# --------------------------------------------------------------------------
# Modbus output
# --------------------------------------------------------------------------

# Which lamps are energised in each status.
LAMPS: dict[Status, tuple[str, ...]] = {
    Status.OK: ("green",),
    Status.VIOLATION: ("red",),
    Status.DEGRADED: ("amber",),
}


class TowerLight:
    """Modbus coil output. Writes only the coils that actually changed."""

    def __init__(self, cfg: TowerCfg) -> None:
        self.cfg = cfg
        self.connected = False
        self._client = None
        self._kw: str = "slave"
        self._state: dict[str, bool] = dict.fromkeys(cfg.coils, False)
        self._lock = threading.Lock()
        self._retry_at = 0.0

    # -- connection --------------------------------------------------------
    def connect(self) -> bool:
        if self._client is not None and self.connected:
            return True
        if now() < self._retry_at:
            return False
        try:
            self._client = self._make_client()
            self.connected = bool(self._client.connect())
        except Exception as exc:  # noqa: BLE001 — the line is allowed to be down
            log.warning("tower connect failed: %s", exc)
            self.connected = False
        if not self.connected:
            self._retry_at = now() + 2.0
        else:
            log.info("tower connected (%s)", self.cfg.transport)
            self._state = dict.fromkeys(self.cfg.coils)  # force a resync
        return self.connected

    def _make_client(self):
        if self.cfg.transport == "rtu":
            from pymodbus.client import ModbusSerialClient

            client = ModbusSerialClient(
                port=self.cfg.serial_port, baudrate=self.cfg.baudrate, timeout=self.cfg.timeout
            )
        else:
            from pymodbus.client import ModbusTcpClient

            client = ModbusTcpClient(
                host=self.cfg.host, port=self.cfg.port, timeout=self.cfg.timeout
            )
        # pymodbus has renamed the unit-id keyword across releases; pick the
        # one this installation actually accepts, once.
        params = inspect.signature(client.write_coil).parameters
        self._kw = next((k for k in ("slave", "device_id", "unit") if k in params), "slave")
        return client

    # -- output ------------------------------------------------------------
    def apply(self, status: Status) -> bool:
        """Drive the lamps for ``status``. Returns True if the bus was written."""
        wanted = dict.fromkeys(self.cfg.coils, False)
        for lamp in LAMPS[status]:
            if lamp in wanted:
                wanted[lamp] = True
        if "buzzer" in wanted:
            wanted["buzzer"] = self.cfg.buzzer_on_violation and status is Status.VIOLATION
        return self.write(wanted)

    def write(self, wanted: dict[str, bool]) -> bool:
        with self._lock:
            if not self.connect():
                return False
            changed = [k for k, v in wanted.items() if self._state.get(k) != v]
            if not changed:
                return False
            try:
                for name in changed:
                    rsp = self._client.write_coil(
                        self.cfg.coils[name], wanted[name], **{self._kw: self.cfg.unit}
                    )
                    if rsp is not None and getattr(rsp, "isError", bool)():
                        raise OSError(str(rsp))
                    self._state[name] = wanted[name]
            except Exception as exc:  # noqa: BLE001
                log.warning("tower write failed: %s", exc)
                self.connected = False
                self._retry_at = now() + 2.0
                return False
            return True

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                try:
                    self.write(dict.fromkeys(self.cfg.coils, False))
                    self._client.close()
                except Exception:  # noqa: BLE001
                    pass
            self._client = None
            self.connected = False


class NullTower:
    """Stand-in when the tower is disabled, so the pipeline stays branch-free."""

    connected = False

    def apply(self, status: Status) -> bool:  # noqa: ARG002 — interface parity
        return False

    def close(self) -> None:
        pass


def make_tower(cfg: TowerCfg) -> TowerLight | NullTower:
    return TowerLight(cfg) if cfg.enabled else NullTower()
