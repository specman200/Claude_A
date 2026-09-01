"""PPE compliance state machine and the Modbus tower light it drives."""

from __future__ import annotations

import contextlib
import inspect
import logging
import threading
from dataclasses import dataclass
from enum import Enum

from .config import EXPECT_ABSENT, PPECfg, TowerCfg
from .detector import Detection
from .latency import now

log = logging.getLogger(__name__)


class Status(Enum):
    OK = "ok"                # the subject is wearing everything required
    VIOLATION = "violation"  # a required item is missing, or a forbidden one appeared
    STANDBY = "standby"      # nobody to check — nothing to judge
    DEGRADED = "degraded"    # can't judge: no camera, or classes the model lacks


@dataclass(slots=True)
class ClassState:
    """What the UI renders for one row of the required-PPE list."""

    name: str
    label: str
    required: bool
    expect: str = "present"  # "absent" for a class whose presence is the fault
    need: int = 1            # how many the subject must be wearing
    hold: float = 1.5        # seconds this class stays "seen" after its last sighting
    count: int = 0           # how many are on them right now
    conf: float = 0.0
    last_seen: float = 0.0
    counted_at: float = 0.0  # when `count` was observed, for the hold window
    available: bool = True   # False when the model has no such class

    @property
    def present(self) -> bool:
        return self.count > 0

    @property
    def forbidden(self) -> bool:
        return self.expect == EXPECT_ABSENT

    @property
    def compliant(self) -> bool:
        """Is this class currently in the state the site rules want?"""
        if self.forbidden:
            return self.count == 0
        return self.count >= self.need

    @property
    def shortfall(self) -> int:
        """How many are still missing; 0 once the rule is satisfied."""
        return 0 if self.forbidden else max(0, self.need - self.count)


class ComplianceMonitor:
    """Turns per-frame detections into a stable station status.

    Two things keep the light steady rather than strobing: a *hold* window, so
    an item flickering out for a frame stays lit, and a *confirm* count, so the
    lamp only changes after several evaluations agree.
    """

    def __init__(self, cfg: PPECfg, unavailable: list[str] | None = None) -> None:
        self.subject_name = cfg.subject
        # Confirm times are per status and in seconds — see PPECfg.confirm_sec.
        self.confirm = {Status(k): float(v) for k, v in cfg.confirm_sec.items()}
        missing = set(unavailable or ())
        self.classes = [
            ClassState(
                c.name, c.label, c.required, c.expect, c.count,
                hold=(c.hold_ms if c.hold_ms is not None else cfg.hold_ms) / 1000.0,
                available=c.name not in missing,
            )
            for c in cfg.classes
        ]
        self._by_name = {c.name: c for c in self.classes}
        # The subject rides the same hold window as everything else, so one
        # dropped frame does not drop the station into standby.
        self._subject = self._by_name.get(cfg.subject) if cfg.subject else None
        self.status = Status.DEGRADED     # what the lamp is showing
        self.raw = Status.DEGRADED        # this cycle's verdict, before debounce
        self.candidate = Status.DEGRADED  # what is waiting to be confirmed
        self.candidate_since = 0.0

    def update(
        self, per_camera: list[list[Detection]], t: float | None = None
    ) -> Status:
        """Fold one cycle's detections into the station status.

        Counts are taken as the **best single camera's** view, never the sum:
        two cameras looking at one worker both see the same two gloves, so
        adding them up would report four and pass a one-gloved worker.
        """
        t = now() if t is None else t

        for state in self.classes:
            seen = 0
            best_conf = 0.0
            for dets in per_camera:
                found = [d for d in dets if d.name == state.name]
                if len(found) > seen:
                    seen = len(found)
                best_conf = max(best_conf, *(d.conf for d in found)) if found else best_conf

            # The hold window keeps the best recent count, so a glove the model
            # loses for a frame does not read as a bare hand.
            if seen >= state.count or (t - state.counted_at) > state.hold:
                state.count = seen
                state.counted_at = t
            if seen:
                state.last_seen = t
                state.conf = best_conf

        return self._debounce(self._evaluate(), t)

    def _evaluate(self) -> Status:
        required = [c for c in self.classes if c.required]
        if not required or any(not c.available for c in required):
            return Status.DEGRADED
        if self._subject is not None and not self._subject.present:
            # No one in the cell: there is no PPE to be missing.
            return Status.STANDBY
        return Status.OK if all(c.compliant for c in required) else Status.VIOLATION

    def _debounce(self, candidate: Status, t: float) -> Status:
        """Hold a candidate status until it has stood for long enough.

        Timed in seconds rather than counted in frames: inference rate moves
        with CPU load, so a frame count is a different amount of real time
        from one minute to the next. The wait is per status and deliberately
        asymmetric — going green is a safety claim and should be slow, going
        red is an alarm and should be quick.
        """
        self.raw = candidate
        if candidate != self.candidate:
            self.candidate = candidate
            self.candidate_since = t
        # Deliberately not an elif: a zero wait should apply on the same
        # update, not cost an extra tick that no setting asked for.
        if candidate != self.status and (t - self.candidate_since) >= self._wait(candidate):
            log.info("status %s -> %s", self.status.value, candidate.value)
            self.status = candidate
        return self.status

    def _wait(self, status: Status) -> float:
        return self.confirm.get(status, 0.5)

    def candidate_age(self, t: float | None = None) -> float:
        """How long the pending status has stood — the debounce, made visible."""
        return (now() if t is None else t) - self.candidate_since

    def confirm_wait(self) -> float:
        """Seconds the pending status still needs before the lamp follows it."""
        return self._wait(self.candidate)

    @property
    def watching(self) -> bool:
        """Is there someone to check right now?"""
        return self._subject is None or self._subject.present

    def missing(self) -> list[str]:
        """Required items the subject is short of, with the count when it matters."""
        if not self.watching:
            return []
        out = []
        for c in self.classes:
            if not c.required or c.forbidden or not c.shortfall:
                continue
            # "Gloves (1 of 2)" tells an operator far more than "Gloves".
            out.append(f"{c.label} ({c.count} of {c.need})" if c.need > 1 else c.label)
        return out

    def banned(self) -> list[str]:
        """Items that must not appear, but are being detected right now."""
        if not self.watching:
            return []
        return [c.label for c in self.classes if c.required and c.forbidden and c.present]

    def faults(self) -> list[str]:
        """Everything currently keeping the station out of compliance."""
        return self.missing() + self.banned()

    def unavailable(self) -> list[str]:
        """Required items the loaded model has no class for."""
        return [c.label for c in self.classes if c.required and not c.available]

    def degrade(self, t: float | None = None) -> Status:
        """Force DEGRADED — used when no camera is delivering frames."""
        return self._debounce(Status.DEGRADED, now() if t is None else t)


# --------------------------------------------------------------------------
# Modbus output
# --------------------------------------------------------------------------

# Which lamps are energised in each status.
LAMPS: dict[Status, tuple[str, ...]] = {
    Status.OK: ("green",),
    Status.VIOLATION: ("red",),
    # Standby is dark: nobody is there to read the lamp, and an unlit tower
    # cannot be confused with a compliance verdict. Amber stays reserved for
    # "the station cannot judge", which is a fault and needs to look like one.
    Status.STANDBY: (),
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
            self._retry_at = now() + self.cfg.reconnect_sec
        else:
            log.info("tower connected (%s)", self.cfg.transport)
            self._blank()
            self._state = dict.fromkeys(self.cfg.coils)  # None = force a resync
        return self.connected

    def _blank(self, why: str = "connect") -> int:
        """Drive every channel on the board low. Returns how many were written.

        Every channel, not only the ones this station maps: the relay may be
        holding coils from a previous run that crashed or was killed, or from
        another tool entirely. Taking the board to a known state costs one
        pass and removes a class of "why is that lamp still on" that no amount
        of careful writing afterwards would explain.

        Written straight to the client rather than through ``write()``, which
        skips coils it believes are already low — a belief that a crash, a
        failed write or another writer on the bus is exactly what invalidates.
        Caller holds the lock, or is the only thread that can be running.
        """
        done = 0
        for coil in range(self.cfg.channels):
            try:
                self._client.write_coil(coil, False, **{self._kw: self.cfg.unit})
                done += 1
            except Exception as exc:  # noqa: BLE001 — best effort, never fatal
                # One failure means the bus is down; the rest would only add
                # a timeout each. Say so — a coil left live is worth knowing.
                log.warning(
                    "tower: could not blank coil %d at %s (%d of %d cleared): %s",
                    coil, why, done, self.cfg.channels, exc,
                )
                break
        return done

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
                self._retry_at = now() + self.cfg.reconnect_sec
                return False
            return True

    def close(self) -> None:
        """Leave the board dark, then drop the connection.

        Blanks directly rather than through ``write()``: write() takes the
        same non-reentrant lock this method holds, so calling it here hung
        shutdown forever — and with it went the one chance to put the lamp
        out. It also clears every channel rather than just the mapped ones,
        for the same reason connect does.
        """
        with self._lock:
            if self._client is not None:
                with contextlib.suppress(Exception):  # nothing left to salvage
                    self._blank("shutdown")
                with contextlib.suppress(Exception):
                    self._client.close()
            self._client = None
            self.connected = False
            self._state = dict.fromkeys(self.cfg.coils, False)


class NullTower:
    """Stand-in when the tower is disabled, so the pipeline stays branch-free."""

    connected = False

    def connect(self) -> bool:  # interface parity — nothing to take low
        return False

    def apply(self, status: Status) -> bool:  # noqa: ARG002 — interface parity
        return False

    def close(self) -> None:
        pass


def make_tower(cfg: TowerCfg) -> TowerLight | NullTower:
    return TowerLight(cfg) if cfg.enabled else NullTower()
