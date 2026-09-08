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
    occluded: float = 1.5    # ...and how long a PARTLY visible set keeps full credit
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
                occluded=(
                    c.occluded_ms
                    if c.occluded_ms is not None
                    else (c.hold_ms if c.hold_ms is not None else cfg.hold_ms)
                ) / 1000.0,
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

            # Two different windows, because they answer two different
            # questions. Seeing NOTHING is a dropout: `hold` bridges it, and
            # it stays short — an item that has vanished entirely may well be
            # off the worker. Seeing SOME but not all is an occlusion: one
            # glove behind the body while the other is plainly on the hand,
            # which is evidence the pair is still worn, so `occluded` may be
            # much longer without ever crediting a worker who shows nothing.
            # That split is what makes a long tolerance safe to configure;
            # one window for both would buy the occlusion by also letting a
            # bare-handed worker coast for the same span.
            window = state.hold if seen == 0 else state.occluded
            if seen >= state.count or (t - state.counted_at) > window:
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

# Which lamps are energised in each status. OK is conditional — see
# lamps_for(), which is what actually decides.
LAMPS: dict[Status, tuple[str, ...]] = {
    Status.OK: ("green",),
    Status.VIOLATION: ("red",),
    # Standby is dark: nobody is there to read the lamp, and an unlit tower
    # cannot be confused with a compliance verdict.
    Status.STANDBY: (),
    Status.DEGRADED: ("amber",),
}


def lamps_for(status: Status, grinder_on: bool, estop_hit: bool = False) -> tuple[str, ...]:
    """Which lamps are lit, given compliance *and* the state of the machine.

    The tower reports the machine, not only the verdict on the worker:

      red     the e-stop is hit, or a violation
      green   compliant AND the grinder is actually running
      amber   compliant but the grinder is idle — nothing is wrong, the
              operator just has not pressed the button yet; also DEGRADED
      dark    nobody in the cell

    The e-stop outranks everything, compliance and an empty cell included.
    Somebody hit it, the machine is down, and that is worth a red lamp
    whether or not anyone is standing in front of the cameras — a dark
    tower over an e-stopped cell says nothing at all about why the
    machine will not start.

    Splitting OK across green and amber costs the one thing amber used to
    say on its own. It now covers both "cannot judge" (a fault) and "all
    good, press the button" (not a fault), which a lamp alone can no
    longer distinguish — the UI still names which, and the alternative
    was a green lamp on a machine that is not running.
    """
    if estop_hit:
        return ("red",)
    if status is Status.OK:
        return ("green",) if grinder_on else ("amber",)
    return LAMPS[status]

# Coils apply() is allowed to touch — every lamp plus the buzzer. Any other
# coil in tower.coils (belt_grinder, say) belongs to a different write path
# and must not be forced low here every cycle just for existing in the dict.
_LAMP_COILS = {lamp for lamps in LAMPS.values() for lamp in lamps} | {"buzzer"}


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
        # Belt grinder latch: whether a press has armed it, and what the
        # button read last cycle so a *new* press can be told from one
        # still being held. Both start in the state that demands a fresh,
        # fully observed release-then-press before anything can run.
        self._grinder_latched = False
        self._button_was_pressed = True
        # Buzzer: sounded as a fixed-length pulse when a violation cuts the
        # grinder, not held for as long as the violation stands. It marks
        # the moment the machine was taken away, which is the thing an
        # operator needs to connect to what they just did.
        self._buzz_until = 0.0
        # The last e-stop reading, for the lamp. None until one has been
        # taken (or after one fails): unknown is not the same as hit, and
        # claiming an emergency nobody observed would be its own lie.
        self._estop_ok: bool | None = None

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
            self._disarm_grinder()
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
        """Drive the lamps and buzzer. Returns True if the bus was written.

        Reads this object's own grinder latch rather than taking it as an
        argument, which means **call update_belt_grinder() first** in a
        cycle: run the other way round, the lamp shows last cycle's
        machine state. Pipeline._cycle() and _go_offline() both do.
        """
        wanted = {k: False for k in self.cfg.coils if k in _LAMP_COILS}
        # A station with no grinder wired has nothing for green to wait on,
        # so it keeps the old meaning: compliant is green, full stop.
        running = self._grinder_latched or "belt_grinder" not in self.cfg.coils
        # `is False` on purpose: None means no reading has been taken, which
        # is not an e-stop.
        for lamp in lamps_for(status, running, estop_hit=self._estop_ok is False):
            if lamp in wanted:
                wanted[lamp] = True
        if "buzzer" in wanted:
            # A window, not a level: set when a violation cut the grinder,
            # and it expires on its own however long the violation lasts.
            wanted["buzzer"] = now() < self._buzz_until
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

    @property
    def estop_hit(self) -> bool:
        """Is the e-stop currently observed as pressed?

        False also covers "no reading yet" and "the read failed" — an
        unknown e-stop is not an asserted one, the same rule the lamp
        follows. Read by the pipeline so the screen can say so too.
        """
        return self._estop_ok is False

    # -- input -------------------------------------------------------------
    def _read_input(self, name: str) -> bool | None:
        """One named discrete input, or None if it could not be read.

        Same lock and connect() gate as write(): this shares the bus with
        the coil writes and must not run concurrently with them either.
        """
        with self._lock:
            if not self.connect():
                return None
            try:
                rsp = self._client.read_discrete_inputs(
                    self.cfg.inputs[name], count=1, **{self._kw: self.cfg.unit}
                )
                if rsp is None or rsp.isError():
                    raise OSError(str(rsp))
                return bool(rsp.bits[0])
            except Exception as exc:  # noqa: BLE001 — the line is allowed to be down
                log.warning("tower: could not read input %s: %s", name, exc)
                self.connected = False
                self._retry_at = now() + self.cfg.reconnect_sec
                return None

    def _disarm_grinder(self) -> None:
        """Forget the latch and demand a fresh press before running again.

        Used wherever the coil is taken low outside this method's own
        control — connect, close — because the motor is physically off at
        that point and must not come back without somebody deciding it
        should. Assuming the button is currently held is the conservative
        half of that: it means only a release-then-press this method has
        actually *seen* can re-arm the latch.
        """
        self._grinder_latched = False
        self._button_was_pressed = True
        self._buzz_until = 0.0  # a blanked board is not mid-annunciation
        self._estop_ok = None

    def _drop_grinder(self, why: str) -> bool:
        """Clear the latch and drive the coil low, logging the transition."""
        if self._grinder_latched:
            log.info("belt grinder off: %s", why)
        self._grinder_latched = False
        return self.write({"belt_grinder": False})

    def update_belt_grinder(self, status: Status) -> bool:
        """Drive the belt grinder relay from the e-stop and push-button
        inputs plus this cycle's compliance status.

        Both switches are wired **active** (normally closed): an idle input
        reads True, and pressing the switch takes it to False. So False is
        "e-stop hit" on one and "button pressed" on the other — which is
        why the run condition wants estop True and push_button False, and
        why that is not the typo it looks like.

        The push button is momentary, so this is a **latch** — the seal-in
        of a standard motor starter — rather than the hold-to-run a plain
        AND chain would give:

          start  a press (a release-then-press this method actually saw)
                 while the e-stop reads True and status is Status.OK
          run    until something drops it; releasing the button does not
          drop   the e-stop going False, status leaving Status.OK, either
                 input failing to read, or the board being taken low by a
                 connect or a close

        Dropping the latch is the point of it. Nothing here restarts on
        its own: when the e-stop is released, or PPE compliance comes
        back, or the bus recovers, the coil stays low until an operator
        presses the button again. That is the restart interlock — a fault
        that clears must not spin the motor back up under someone's
        hands — and it is why the button state is tracked as an edge.
        Holding the button down through a fault therefore starts nothing
        when the fault clears; the operator has to let go first.

        No-op — nothing read, nothing written — on a station that has not
        wired this up: belt_grinder must be in tower.coils and both estop
        and push_button must be in tower.inputs. Config.validate() already
        refuses a half-configured version of this, so in practice this is
        either fully wired or entirely absent.
        """
        if "belt_grinder" not in self.cfg.coils:
            return False
        if not {"estop", "push_button"} <= set(self.cfg.inputs):
            return False

        estop = self._read_input("estop")
        push_button = self._read_input("push_button")
        if estop is None or push_button is None:
            # The button's position is unknown, so the next press cannot be
            # told from a hold that spanned the outage. Demand one this
            # method has seen from both sides rather than guessing.
            self._button_was_pressed = True
            self._estop_ok = None  # unknown, so the lamp falls back to status
            return self._drop_grinder("input read failed")

        # Track the button every cycle, faults included: a release *during*
        # a fault is what makes the operator's next press a real edge.
        pressed = not push_button
        was_pressed, self._button_was_pressed = self._button_was_pressed, pressed
        self._estop_ok = bool(estop)

        if not estop:
            return self._drop_grinder("e-stop hit")
        if status is not Status.OK:
            # Only a violation sounds the buzzer, and only if it actually
            # took a running machine away: STANDBY (the operator stepped
            # out) and DEGRADED (the station cannot judge) stop the
            # grinder just as hard, but neither is the worker doing
            # something the buzzer is there to call out.
            if (
                self._grinder_latched
                and status is Status.VIOLATION
                and self.cfg.buzzer_on_violation
            ):
                self._buzz_until = now() + self.cfg.buzzer_sec
                log.info("buzzer on for %.1fs: a violation stopped the grinder",
                         self.cfg.buzzer_sec)
            return self._drop_grinder(f"status is {status.value}")

        if pressed and not was_pressed:
            log.info("belt grinder on: button pressed")
            self._grinder_latched = True
        return self.write({"belt_grinder": self._grinder_latched})

    def close(self) -> None:
        """Leave the board dark, then drop the connection.

        Blanks directly rather than through ``write()``: write() takes the
        same non-reentrant lock this method holds, so calling it here hung
        shutdown forever — and with it went the one chance to put the lamp
        out. It also clears every channel rather than just the mapped ones,
        for the same reason connect does.
        """
        with self._lock:
            # Before the blanking, not after: _blank() runs under suppress
            # because a dead bus must not stop a shutdown, and the latch
            # must be forgotten on the path where that blanking throws too.
            self._disarm_grinder()
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
    estop_hit = False   # no board, no e-stop to read

    def connect(self) -> bool:  # interface parity — nothing to take low
        return False

    def apply(self, status: Status) -> bool:  # noqa: ARG002 — interface parity
        return False

    def update_belt_grinder(self, status: Status) -> bool:  # noqa: ARG002
        return False

    def close(self) -> None:
        pass


def make_tower(cfg: TowerCfg) -> TowerLight | NullTower:
    return TowerLight(cfg) if cfg.enabled else NullTower()
