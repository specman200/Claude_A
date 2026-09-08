"""The state machine that decides what colour the tower light shows."""

import pytest

from ppe.config import ClassCfg, PPECfg, TowerCfg
from ppe.detector import Detection
from ppe.tower import LAMPS, ComplianceMonitor, Status, TowerLight, make_tower


def det(name, conf=0.9):
    return Detection(name, conf, (0.0, 0.0, 10.0, 10.0))


def confirm_all(seconds):
    """Same confirm wait for every status, so tests can pin the timing."""
    return dict.fromkeys(("ok", "violation", "standby", "degraded"), seconds)


def monitor(hold_ms=1000, confirm=0.0, optional=(), missing=(), forbidden=()):
    cfg = PPECfg(
        classes=[ClassCfg("helmet"), ClassCfg("vest")]
        + [ClassCfg(n, required=False) for n in optional]
        + [ClassCfg(n, expect="absent") for n in forbidden],
        hold_ms=hold_ms,
        confirm_sec=confirm_all(confirm),
    )
    return ComplianceMonitor(cfg, list(missing))


def test_all_required_present_is_ok():
    m = monitor()
    assert m.update([[det("helmet"), det("vest")]], t=1.0) is Status.OK
    assert m.missing() == []


def test_one_required_missing_is_a_violation():
    m = monitor()
    assert m.update([[det("helmet")]], t=1.0) is Status.VIOLATION
    assert m.missing() == ["Vest"]


def test_optional_classes_do_not_gate_the_light():
    m = monitor(optional=["mask"])
    assert m.update([[det("helmet"), det("vest")]], t=1.0) is Status.OK


def test_a_class_the_model_lacks_degrades_instead_of_passing():
    m = monitor(missing=["vest"])
    assert m.update([[det("helmet")]], t=1.0) is Status.DEGRADED
    assert [c.available for c in m.classes] == [True, False]
    assert m.unavailable() == ["Vest"]


def test_an_unavailable_optional_class_does_not_degrade_the_station():
    m = monitor(optional=["mask"], missing=["mask"])
    assert m.update([[det("helmet"), det("vest")]], t=1.0) is Status.OK
    assert m.unavailable() == []


def test_hold_window_bridges_a_dropped_frame():
    m = monitor(hold_ms=1000)
    m.update([[det("helmet"), det("vest")]], t=10.0)
    assert m.update([[det("helmet")]], t=10.5) is Status.OK      # vest still held
    assert m.update([[det("helmet")]], t=11.5) is Status.VIOLATION  # hold expired


def test_a_status_must_stand_for_its_confirm_time_before_the_lamp_follows():
    m = monitor(confirm=1.0)
    # The clock starts when the candidate first appears, not when it is asserted.
    assert m.update([[det("helmet"), det("vest")]], t=1.0) is Status.DEGRADED
    assert m.update([[det("helmet"), det("vest")]], t=1.5) is Status.DEGRADED
    assert m.update([[det("helmet"), det("vest")]], t=2.0) is Status.OK

    # A single bad frame restarts the wait rather than flipping the lamp.
    assert m.update([[det("helmet")]], t=2.1) is Status.OK
    assert m.update([[det("helmet"), det("vest")]], t=2.2) is Status.OK


def test_confirm_time_is_wall_clock_not_frames():
    """The same number of updates flips or does not flip purely on elapsed
    time — a frame count would behave differently under CPU load."""
    fast = monitor(confirm=1.0)
    for t in (0.0, 0.01, 0.02, 0.03, 0.04):  # 5 updates, 40 ms of real time
        fast.update([[det("helmet"), det("vest")]], t=t)
    assert fast.status is Status.DEGRADED, "40 ms of agreement must not confirm a 1 s wait"

    slow = monitor(confirm=1.0)
    for t in (0.0, 1.5):  # 2 updates, 1.5 s of real time
        slow.update([[det("helmet"), det("vest")]], t=t)
    assert slow.status is Status.OK


def test_going_red_is_quicker_than_going_green():
    """Fail-safe asymmetry: an alarm should beat a safety claim to the lamp."""
    cfg = PPECfg(
        classes=[ClassCfg("helmet"), ClassCfg("vest")],
        hold_ms=0,
        confirm_sec={"ok": 1.0, "violation": 0.4, "standby": 1.0, "degraded": 0.0},
    )
    m = ComplianceMonitor(cfg)
    m.update([[det("helmet"), det("vest")]], t=0.0)   # candidate OK starts here
    assert m.update([[det("helmet"), det("vest")]], t=0.5) is Status.DEGRADED  # 0.5 < 1.0
    assert m.update([[det("helmet"), det("vest")]], t=1.1) is Status.OK

    # Now lose the vest: red lands in 0.4 s, less than the 1.0 s green took.
    m.update([[det("helmet")]], t=1.2)               # candidate VIOLATION starts
    assert m.status is Status.OK
    # 0.5 s later, comfortably past the 0.4 s violation wait but still under
    # the 1.0 s it took to go green in the first place.
    assert m.update([[det("helmet")]], t=1.7) is Status.VIOLATION


def test_a_sustained_change_does_flip_after_confirmation():
    m = monitor(hold_ms=0, confirm=0.2)
    for t in (1.0, 1.2, 1.4):
        m.update([[det("helmet"), det("vest")]], t=t)
    assert m.status is Status.OK
    m.update([[det("helmet")]], t=1.5)          # candidate starts here
    assert m.update([[det("helmet")]], t=1.8) is Status.VIOLATION


def test_confidence_keeps_the_best_view_across_cameras():
    m = monitor()
    m.update([[det("helmet", 0.4), det("helmet", 0.8), det("vest", 0.7)]], t=1.0)
    assert m.classes[0].conf == pytest.approx(0.8)


def test_unknown_detections_are_ignored():
    m = monitor()
    assert m.update([[det("forklift"), det("helmet"), det("vest")]], t=1.0) is Status.OK


def test_degrade_forces_the_amber_state():
    m = monitor(confirm=1)
    m.update([[det("helmet"), det("vest")]], t=1.0)
    assert m.degrade() is Status.DEGRADED


def test_every_status_maps_to_a_lamp_pattern():
    assert set(LAMPS) == set(Status)
    for status, lamps in LAMPS.items():
        # Standby is deliberately dark; every other status lights exactly one.
        assert len(lamps) == (0 if status is Status.STANDBY else 1)


@pytest.mark.parametrize(
    "status,grinder_on,expected",
    [
        (Status.OK, True, ("green",)),    # compliant and running
        (Status.OK, False, ("amber",)),   # compliant but nobody pressed start
        (Status.VIOLATION, True, ("red",)),
        (Status.VIOLATION, False, ("red",)),
        (Status.STANDBY, False, ()),
        (Status.DEGRADED, False, ("amber",)),
    ],
)
def test_green_means_running_not_merely_compliant(status, grinder_on, expected):
    from ppe.tower import lamps_for

    assert lamps_for(status, grinder_on) == expected


@pytest.mark.parametrize("status", list(Status))
@pytest.mark.parametrize("grinder_on", [True, False])
def test_a_hit_estop_is_red_whatever_else_is_true(status, grinder_on):
    """It outranks everything, an empty cell included: a dark tower over an
    e-stopped machine says nothing about why it will not start."""
    from ppe.tower import lamps_for

    assert lamps_for(status, grinder_on, estop_hit=True) == ("red",)


# -- Modbus output ---------------------------------------------------------


class FakeClient:
    """Stands in for pymodbus: records coil writes, can be told to fail."""

    def __init__(self):
        self.writes = []
        self.reads = []          # addresses read, in call order
        self.fail = False
        self.fail_read = False
        self.closed = False
        self.inputs: dict[int, bool] = {}  # address -> current bit

    def connect(self):
        return True

    def write_coil(self, address, value, slave=None):
        if self.fail:
            raise OSError("bus down")
        self.writes.append((address, value))
        return type("Rsp", (), {"isError": lambda _self: False})()

    def read_discrete_inputs(self, address, count=1, slave=None):
        if self.fail_read:
            raise OSError("bus down")
        self.reads.append(address)
        bits = [self.inputs.get(address + i, False) for i in range(count)]
        return type("Rsp", (), {"isError": lambda _self: False, "bits": bits})()

    def close(self):
        self.closed = True


def tower_with_fake():
    tower = TowerLight(TowerCfg(coils={"green": 0, "amber": 1, "red": 2, "buzzer": 3}))
    fake = FakeClient()
    tower._make_client = lambda: fake
    return tower, fake


def tower_with_grinder():
    """A tower wired for the belt grinder interlock — coil 4, inputs 0 and 1."""
    tower = TowerLight(TowerCfg(
        coils={"green": 0, "amber": 1, "red": 2, "buzzer": 3, "belt_grinder": 4},
        inputs={"estop": 0, "push_button": 1},
    ))
    fake = FakeClient()
    tower._make_client = lambda: fake
    return tower, fake


def test_every_channel_is_blanked_before_the_station_takes_control():
    """A previous run that crashed may have left coils energised — including
    channels this station does not manage."""
    tower, fake = tower_with_fake()
    tower.cfg.channels = 8
    tower.connect()
    assert [c for c, _ in fake.writes] == list(range(8))
    assert all(v is False for _, v in fake.writes)


def test_every_channel_is_blanked_on_shutdown():
    """Closing the app must leave the board dark, mapped coils or not.

    The lamp outlives the process otherwise: a relay holds its last commanded
    state, so an operator closing the station is left looking at a light that
    means nothing.
    """
    tower, fake = tower_with_fake()
    tower.cfg.channels = 8
    tower.connect()
    tower.apply(Status.VIOLATION)
    fake.writes.clear()

    tower.close()
    assert [c for c, _ in fake.writes] == list(range(8))
    assert all(v is False for _, v in fake.writes)
    assert fake.closed


def test_close_returns_instead_of_deadlocking():
    """close() took the lock and then called write(), which takes it again.

    threading.Lock is not reentrant, so shutdown hung forever holding the one
    chance to put the lamp out — and the pipeline thread never got to flush
    telemetry either. Run it on a thread so a regression fails rather than
    hanging the suite.
    """
    import threading

    tower, _ = tower_with_fake()
    tower.cfg.channels = 8
    tower.connect()

    done = threading.Event()
    threading.Thread(target=lambda: (tower.close(), done.set()), daemon=True).start()
    assert done.wait(5), "close() deadlocked"


def test_shutdown_clears_coils_it_believes_are_already_low():
    """Shutdown must not trust its own bookkeeping.

    write() skips a coil whose recorded state already matches — but a crash,
    a failed write or another writer on the bus is exactly what makes that
    record wrong, and shutdown is when being wrong leaves a lamp on.
    """
    tower, fake = tower_with_fake()
    tower.cfg.channels = 4
    tower.connect()
    tower._state = dict.fromkeys(tower.cfg.coils, False)  # "everything is off"
    fake.writes.clear()

    tower.close()
    assert [c for c, _ in fake.writes] == [0, 1, 2, 3], "believed-off coils were skipped"


def test_a_failed_blank_at_shutdown_is_survived():
    """A bus that dies mid-shutdown must not take the app down with it."""
    tower, fake = tower_with_fake()
    tower.connect()
    fake.fail = True
    tower.close()          # must not raise
    assert tower.connected is False


def test_apply_energises_only_the_status_lamp():
    tower, fake = tower_with_fake()
    tower.connect()
    fake.writes.clear()          # discard the blanking pass
    assert tower.apply(Status.OK)
    assert dict(fake.writes) == {0: True, 1: False, 2: False, 3: False}


def test_unchanged_status_does_not_touch_the_bus():
    tower, fake = tower_with_fake()
    tower.apply(Status.OK)
    fake.writes.clear()
    assert tower.apply(Status.OK) is False
    assert fake.writes == []


def test_a_status_change_writes_only_the_changed_coils():
    tower, fake = tower_with_fake()
    tower.apply(Status.OK)
    fake.writes.clear()
    tower.apply(Status.VIOLATION)
    assert dict(fake.writes) == {0: False, 2: True}


def test_a_violation_alone_no_longer_sounds_the_buzzer():
    """It used to be a level output held for as long as the violation
    stood. It is now a pulse tied to the grinder being cut, so a station
    with no grinder at all never sounds it."""
    tower, fake = tower_with_fake()
    tower.cfg.buzzer_on_violation = True
    tower.apply(Status.VIOLATION)
    assert dict(fake.writes).get(3) is not True


def test_a_bus_failure_is_survived_and_resynced():
    tower, fake = tower_with_fake()
    tower.apply(Status.OK)
    fake.fail = True
    assert tower.apply(Status.VIOLATION) is False
    assert tower.connected is False

    fake.fail = False
    tower._retry_at = 0.0
    assert tower.apply(Status.VIOLATION)
    # After a reconnect the board is blanked and every managed coil rewritten,
    # rather than trusting a cache that a failed write may have poisoned.
    assert set(dict(fake.writes)) >= {0, 1, 2, 3}


def test_a_disabled_tower_is_a_no_op():
    tower = make_tower(TowerCfg(enabled=False))
    assert tower.apply(Status.VIOLATION) is False
    tower.close()


def test_apply_never_touches_a_coil_it_does_not_manage():
    """belt_grinder lives in the same coils dict as the lamps, but apply()
    must never force it low just for being there — that would fight
    update_belt_grinder()'s own write every single cycle."""
    tower, fake = tower_with_grinder()
    tower.connect()
    fake.writes.clear()  # discard the connect-time blanking pass
    tower.apply(Status.OK)
    assert 4 not in dict(fake.writes)


# -- belt grinder interlock -------------------------------------------------
# Digital Input 1 (estop) and Digital Input 2 (push_button) gate Digital
# Output 5 (belt_grinder). Both switches are wired ACTIVE (normally
# closed), so an idle input reads True and pressing takes it to False —
# hence estop True means "not hit" and push_button False means "held
# down".
#
# The button is momentary, so the output latches: a press while the e-stop
# is clear and PPE is compliant starts the motor and it keeps running once
# the button is let go. What these mostly pin is the *other* half of a
# latch — that nothing restarts on its own. Every fault clears the latch,
# so a released e-stop, restored compliance or a recovered bus leaves the
# coil low until somebody presses the button again.


def press(tower, fake, status=Status.OK):
    """A real button push: one cycle released, then one cycle held.

    Two cycles because the latch arms on the edge, not the level — a
    press that was already held when the last fault happened is not a
    new press, and must not start anything.
    """
    fake.inputs[1] = True   # released — idle is True on an active input
    tower.update_belt_grinder(status)
    fake.inputs[1] = False  # pressed
    return tower.update_belt_grinder(status)


def test_belt_grinder_is_a_noop_without_the_coil_configured():
    tower, fake = tower_with_fake()  # no belt_grinder in coils, no inputs
    assert tower.update_belt_grinder(Status.OK) is False
    assert fake.reads == []
    assert fake.writes == []


def test_belt_grinder_is_a_noop_without_both_inputs_configured():
    tower = TowerLight(TowerCfg(
        coils={"green": 0, "amber": 1, "red": 2, "buzzer": 3, "belt_grinder": 4},
        inputs={"estop": 0},  # push_button missing
    ))
    fake = FakeClient()
    tower._make_client = lambda: fake
    assert tower.update_belt_grinder(Status.OK) is False
    assert fake.reads == []
    assert fake.writes == []


def test_a_press_starts_the_grinder_and_it_keeps_running_once_released():
    """The whole point of the latch: a momentary button cannot be held for
    the length of a job, so letting go must not stop the motor."""
    tower, fake = tower_with_grinder()
    tower.connect()
    fake.inputs = {0: True, 1: True}  # e-stop clear, button idle
    fake.writes.clear()

    assert press(tower, fake) is True
    assert dict(fake.writes) == {4: True}

    fake.writes.clear()
    fake.inputs[1] = True  # let go
    assert tower.update_belt_grinder(Status.OK) is False  # still on, nothing to write
    assert fake.writes == []


def test_nothing_starts_without_a_press():
    """An idle button with everything else healthy is not a start."""
    tower, fake = tower_with_grinder()
    tower.connect()
    fake.inputs = {0: True, 1: True}
    fake.writes.clear()
    tower.update_belt_grinder(Status.OK)
    assert dict(fake.writes).get(4) is not True


def test_the_estop_stops_it_and_releasing_the_estop_does_not_restart_it():
    """The restart interlock. A cleared fault must not spin the motor back
    up under someone's hands — that is what a latch is for."""
    tower, fake = tower_with_grinder()
    tower.connect()
    fake.inputs = {0: True, 1: True}
    press(tower, fake)
    fake.writes.clear()

    fake.inputs[0] = False  # e-stop hit — active wiring, so False
    assert tower.update_belt_grinder(Status.OK) is True
    assert dict(fake.writes) == {4: False}

    fake.writes.clear()
    fake.inputs[0] = True  # e-stop released
    assert tower.update_belt_grinder(Status.OK) is False
    assert fake.writes == [], "released e-stop restarted the motor on its own"

    assert press(tower, fake) is True  # a fresh press does start it again
    assert dict(fake.writes) == {4: True}


def test_losing_compliance_stops_it_and_regaining_it_does_not_restart_it():
    tower, fake = tower_with_grinder()
    tower.connect()
    fake.inputs = {0: True, 1: True}
    press(tower, fake)
    fake.writes.clear()

    assert tower.update_belt_grinder(Status.VIOLATION) is True
    assert dict(fake.writes) == {4: False}

    fake.writes.clear()
    assert tower.update_belt_grinder(Status.OK) is False
    assert fake.writes == [], "restored compliance restarted the motor on its own"


def test_standby_stops_it_too():
    """Nobody in view is not compliance — Status.OK is the only run state."""
    tower, fake = tower_with_grinder()
    tower.connect()
    fake.inputs = {0: True, 1: True}
    press(tower, fake)
    fake.writes.clear()
    assert tower.update_belt_grinder(Status.STANDBY) is True
    assert dict(fake.writes) == {4: False}


def test_a_button_held_through_a_fault_does_not_restart_on_its_own():
    """The defeat case, and the reason the latch arms on an edge rather
    than a level: a button taped or wedged down must not turn a cleared
    fault into a start. The operator has to let go first."""
    tower, fake = tower_with_grinder()
    tower.connect()
    fake.inputs = {0: True, 1: True}
    press(tower, fake)

    fake.inputs[0] = False       # e-stop hit while the button stays held
    tower.update_belt_grinder(Status.OK)
    fake.inputs[0] = True        # fault clears, button STILL held
    fake.writes.clear()
    assert tower.update_belt_grinder(Status.OK) is False
    assert fake.writes == [], "a held button restarted the motor"

    # Letting go and pressing again is a real start.
    assert press(tower, fake) is True
    assert dict(fake.writes) == {4: True}


def test_a_press_while_non_compliant_is_not_banked_for_later():
    """Pressing during a violation must not arm anything that fires the
    moment PPE comes good — the press has to happen while it is safe."""
    tower, fake = tower_with_grinder()
    tower.connect()
    fake.inputs = {0: True, 1: True}
    press(tower, fake, status=Status.VIOLATION)  # pressed and still held
    fake.writes.clear()

    assert tower.update_belt_grinder(Status.OK) is False
    assert fake.writes == [], "a press made during a violation started it late"


def test_belt_grinder_reads_both_inputs_by_their_configured_address():
    tower, fake = tower_with_grinder()
    tower.update_belt_grinder(Status.OK)
    assert fake.reads == [0, 1]


def test_a_failed_input_read_stops_it_and_needs_a_fresh_press():
    """A read failure marks the bus down the same way a write failure does
    (test_a_bus_failure_is_survived_and_resynced), so the fallback write to
    False goes out over the same bus that just failed, and fails with it
    too — nothing reaches the coil while the bus is down.

    The latch is still dropped, though, and dropping it is what matters:
    when the bus comes back the motor stays off, because a cycle where the
    button could not be read cannot be told from one where it was held.
    """
    tower, fake = tower_with_grinder()
    tower.connect()
    fake.inputs = {0: True, 1: True}
    press(tower, fake)
    fake.writes.clear()

    fake.fail_read = True
    assert tower.update_belt_grinder(Status.OK) is False  # bus down — nothing sent
    assert fake.writes == []
    assert tower.connected is False

    fake.fail_read = False
    tower._retry_at = 0.0  # skip the reconnect backoff for the test
    tower.update_belt_grinder(Status.OK)
    assert dict(fake.writes).get(4) is not True, "recovered bus restarted the motor"

    fake.writes.clear()
    assert press(tower, fake) is True
    assert dict(fake.writes) == {4: True}


def test_reconnecting_forgets_the_latch():
    """connect() blanks the whole board, so the motor is physically off at
    that point — the latch must not claim otherwise and rewrite it high."""
    tower, fake = tower_with_grinder()
    tower.connect()
    fake.inputs = {0: True, 1: True}
    press(tower, fake)
    assert tower._grinder_latched is True

    tower.connected = False  # as a failed write or read would leave it
    tower._retry_at = 0.0
    tower.connect()
    assert tower._grinder_latched is False
    fake.writes.clear()
    # A reconnect resyncs every coil from an unknown state, so a write of
    # False is expected here — a write of True is the regression.
    tower.update_belt_grinder(Status.OK)
    assert dict(fake.writes).get(4) is not True


def test_closing_forgets_the_latch_even_if_blanking_throws():
    """close() blanks under suppress because a dead bus must not stop a
    shutdown; the latch has to be dropped on that path too."""
    tower, fake = tower_with_grinder()
    tower.connect()
    fake.inputs = {0: True, 1: True}
    press(tower, fake)
    fake.fail = True  # blanking will raise all the way through close()

    tower.close()
    assert tower._grinder_latched is False


def test_belt_grinder_write_is_skipped_once_already_off():
    tower, fake = tower_with_grinder()
    fake.inputs = {0: False, 1: True}  # e-stop hit from the start
    tower.update_belt_grinder(Status.OK)  # first call always writes once, to sync
    fake.writes.clear()
    assert tower.update_belt_grinder(Status.OK) is False  # unchanged — no bus write
    assert fake.writes == []


# -- forbidden classes -----------------------------------------------------
# Some models carry violation classes (e.g. "Wrong Sleeve"): detecting one IS
# the fault. Treating those like ordinary PPE would turn the light green on
# exactly the condition it exists to catch.


def test_a_forbidden_class_passes_while_it_is_absent():
    m = monitor(forbidden=["wrong_sleeve"])
    assert m.update([[det("helmet"), det("vest")]], t=1.0) is Status.OK
    assert m.banned() == []
    assert m.faults() == []


def test_detecting_a_forbidden_class_is_a_violation():
    m = monitor(forbidden=["wrong_sleeve"])
    status = m.update([[det("helmet"), det("vest"), det("wrong_sleeve")]], t=1.0)
    assert status is Status.VIOLATION
    assert m.banned() == ["Wrong Sleeve"]
    assert m.missing() == []


def test_a_forbidden_class_is_never_reported_as_missing():
    """The old logic would have listed it as absent PPE — the opposite fault."""
    m = monitor(forbidden=["wrong_sleeve"])
    m.update([[det("helmet"), det("vest")]], t=1.0)
    assert "Wrong Sleeve" not in m.missing()


def test_both_kinds_of_fault_are_reported_together():
    m = monitor(forbidden=["wrong_sleeve"])
    m.update([[det("helmet"), det("wrong_sleeve")]], t=1.0)
    assert m.missing() == ["Vest"]
    assert m.banned() == ["Wrong Sleeve"]
    assert m.faults() == ["Vest", "Wrong Sleeve"]


def test_the_hold_window_applies_to_forbidden_classes_too():
    """A violation must not clear the instant the model blinks."""
    m = monitor(hold_ms=1000, forbidden=["wrong_sleeve"])
    m.update([[det("helmet"), det("vest"), det("wrong_sleeve")]], t=10.0)
    assert m.update([[det("helmet"), det("vest")]], t=10.5) is Status.VIOLATION
    assert m.update([[det("helmet"), det("vest")]], t=11.5) is Status.OK


def test_an_unrequired_forbidden_class_does_not_gate_the_light():
    cfg = PPECfg(
        classes=[ClassCfg("helmet"), ClassCfg("wrong_sleeve", required=False, expect="absent")],
        hold_ms=1000,
        confirm_sec=confirm_all(0.0),
    )
    m = ComplianceMonitor(cfg)
    assert m.update([[det("helmet"), det("wrong_sleeve")]], t=1.0) is Status.OK
    assert m.banned() == []


def test_class_state_reports_compliance_not_mere_presence():
    m = monitor(forbidden=["wrong_sleeve"])
    m.update([[det("helmet"), det("vest")]], t=1.0)
    states = {c.name: c for c in m.classes}
    assert states["helmet"].present and states["helmet"].compliant
    assert not states["wrong_sleeve"].present and states["wrong_sleeve"].compliant
    assert states["wrong_sleeve"].forbidden and not states["helmet"].forbidden


# -- standby ---------------------------------------------------------------
# PPE is only meaningful on a person. With a subject class configured, an empty
# cell must read STANDBY, not "every item missing".


def gated(hold_ms=1000, confirm=0.0, forbidden=()):
    cfg = PPECfg(
        classes=[ClassCfg("helmet"), ClassCfg("vest"), ClassCfg("person", required=False)]
        + [ClassCfg(n, expect="absent") for n in forbidden],
        hold_ms=hold_ms,
        confirm_sec=confirm_all(confirm),
        subject="person",
    )
    return ComplianceMonitor(cfg)


def test_no_person_means_standby_not_a_pile_of_violations():
    m = gated()
    assert m.update([[]], t=1.0) is Status.STANDBY
    assert m.missing() == []
    assert m.banned() == []
    assert not m.watching


def test_a_person_with_full_ppe_passes():
    m = gated()
    assert m.update([[det("person"), det("helmet"), det("vest")]], t=1.0) is Status.OK
    assert m.watching


def test_a_person_missing_ppe_is_a_violation():
    m = gated()
    assert m.update([[det("person"), det("helmet")]], t=1.0) is Status.VIOLATION
    assert m.missing() == ["Vest"]


def test_the_person_leaving_returns_the_station_to_standby():
    m = gated(hold_ms=0)
    m.update([[det("person"), det("helmet")]], t=1.0)
    assert m.status is Status.VIOLATION
    assert m.update([[]], t=2.0) is Status.STANDBY


def test_a_dropped_person_frame_does_not_flicker_into_standby():
    """The subject rides the same hold window as the PPE it gates."""
    m = gated(hold_ms=1000)
    m.update([[det("person"), det("helmet"), det("vest")]], t=10.0)
    assert m.update([[]], t=10.5) is Status.OK       # person still held
    assert m.update([[]], t=11.5) is Status.STANDBY  # hold expired


def test_a_forbidden_item_with_nobody_there_is_not_a_violation():
    m = gated(forbidden=["wrong_sleeve"])
    assert m.update([[det("wrong_sleeve")]], t=1.0) is Status.STANDBY
    assert m.banned() == []


def test_without_a_subject_class_the_station_never_stands_by():
    m = monitor()  # no subject configured
    assert m.update([[]], t=1.0) is Status.VIOLATION
    assert m.watching


def test_standby_is_debounced_like_any_other_transition():
    m = gated(hold_ms=0, confirm=0.5)
    for t in (1.0, 1.5, 2.0):
        m.update([[det("person"), det("helmet"), det("vest")]], t=t)
    assert m.status is Status.OK
    assert m.update([[]], t=2.1) is Status.OK        # 0.1 s is not enough
    assert m.update([[]], t=2.7) is Status.STANDBY   # 0.6 s clears the 0.5 s wait


def test_a_model_missing_a_required_class_still_beats_standby():
    """An unusable model is a fault, and must not hide behind an empty cell."""
    cfg = PPECfg(
        classes=[ClassCfg("helmet"), ClassCfg("person", required=False)],
        hold_ms=1000,
        confirm_sec=confirm_all(0.0),
        subject="person",
    )
    m = ComplianceMonitor(cfg, ["helmet"])
    assert m.update([[]], t=1.0) is Status.DEGRADED


def test_standby_shows_no_lamps():
    """Amber stays reserved for faults; an idle cell leaves the tower dark."""
    assert LAMPS[Status.STANDBY] == ()
    tower, fake = tower_with_fake()
    tower.apply(Status.OK)
    fake.writes.clear()
    tower.apply(Status.STANDBY)
    assert dict(fake.writes) == {0: False}  # green off, nothing else lit


# -- counts ----------------------------------------------------------------
# A worker has two hands and two arms. One glove is not "gloves: present".


def paired(hold_ms=1000, confirm=0.0):
    cfg = PPECfg(
        classes=[ClassCfg("Gloves", count=2), ClassCfg("person", required=False)],
        hold_ms=hold_ms,
        confirm_sec=confirm_all(confirm),
        subject="person",
    )
    return ComplianceMonitor(cfg)


PERSON = Detection("person", 0.9, (0.0, 0.0, 400.0, 900.0))


def glove(x, conf=0.9):
    return Detection("Gloves", conf, (x, 400.0, x + 40.0, 450.0))


def test_one_glove_is_a_violation():
    m = paired()
    assert m.update([[PERSON, glove(10)]], t=1.0) is Status.VIOLATION
    assert m.missing() == ["Gloves (1 of 2)"]


def test_two_gloves_pass():
    m = paired()
    assert m.update([[PERSON, glove(10), glove(200)]], t=1.0) is Status.OK
    assert m.missing() == []


def test_no_gloves_reports_the_full_shortfall():
    m = paired()
    assert m.update([[PERSON]], t=1.0) is Status.VIOLATION
    assert m.missing() == ["Gloves (0 of 2)"]


def test_more_than_required_still_passes():
    m = paired()
    assert m.update([[PERSON, glove(10), glove(100), glove(200)]], t=1.0) is Status.OK


def test_counts_are_the_best_single_view_not_the_sum():
    """Both cameras see the SAME hand; summing would pass a one-gloved worker."""
    m = paired()
    one_each = [[PERSON, glove(10)], [PERSON, glove(12)]]
    assert m.update(one_each, t=1.0) is Status.VIOLATION
    assert m.classes[0].count == 1


def test_the_better_camera_angle_wins():
    """A side view seeing one glove must not veto a front view seeing both."""
    m = paired()
    views = [[PERSON, glove(10)], [PERSON, glove(10), glove(200)]]
    assert m.update(views, t=1.0) is Status.OK
    assert m.classes[0].count == 2


def test_the_hold_window_keeps_the_best_recent_count():
    """A glove the model loses for a frame must not read as a bare hand."""
    m = paired(hold_ms=1000)
    m.update([[PERSON, glove(10), glove(200)]], t=10.0)
    assert m.update([[PERSON, glove(10)]], t=10.5) is Status.OK        # held at 2
    assert m.update([[PERSON, glove(10)]], t=11.6) is Status.VIOLATION  # expired


def test_a_dropped_glove_is_eventually_believed():
    m = paired(hold_ms=0)
    m.update([[PERSON, glove(10), glove(200)]], t=1.0)
    assert m.update([[PERSON, glove(10)]], t=2.0) is Status.VIOLATION


def test_single_count_classes_are_unchanged():
    m = monitor()
    assert m.update([[det("helmet"), det("vest")]], t=1.0) is Status.OK
    assert m.classes[0].count == 1 and m.classes[0].present


def test_a_plain_label_is_used_when_only_one_is_required():
    m = monitor()
    m.update([[det("helmet")]], t=1.0)
    assert m.missing() == ["Vest"]  # no "(0 of 1)" noise


def test_shortfall_and_presence_agree():
    m = paired()
    m.update([[PERSON, glove(10)]], t=1.0)
    state = m.classes[0]
    assert state.present and not state.compliant and state.shortfall == 1


def test_confidence_is_still_the_best_seen():
    m = paired()
    m.update([[PERSON, glove(10, 0.4), glove(200, 0.85)]], t=1.0)
    assert m.classes[0].conf == pytest.approx(0.85)


# -- the lamp follows the machine, and the buzzer marks it being cut -------
# Green now asserts the grinder is actually running, so it is decided by
# the latch rather than by compliance alone; and the buzzer is a fixed
# pulse fired when a violation takes a running machine away, rather than a
# tone held for as long as the violation stands.


def clock(monkeypatch, start=1000.0):
    """A hand-cranked clock, so a 3-second pulse costs no wall time."""
    t = {"now": start}
    monkeypatch.setattr("ppe.tower.now", lambda: t["now"])
    return t


def running_grinder(fake, tower, status=Status.OK):
    """Press the button and leave the grinder latched on."""
    fake.inputs = {0: True, 1: True}
    tower.update_belt_grinder(status)
    fake.inputs[1] = False
    tower.update_belt_grinder(status)
    fake.inputs[1] = True
    assert tower._grinder_latched is True


def test_compliant_but_idle_shows_amber_not_green():
    tower, fake = tower_with_grinder()
    tower.connect()
    fake.inputs = {0: True, 1: True}   # e-stop clear, nobody pressed start
    tower.update_belt_grinder(Status.OK)
    fake.writes.clear()

    tower.apply(Status.OK)
    assert dict(fake.writes) == {0: False, 1: True, 2: False, 3: False}


def test_compliant_and_running_shows_green():
    tower, fake = tower_with_grinder()
    tower.connect()
    running_grinder(fake, tower)
    fake.writes.clear()

    tower.apply(Status.OK)
    assert dict(fake.writes)[0] is True    # green
    assert dict(fake.writes)[1] is False   # amber off


def test_a_station_with_no_grinder_keeps_the_old_green(monkeypatch):
    """Nothing for green to wait on, so compliant is green as before."""
    tower, fake = tower_with_fake()        # no belt_grinder coil
    tower.connect()
    fake.writes.clear()
    tower.apply(Status.OK)
    assert dict(fake.writes)[0] is True


def test_a_violation_that_stops_a_running_grinder_sounds_the_buzzer(monkeypatch):
    clock(monkeypatch)
    tower, fake = tower_with_grinder()
    tower.cfg.buzzer_on_violation = True
    tower.connect()
    running_grinder(fake, tower)
    fake.writes.clear()

    tower.update_belt_grinder(Status.VIOLATION)
    tower.apply(Status.VIOLATION)
    assert dict(fake.writes)[4] is False   # grinder cut
    assert dict(fake.writes)[3] is True    # buzzer sounding
    assert dict(fake.writes)[2] is True    # red


def test_the_buzzer_stops_after_buzzer_sec(monkeypatch):
    t = clock(monkeypatch)
    tower, fake = tower_with_grinder()
    tower.cfg.buzzer_on_violation = True
    tower.cfg.buzzer_sec = 3.0
    tower.connect()
    running_grinder(fake, tower)
    tower.update_belt_grinder(Status.VIOLATION)
    tower.apply(Status.VIOLATION)
    assert tower._state["buzzer"] is True

    t["now"] += 2.9                       # still inside the window
    tower.apply(Status.VIOLATION)
    assert tower._state["buzzer"] is True

    t["now"] += 0.2                       # 3.1 s — expired
    fake.writes.clear()
    tower.apply(Status.VIOLATION)
    assert dict(fake.writes) == {3: False}, "the pulse must expire on its own"


def test_the_buzzer_does_not_re_sound_while_the_violation_stands(monkeypatch):
    """One pulse per stop, not one per cycle — the grinder is already off,
    so there is nothing further to announce."""
    t = clock(monkeypatch)
    tower, fake = tower_with_grinder()
    tower.cfg.buzzer_on_violation = True
    tower.connect()
    running_grinder(fake, tower)
    tower.update_belt_grinder(Status.VIOLATION)
    tower.apply(Status.VIOLATION)

    t["now"] += 5.0
    for _ in range(3):
        tower.update_belt_grinder(Status.VIOLATION)
        tower.apply(Status.VIOLATION)
    assert tower._state["buzzer"] is False


def test_a_violation_with_the_grinder_already_idle_is_silent(monkeypatch):
    clock(monkeypatch)
    tower, fake = tower_with_grinder()
    tower.cfg.buzzer_on_violation = True
    tower.connect()
    fake.inputs = {0: True, 1: True}      # never pressed
    tower.update_belt_grinder(Status.OK)
    fake.writes.clear()

    tower.update_belt_grinder(Status.VIOLATION)
    tower.apply(Status.VIOLATION)
    assert dict(fake.writes).get(3) is not True


@pytest.mark.parametrize("status", [Status.STANDBY, Status.DEGRADED])
def test_only_a_violation_sounds_it_not_every_stop(monkeypatch, status):
    """Stepping out of view or losing the cameras cuts the grinder just as
    hard, but neither is the worker doing the thing this calls out."""
    clock(monkeypatch)
    tower, fake = tower_with_grinder()
    tower.cfg.buzzer_on_violation = True
    tower.connect()
    running_grinder(fake, tower)
    fake.writes.clear()

    tower.update_belt_grinder(status)
    tower.apply(status)
    assert dict(fake.writes)[4] is False              # still stopped
    assert dict(fake.writes).get(3) is not True       # but silently


def test_buzzer_on_violation_false_keeps_it_silent(monkeypatch):
    clock(monkeypatch)
    tower, fake = tower_with_grinder()
    tower.cfg.buzzer_on_violation = False
    tower.connect()
    running_grinder(fake, tower)
    fake.writes.clear()

    tower.update_belt_grinder(Status.VIOLATION)
    tower.apply(Status.VIOLATION)
    assert dict(fake.writes).get(3) is not True


def test_the_lamp_goes_red_when_the_estop_is_hit():
    tower, fake = tower_with_grinder()
    tower.connect()
    running_grinder(fake, tower)
    tower.apply(Status.OK)
    assert tower._state["green"] is True

    fake.inputs[0] = False            # e-stop hit — active wiring, so False
    fake.writes.clear()
    tower.update_belt_grinder(Status.OK)
    tower.apply(Status.OK)
    assert dict(fake.writes)[2] is True     # red
    assert dict(fake.writes)[0] is False    # green off
    assert dict(fake.writes)[4] is False    # and the grinder is cut


def test_an_estop_over_an_empty_cell_is_red_not_dark():
    tower, fake = tower_with_grinder()
    tower.connect()
    fake.inputs = {0: False, 1: True}       # e-stop hit, nobody in the cell
    tower.update_belt_grinder(Status.STANDBY)
    fake.writes.clear()
    tower.apply(Status.STANDBY)
    assert dict(fake.writes)[2] is True


def test_releasing_the_estop_returns_the_lamp_to_the_status():
    tower, fake = tower_with_grinder()
    tower.connect()
    fake.inputs = {0: False, 1: True}
    tower.update_belt_grinder(Status.OK)
    tower.apply(Status.OK)
    assert tower._state["red"] is True

    fake.inputs[0] = True                   # released
    fake.writes.clear()
    tower.update_belt_grinder(Status.OK)
    tower.apply(Status.OK)
    # Compliant but idle — the latch stayed dropped, so amber, not green.
    assert dict(fake.writes)[1] is True
    assert dict(fake.writes)[2] is False


def test_an_unreadable_estop_does_not_claim_an_emergency():
    """A failed read is unknown, not hit. Asserting red on a bus glitch
    would cry wolf on the one colour that has to mean something."""
    tower, fake = tower_with_grinder()
    tower.connect()
    fake.inputs = {0: True, 1: True}
    tower.update_belt_grinder(Status.OK)
    fake.fail_read = True
    tower.update_belt_grinder(Status.OK)
    assert tower._estop_ok is None

    fake.fail_read = False
    tower._retry_at = 0.0
    fake.writes.clear()
    tower.apply(Status.VIOLATION)
    assert dict(fake.writes)[2] is True   # red, but because of the violation
    tower.apply(Status.OK)
    assert tower._state["red"] is False   # not a lingering fake e-stop


def test_a_station_with_no_estop_wired_never_shows_estop_red():
    tower, fake = tower_with_fake()       # no inputs configured at all
    tower.connect()
    tower.update_belt_grinder(Status.OK)  # no-ops
    fake.writes.clear()
    tower.apply(Status.OK)
    assert dict(fake.writes)[0] is True   # plain green, as before


# -- occlusion vs dropout --------------------------------------------------
# A pair only one of which is ever visible is the shape of this station's
# real geometry: the side camera sees one glove, the front sees none. The
# question a single hold window cannot answer is whether the other glove is
# behind the worker's body or off their hand, so there are two windows.


def paired_occluded(hold_ms=1000, occluded_ms=5000, count=2):
    cfg = PPECfg(
        classes=[ClassCfg("Gloves", count=count, hold_ms=hold_ms, occluded_ms=occluded_ms)],
        confirm_sec=confirm_all(0.0),
    )
    return ComplianceMonitor(cfg)


def gloves(n):
    return [[det("Gloves") for _ in range(n)]]


def test_one_glove_keeps_the_pair_credited_for_the_occlusion_window():
    m = paired_occluded()
    assert m.update(gloves(2), t=0.0) is Status.OK
    assert m.update(gloves(1), t=2.0) is Status.OK, "one visible is not one worn"
    assert m.update(gloves(1), t=4.9) is Status.OK
    assert m.update(gloves(1), t=5.1) is Status.VIOLATION, "the evidence went stale"


def test_seeing_nothing_falls_back_to_the_short_hold():
    """The safety half. A long occlusion window must not also buy a worker
    who shows no gloves at all the same span of credit."""
    m = paired_occluded(hold_ms=1000, occluded_ms=5000)
    m.update(gloves(2), t=0.0)
    assert m.update(gloves(0), t=0.5) is Status.OK          # bridged, as before
    assert m.update(gloves(0), t=1.5) is Status.VIOLATION   # 1.0s, not 5.0s


def test_the_pair_must_have_been_seen_to_be_credited():
    """The window extends evidence, it does not invent it: a worker who
    only ever shows one glove never passes."""
    m = paired_occluded()
    for t in (0.0, 1.0, 2.0, 3.0):
        assert m.update(gloves(1), t=t) is Status.VIOLATION


def test_occluded_defaults_to_hold_so_nothing_changes_unasked():
    m = paired_occluded(hold_ms=1000, occluded_ms=None)
    assert m.classes[0].occluded == m.classes[0].hold == 1.0
    m.update(gloves(2), t=0.0)
    assert m.update(gloves(1), t=0.5) is Status.OK
    assert m.update(gloves(1), t=1.5) is Status.VIOLATION


def test_a_single_count_class_is_unaffected_by_the_occlusion_window():
    """count: 1 has no partial state — it is seen or it is not, so only the
    dropout window can ever apply to it."""
    cfg = PPECfg(
        classes=[ClassCfg("helmet", hold_ms=500, occluded_ms=9000)],
        confirm_sec=confirm_all(0.0),
    )
    m = ComplianceMonitor(cfg)
    m.update([[det("helmet")]], t=0.0)
    assert m.update([[]], t=0.4) is Status.OK
    assert m.update([[]], t=0.6) is Status.VIOLATION, "the 9s window must not apply"


def test_regaining_the_pair_restarts_the_occlusion_window():
    m = paired_occluded()
    m.update(gloves(2), t=0.0)
    m.update(gloves(1), t=4.0)
    m.update(gloves(2), t=4.5)                              # both seen again
    assert m.update(gloves(1), t=9.0) is Status.OK          # 4.5s since, under 5
    assert m.update(gloves(1), t=9.6) is Status.VIOLATION
