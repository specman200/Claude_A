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


# -- Modbus output ---------------------------------------------------------


class FakeClient:
    """Stands in for pymodbus: records coil writes, can be told to fail."""

    def __init__(self):
        self.writes = []
        self.fail = False
        self.closed = False

    def connect(self):
        return True

    def write_coil(self, address, value, slave=None):
        if self.fail:
            raise OSError("bus down")
        self.writes.append((address, value))
        return type("Rsp", (), {"isError": lambda _self: False})()

    def close(self):
        self.closed = True


def tower_with_fake():
    tower = TowerLight(TowerCfg(coils={"green": 0, "amber": 1, "red": 2, "buzzer": 3}))
    fake = FakeClient()
    tower._make_client = lambda: fake
    return tower, fake


def test_apply_energises_only_the_status_lamp():
    tower, fake = tower_with_fake()
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


def test_buzzer_only_sounds_on_violation_when_enabled():
    tower, fake = tower_with_fake()
    tower.cfg.buzzer_on_violation = True
    tower.apply(Status.VIOLATION)
    assert dict(fake.writes)[3] is True
    fake.writes.clear()
    tower.apply(Status.OK)
    assert dict(fake.writes)[3] is False


def test_a_bus_failure_is_survived_and_resynced():
    tower, fake = tower_with_fake()
    tower.apply(Status.OK)
    fake.fail = True
    assert tower.apply(Status.VIOLATION) is False
    assert tower.connected is False

    fake.fail = False
    tower._retry_at = 0.0
    assert tower.apply(Status.VIOLATION)
    # After a reconnect every coil is rewritten, not just the changed ones.
    assert set(dict(fake.writes)) == {0, 1, 2, 3}


def test_a_disabled_tower_is_a_no_op():
    tower = make_tower(TowerCfg(enabled=False))
    assert tower.apply(Status.VIOLATION) is False
    tower.close()


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
