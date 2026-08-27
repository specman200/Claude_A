"""The state machine that decides what colour the tower light shows."""

import pytest

from ppe.config import ClassCfg, PPECfg, TowerCfg
from ppe.detector import Detection
from ppe.tower import LAMPS, ComplianceMonitor, Status, TowerLight, make_tower


def det(name, conf=0.9):
    return Detection(name, conf, (0.0, 0.0, 10.0, 10.0))


def monitor(hold_ms=1000, confirm=1, optional=(), missing=()):
    cfg = PPECfg(
        classes=[ClassCfg("helmet"), ClassCfg("vest")]
        + [ClassCfg(n, required=False) for n in optional],
        hold_ms=hold_ms,
        confirm_frames=confirm,
    )
    return ComplianceMonitor(cfg, list(missing))


def test_all_required_present_is_ok():
    m = monitor()
    assert m.update([det("helmet"), det("vest")], t=1.0) is Status.OK
    assert m.missing() == []


def test_one_required_missing_is_a_violation():
    m = monitor()
    assert m.update([det("helmet")], t=1.0) is Status.VIOLATION
    assert m.missing() == ["Vest"]


def test_optional_classes_do_not_gate_the_light():
    m = monitor(optional=["mask"])
    assert m.update([det("helmet"), det("vest")], t=1.0) is Status.OK


def test_a_class_the_model_lacks_degrades_instead_of_passing():
    m = monitor(missing=["vest"])
    assert m.update([det("helmet")], t=1.0) is Status.DEGRADED
    assert [c.available for c in m.classes] == [True, False]
    assert m.unavailable() == ["Vest"]


def test_an_unavailable_optional_class_does_not_degrade_the_station():
    m = monitor(optional=["mask"], missing=["mask"])
    assert m.update([det("helmet"), det("vest")], t=1.0) is Status.OK
    assert m.unavailable() == []


def test_hold_window_bridges_a_dropped_frame():
    m = monitor(hold_ms=1000)
    m.update([det("helmet"), det("vest")], t=10.0)
    assert m.update([det("helmet")], t=10.5) is Status.OK      # vest still held
    assert m.update([det("helmet")], t=11.5) is Status.VIOLATION  # hold expired


def test_confirm_frames_debounce_the_relay():
    m = monitor(confirm=3)
    # Two agreeing frames are not enough; the third one flips the lamp.
    assert m.update([det("helmet"), det("vest")], t=1.0) is Status.DEGRADED
    assert m.update([det("helmet"), det("vest")], t=1.1) is Status.DEGRADED
    assert m.update([det("helmet"), det("vest")], t=1.2) is Status.OK

    # A single bad frame must not flip the light back.
    assert m.update([det("helmet")], t=1.4) is Status.OK
    assert m.update([det("helmet"), det("vest")], t=1.5) is Status.OK


def test_a_sustained_change_does_flip_after_confirmation():
    m = monitor(hold_ms=0, confirm=2)
    for t in (1.0, 1.1, 1.2):
        m.update([det("helmet"), det("vest")], t=t)
    assert m.status is Status.OK
    m.update([det("helmet")], t=1.3)
    assert m.update([det("helmet")], t=1.4) is Status.VIOLATION


def test_confidence_keeps_the_best_view_across_cameras():
    m = monitor()
    m.update([det("helmet", 0.4), det("helmet", 0.8), det("vest", 0.7)], t=1.0)
    assert m.classes[0].conf == pytest.approx(0.8)


def test_unknown_detections_are_ignored():
    m = monitor()
    assert m.update([det("forklift"), det("helmet"), det("vest")], t=1.0) is Status.OK


def test_degrade_forces_the_amber_state():
    m = monitor(confirm=1)
    m.update([det("helmet"), det("vest")], t=1.0)
    assert m.degrade() is Status.DEGRADED


def test_every_status_maps_to_exactly_one_lamp():
    assert set(LAMPS) == set(Status)
    for lamps in LAMPS.values():
        assert len(lamps) == 1


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
