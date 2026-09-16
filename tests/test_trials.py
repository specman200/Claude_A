"""The trial log: does it name the class that caused a nuisance stop?"""

import csv

from ppe.config import ClassCfg, PPECfg
from ppe.detector import Detection
from ppe.tower import ClassState, ComplianceMonitor, Status
from ppe.trials import ABSENT, SHORT, TrialLog, report


def det(name, conf=0.9):
    return Detection(name, conf, (0.0, 0.0, 10.0, 10.0))


class FakeResult:
    """Only the fields TrialLog reads."""

    def __init__(self, classes, status, grinder_on, cause="", missing=(), banned=()):
        self.classes = classes
        self.status = status
        self.grinder_on = grinder_on
        self.stop_cause = cause
        self.missing = list(missing)
        self.banned = list(banned)


def monitor(**kw):
    cfg = PPECfg(
        classes=[ClassCfg("mask", hold_ms=700), ClassCfg("gloves", count=2,
                                                         hold_ms=1000, occluded_ms=5000)],
        hold_ms=1000,
        confirm_sec=dict.fromkeys(("ok", "violation", "standby", "degraded"), 0.4),
        **kw,
    )
    return ComplianceMonitor(cfg)


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_a_dropout_that_stops_the_machine_names_the_class(tmp_path):
    """The whole point: a stop the worker did not earn, traced to its cause."""
    log = TrialLog(tmp_path)
    m = monitor()
    worn = [det("mask"), det("gloves"), det("gloves")]
    t = 0.0

    # Running and compliant.
    for _ in range(10):
        status = m.update([worn], t)
        log.observe(FakeResult(m.classes, status, grinder_on=True), t)
        t += 0.1
    assert status is Status.OK

    # The mask drops out of view for 1.4 s — longer than its 700 ms hold, so
    # the verdict flips and the latch goes, though the worker never took it off.
    blind = [det("gloves"), det("gloves")]
    stopped = False
    for _ in range(14):
        status = m.update([blind], t)
        stopped = stopped or status is not Status.OK
        log.observe(FakeResult(m.classes, status, grinder_on=not stopped,
                               cause="status is violation", missing=m.missing()), t)
        t += 0.1
    assert stopped, "1.4s without the mask should have taken the machine"

    # ...and it comes straight back, which is what makes it a nuisance stop.
    for _ in range(5):
        status = m.update([worn], t)
        log.observe(FakeResult(m.classes, status, grinder_on=False), t)
        t += 0.1
    log.close()

    stops = rows(tmp_path / "stops.csv")
    assert len(stops) == 1
    assert stops[0]["blamed"] == "mask"
    assert 0.5 < float(stops[0]["blamed_gap_s"]) < 1.6
    assert "mask" in stops[0]["open_gaps"]

    gaps = [g for g in rows(tmp_path / "gaps.csv") if g["class"] == "mask"]
    assert len(gaps) == 1
    assert gaps[0]["kind"] == ABSENT
    assert float(gaps[0]["duration_s"]) > 0.7
    assert gaps[0]["exceeded_window"] == "1", "a gap past the hold window"


def test_a_partly_visible_pair_is_logged_as_short_not_absent(tmp_path):
    """One glove behind the body is a different fault to no gloves at all, and
    rides a different window — the log has to keep them apart or the report
    recommends the wrong setting."""
    log = TrialLog(tmp_path)
    m = monitor()
    t = 0.0
    for _ in range(5):
        log.observe(FakeResult(m.classes, m.update([[det("mask"), det("gloves"),
                                                     det("gloves")]], t), True), t)
        t += 0.1
    for _ in range(8):                      # one glove only
        log.observe(FakeResult(m.classes, m.update([[det("mask"), det("gloves")]], t),
                               True), t)
        t += 0.1
    for _ in range(3):
        log.observe(FakeResult(m.classes, m.update([[det("mask"), det("gloves"),
                                                     det("gloves")]], t), True), t)
        t += 0.1
    log.close()

    gaps = [g for g in rows(tmp_path / "gaps.csv") if g["class"] == "gloves"]
    assert [g["kind"] for g in gaps] == [SHORT]
    assert gaps[0]["worst_seen"] == "1" and gaps[0]["need"] == "2"
    assert float(gaps[0]["window_s"]) == 5.0, "the occlusion window, not the hold"
    assert gaps[0]["exceeded_window"] == "0", "0.8s is well inside 5s — no stop earned"
    assert not rows(tmp_path / "stops.csv")


def test_the_report_recommends_a_window_and_prices_it(tmp_path):
    log = TrialLog(tmp_path)
    m = monitor()
    worn = [det("mask"), det("gloves"), det("gloves")]
    t = 0.0
    for _ in range(3):                      # three identical nuisance dropouts
        for _ in range(8):
            log.observe(FakeResult(m.classes, m.update([worn], t), True), t)
            t += 0.1
        for _ in range(12):                 # 1.2 s without the mask
            st = m.update([[det("gloves"), det("gloves")]], t)
            log.observe(FakeResult(m.classes, st, grinder_on=st is Status.OK), t)
            t += 0.1
    log.close()

    text = report(tmp_path, confirm_s=0.4)
    assert "mask" in text
    assert "hold_ms" in text and "->" in text
    assert "costs" in text, "the report must price the change, not just suggest it"
    assert "stops/hour" in text


def test_only_required_and_available_classes_are_tracked(tmp_path):
    """An optional class dropping out stops nothing, so it is noise here; a
    class the model does not carry is a DEGRADED problem, not a gap."""
    log = TrialLog(tmp_path)
    states = [
        ClassState("person", "Person", required=False, need=1, seen=0),
        ClassState("ghost", "Ghost", required=True, need=1, seen=0, available=False),
        ClassState("mask", "Mask", required=True, need=1, seen=0),
    ]
    log.observe(FakeResult(states, Status.VIOLATION, grinder_on=True), 0.0)
    for s in states:
        s.seen = 1
    log.observe(FakeResult(states, Status.OK, grinder_on=True), 1.0)
    log.close()
    assert {g["class"] for g in rows(tmp_path / "gaps.csv")} == {"mask"}


def test_a_stop_that_lands_after_the_item_came_back_is_still_blamed_on_it(tmp_path):
    """The purest nuisance stop: the glove is back in view, but the verdict is
    debounced and the latch follows the verdict, so the machine stops anyway.
    'No gap was open' would hide precisely the case worth seeing."""
    log = TrialLog(tmp_path)
    states = [ClassState("mask", "Mask", required=True, need=1, seen=1, hold=0.7)]
    log.observe(FakeResult(states, Status.OK, grinder_on=True), 0.0)
    states[0].seen = 0                                  # out of view for 1.2 s
    for t in (0.1, 0.6, 1.2):
        log.observe(FakeResult(states, Status.OK, grinder_on=True), t)
    states[0].seen = 1                                  # ...and back
    log.observe(FakeResult(states, Status.VIOLATION, grinder_on=True), 1.3)
    log.observe(FakeResult(states, Status.VIOLATION, grinder_on=False), 1.4)
    log.close()

    stops = rows(tmp_path / "stops.csv")
    assert len(stops) == 1
    assert stops[0]["blamed"] == "mask (recovered)"
    assert "just closed" in stops[0]["open_gaps"]


def test_a_forbidden_class_firing_is_logged_with_the_confidence_to_tune_against(tmp_path):
    """Absence is what a forbidden class is supposed to be, so its permanent
    'gap' is compliance. What costs a stop is a sighting — and what fixes a
    false one is the confidence floor, not a window."""
    log = TrialLog(tmp_path)
    states = [ClassState("Wrong Sleeve", "Wrong Sleeve", required=True, expect="absent",
                         need=1, seen=0, hold=0.4)]
    for t in (0.0, 0.1, 0.2):
        log.observe(FakeResult(states, Status.OK, grinder_on=True), t)
    states[0].seen, states[0].conf = 1, 0.93            # a false fire
    log.observe(FakeResult(states, Status.VIOLATION, grinder_on=True), 0.3)
    states[0].conf = 0.95
    log.observe(FakeResult(states, Status.VIOLATION, grinder_on=False), 0.4)
    states[0].seen, states[0].conf = 0, 0.0
    log.observe(FakeResult(states, Status.OK, grinder_on=False), 0.6)
    log.close()

    gaps = rows(tmp_path / "gaps.csv")
    assert len(gaps) == 1, "the standing absence is compliance, not a gap"
    assert gaps[0]["kind"] == "fired"
    assert float(gaps[0]["peak_conf"]) == 0.95
    assert gaps[0]["exceeded_window"] == "1", "any sighting is a fault at once"
    assert rows(tmp_path / "stops.csv")[0]["blamed"] == "Wrong Sleeve"

    text = report(tmp_path)
    assert "FORBIDDEN-CLASS SIGHTINGS" in text
    assert "0.95" in text and "conf" in text


def test_the_subject_is_tracked_even_though_it_gates_nothing(tmp_path):
    """Losing the person takes the station to STANDBY, which stops the motor
    just as hard as a violation does."""
    log = TrialLog(tmp_path, subject="person")
    states = [ClassState("person", "Person", required=False, need=1, seen=1, hold=1.0)]
    log.observe(FakeResult(states, Status.OK, grinder_on=True), 0.0)
    states[0].seen = 0
    log.observe(FakeResult(states, Status.OK, grinder_on=True), 1.0)
    log.observe(FakeResult(states, Status.STANDBY, grinder_on=False), 2.1)
    log.close()
    assert rows(tmp_path / "stops.csv")[0]["blamed"] == "person"


def test_a_class_the_subject_gate_threw_away_is_not_reported_as_a_detection_problem(tmp_path):
    """The mask was in view the whole time; containment put it off the person
    and dropped it. Raising hold_ms would do nothing at all here, and the
    report has to say so rather than recommend it."""
    log = TrialLog(tmp_path)
    states = [ClassState("Mask", "Face Mask", required=True, need=1, seen=1, hold=0.7)]
    log.observe(FakeResult(states, Status.OK, grinder_on=True), 0.0)
    states[0].seen = 0
    for t in (0.1, 0.4, 0.7, 1.0, 1.3):
        r = FakeResult(states, Status.VIOLATION, grinder_on=True)
        r.ignored = [[det("Mask")]]          # found, but not on the subject
        log.observe(r, t)
    states[0].seen = 1
    log.observe(FakeResult(states, Status.OK, grinder_on=True), 1.6)
    log.close()

    gap = rows(tmp_path / "gaps.csv")[0]
    assert gap["off_subject_frames"] == gap["frames"] == "5"

    text = report(tmp_path)
    assert "SEEN, BUT NOT CREDITED" in text
    assert "containment" in text
    assert "100% of the gap frames" in text


def test_a_genuine_dropout_is_not_blamed_on_containment(tmp_path):
    log = TrialLog(tmp_path)
    states = [ClassState("Mask", "Face Mask", required=True, need=1, seen=1, hold=0.7)]
    log.observe(FakeResult(states, Status.OK, grinder_on=True), 0.0)
    states[0].seen = 0
    for t in (0.1, 0.4, 0.7, 1.0, 1.3):
        r = FakeResult(states, Status.VIOLATION, grinder_on=True)
        r.ignored = [[]]                     # nothing found anywhere
        log.observe(r, t)
    states[0].seen = 1
    log.observe(FakeResult(states, Status.OK, grinder_on=True), 1.6)
    log.close()
    assert rows(tmp_path / "gaps.csv")[0]["off_subject_frames"] == "0"
    assert "SEEN, BUT NOT CREDITED" not in report(tmp_path)


def test_the_report_does_not_recommend_a_window_it_has_just_said_would_not_help(tmp_path):
    """Contradicting itself one section later is how a report stops being read."""
    log = TrialLog(tmp_path)
    states = [ClassState("Mask", "Face Mask", required=True, need=1, seen=1, hold=0.7)]
    for run in range(3):
        log.observe(FakeResult(states, Status.OK, grinder_on=True), run * 10.0)
        states[0].seen = 0
        for t in (0.2, 0.6, 1.0, 1.4):
            r = FakeResult(states, Status.VIOLATION, grinder_on=True)
            r.ignored = [[det("Mask")]]
            log.observe(r, run * 10.0 + t)
        states[0].seen = 1
        log.observe(FakeResult(states, Status.OK, grinder_on=True), run * 10.0 + 1.8)
    log.close()

    text = report(tmp_path)
    assert "SEEN, BUT NOT CREDITED" in text
    assert "no window would help" in text
    assert "hold_ms" not in text.split("WHAT WOULD HAVE HELD")[1], (
        "a hold_ms recommendation here contradicts the containment finding above"
    )
