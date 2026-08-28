"""The spoken prompt: when it fires, and — more importantly — when it does not.

A prompt that fires the instant PPE is missing trains people to ignore it, so
the grace period and the repeat interval are the parts that matter.
"""

import pytest

from ppe.annunciator import Annunciator
from ppe.tower import Status


class Fake(Annunciator):
    """Counts prompts without needing a sound card."""

    def __init__(self, grace=3.0, repeat=3.0, busy=False):
        super().__init__(None, grace, repeat)
        self.busy = busy

    @property
    def enabled(self):
        return True

    def _play(self):
        if self.busy:
            return False
        self.plays += 1
        return True


def test_no_audio_file_configured_is_silent_not_broken():
    a = Annunciator(None)
    assert not a.enabled
    assert a.update(Status.VIOLATION, t=0.0) is False
    assert a.update(Status.VIOLATION, t=100.0) is False  # never raises
    a.close()


def test_a_missing_audio_file_disables_rather_than_crashing(tmp_path):
    a = Annunciator(tmp_path / "nope.mp3")
    assert not a.enabled


def test_the_first_prompt_waits_out_the_grace_period():
    """Time to finish putting the glove on before being told about it."""
    a = Fake(grace=3.0)
    assert a.update(Status.VIOLATION, t=0.0) is False   # violation begins
    assert a.update(Status.VIOLATION, t=2.9) is False   # still inside grace
    assert a.update(Status.VIOLATION, t=3.0) is True    # grace elapsed
    assert a.plays == 1


def test_a_sustained_violation_keeps_prompting():
    a = Fake(grace=1.0, repeat=2.0)
    a.update(Status.VIOLATION, t=0.0)
    assert a.update(Status.VIOLATION, t=1.0) is True    # first, after grace
    assert a.update(Status.VIOLATION, t=2.0) is False   # inside repeat gap
    assert a.update(Status.VIOLATION, t=3.0) is True    # repeat due
    assert a.plays == 2


def test_complying_stops_the_prompts_and_resets_the_grace():
    a = Fake(grace=2.0)
    a.update(Status.VIOLATION, t=0.0)
    assert a.update(Status.VIOLATION, t=2.0) is True

    assert a.update(Status.OK, t=3.0) is False          # fixed it
    # A new violation gets a fresh grace period, not an instant nag.
    assert a.update(Status.VIOLATION, t=4.0) is False
    assert a.update(Status.VIOLATION, t=5.9) is False
    assert a.update(Status.VIOLATION, t=6.0) is True
    assert a.plays == 2


@pytest.mark.parametrize("status", [Status.OK, Status.STANDBY, Status.DEGRADED])
def test_only_a_violation_speaks(status):
    """An empty cell and a broken station are not things to announce."""
    a = Fake(grace=0.0)
    assert a.update(status, t=0.0) is False
    assert a.update(status, t=10.0) is False
    assert a.plays == 0


def test_prompts_do_not_stack_while_one_is_still_playing():
    a = Fake(grace=0.0, repeat=0.5, busy=True)
    a.update(Status.VIOLATION, t=0.0)
    for t in (0.5, 1.0, 1.5):
        assert a.update(Status.VIOLATION, t=t) is False
    assert a.plays == 0


def test_a_zero_grace_still_waits_one_update_to_establish_the_violation():
    a = Fake(grace=0.0)
    assert a.update(Status.VIOLATION, t=0.0) is False   # violation starts here
    assert a.update(Status.VIOLATION, t=0.0) is True    # due immediately after


def test_repeat_is_floored_so_a_zero_cannot_spin():
    a = Annunciator(None, grace_sec=0.0, repeat_sec=0.0)
    assert a.repeat >= 0.5


def test_negative_grace_is_clamped():
    assert Annunciator(None, grace_sec=-5.0).grace == 0.0
