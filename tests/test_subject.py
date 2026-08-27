"""Picking the person under assessment, and the gear that belongs to them."""

import pytest

from ppe.detector import Detection
from ppe.subject import area, focus, largest, overlap


def det(name, box, conf=0.9):
    return Detection(name, conf, box)


PERSON_NEAR = det("person", (100.0, 100.0, 500.0, 900.0))   # 400 x 800
PERSON_FAR = det("person", (600.0, 300.0, 700.0, 500.0))    # 100 x 200, smaller


def test_area_of_a_box():
    assert area((0.0, 0.0, 10.0, 4.0)) == 40.0
    assert area((10.0, 10.0, 0.0, 0.0)) == 0.0  # degenerate, not negative


@pytest.mark.parametrize(
    "inner,expected",
    [
        ((10.0, 10.0, 20.0, 20.0), 1.0),     # fully inside
        ((200.0, 200.0, 210.0, 210.0), 0.0),  # fully outside
        ((-10.0, 0.0, 10.0, 10.0), 0.5),     # half in
        ((0.0, 0.0, 0.0, 0.0), 0.0),         # degenerate
    ],
)
def test_overlap_is_measured_against_the_inner_box(inner, expected):
    outer = (0.0, 0.0, 100.0, 100.0)
    assert overlap(inner, outer) == pytest.approx(expected)


def test_a_small_item_fully_inside_scores_one():
    """IoU would score this near zero; what matters is that the glove is on them."""
    glove = (200.0, 400.0, 240.0, 440.0)
    assert overlap(glove, PERSON_NEAR.xyxy) == 1.0


def test_largest_picks_the_nearest_person():
    assert largest([PERSON_FAR, PERSON_NEAR], "person") is PERSON_NEAR


def test_largest_returns_nothing_when_the_class_is_absent():
    assert largest([det("Gloves", (0.0, 0.0, 5.0, 5.0))], "person") is None
    assert largest([], "person") is None


# -- focus -----------------------------------------------------------------


def test_no_subject_configured_accepts_everything():
    """Gating is opt-in: an empty subject leaves the old behaviour intact."""
    dets = [det("Gloves", (0.0, 0.0, 5.0, 5.0)), PERSON_FAR]
    result = focus(dets, subject="", containment=0.5)
    assert result.accepted == dets
    assert result.rejected == [] and result.subject is None


def test_nobody_present_means_nothing_counts():
    gloves = det("Gloves", (0.0, 0.0, 5.0, 5.0))
    result = focus([gloves], subject="person", containment=0.5)
    assert not result.has_subject
    assert result.accepted == []
    assert result.rejected == [gloves]  # kept so the UI can still show them


def test_only_gear_on_the_subject_is_counted():
    on = det("Gloves", (200.0, 400.0, 260.0, 460.0))       # inside the near person
    off = det("Mask", (620.0, 320.0, 660.0, 360.0))        # on the far person
    bench = det("headnet", (900.0, 900.0, 950.0, 950.0))   # nowhere near anyone

    result = focus([PERSON_NEAR, PERSON_FAR, on, off, bench], "person", 0.5)
    assert result.subject is PERSON_NEAR
    assert on in result.accepted
    assert off in result.rejected and bench in result.rejected


def test_the_smaller_person_is_a_bystander_not_a_second_subject():
    result = focus([PERSON_NEAR, PERSON_FAR], "person", 0.5)
    assert result.accepted == [PERSON_NEAR]
    assert result.rejected == [PERSON_FAR]


def test_a_bystander_inside_the_subject_box_is_still_not_the_subject():
    """Overlapping people must not be absorbed into the subject's kit."""
    behind = det("person", (150.0, 150.0, 300.0, 600.0))  # smaller, fully overlapped
    result = focus([PERSON_NEAR, behind], "person", 0.5)
    assert result.subject is PERSON_NEAR
    assert behind in result.rejected


@pytest.mark.parametrize(
    "threshold,accepted",
    [(0.3, True), (0.5, True), (0.8, False)],
)
def test_containment_threshold_decides_partial_overlaps(threshold, accepted):
    # A box straddling the subject's left edge: 60% of it is inside.
    straddle = det("Gloves", (60.0, 400.0, 160.0, 500.0))
    assert overlap(straddle.xyxy, PERSON_NEAR.xyxy) == pytest.approx(0.6)
    result = focus([PERSON_NEAR, straddle], "person", threshold)
    assert (straddle in result.accepted) is accepted


def test_the_subject_is_always_among_the_accepted():
    result = focus([PERSON_NEAR], "person", 0.5)
    assert result.accepted == [PERSON_NEAR] and result.has_subject


def test_nothing_is_lost_between_accepted_and_rejected():
    dets = [PERSON_NEAR, PERSON_FAR, det("Gloves", (200.0, 400.0, 260.0, 460.0))]
    result = focus(dets, "person", 0.5)
    assert len(result.accepted) + len(result.rejected) == len(dets)
