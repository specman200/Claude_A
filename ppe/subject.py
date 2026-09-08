"""Decide who is being checked, and which detections belong to them.

PPE only means anything relative to a person. This module picks the subject —
the largest detection of the subject class, i.e. the one nearest the camera —
and keeps only the equipment found on that person. Everything else (bystanders
further back, gear on a bench) is set aside rather than counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Detection

Box = tuple[float, float, float, float]


def area(box: Box) -> float:
    """Area of an xyxy box; zero for a degenerate one."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def overlap(inner: Box, outer: Box) -> float:
    """Fraction of ``inner`` that lies inside ``outer`` (intersection over area).

    Measured against the inner box rather than the union, so a small glove
    fully inside a large person still scores 1.0 — which IoU would not.
    """
    own = area(inner)
    if own <= 0:
        return 0.0
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    if w <= 0 or h <= 0:
        return 0.0
    return (w * h) / own


def largest(detections: list[Detection], name: str) -> Detection | None:
    """The biggest detection of ``name`` — the person closest to the camera."""
    candidates = [d for d in detections if d.name == name]
    return max(candidates, key=lambda d: area(d.xyxy), default=None)


@dataclass(slots=True)
class Focus:
    """One camera's view, split into what counts and what does not."""

    subject: Detection | None = None
    accepted: list[Detection] = field(default_factory=list)
    rejected: list[Detection] = field(default_factory=list)

    @property
    def has_subject(self) -> bool:
        return self.subject is not None


def focus(
    detections: list[Detection],
    subject: str,
    containment: float = 0.5,
    per_class: dict[str, float] | None = None,
) -> Focus:
    """Split ``detections`` into the subject's equipment and everything else.

    With no ``subject`` class configured, every detection is accepted and the
    station behaves as it did before — the gating is opt-in.

    ``per_class`` overrides ``containment`` for the classes that name one.
    Not every item sits on the body the same way: a mask is squarely inside
    the person box, while gloves and sleeves ride the ends of outstretched
    arms and routinely fall outside it. One global threshold has to be
    loose enough for the worst case, which then lets a bystander's gear
    count for everything else.
    """
    if not subject:
        return Focus(None, list(detections), [])

    chosen = largest(detections, subject)
    if chosen is None:
        # Nobody to check: nothing counts, but keep the rest for display.
        return Focus(None, [], list(detections))

    accepted, rejected = [chosen], []
    for det in detections:
        if det is chosen:
            continue
        # Other people are bystanders, not the subject, whatever they wear.
        need = per_class.get(det.name, containment) if per_class else containment
        if det.name == subject or overlap(det.xyxy, chosen.xyxy) < need:
            rejected.append(det)
        else:
            accepted.append(det)
    return Focus(chosen, accepted, rejected)
