"""Trial instrumentation: why the station stopped, and what it could not see.

A nuisance stop is a stop the worker did not earn — the PPE was on, the
detector lost sight of it, the hold window ran out and the latch dropped.
Tuning that away by raising thresholds until it stops happening also raises
how long a genuinely bare-handed worker keeps the machine, so it is not a
knob to turn blind. This module records the evidence needed to turn it with
a number instead:

  stops.csv   one row per latch drop, naming the classes whose sighting had
              lapsed at that moment and for how long
  gaps.csv    one row per detection gap per class, whether or not it caused
              a stop — the distribution a hold window has to cover

Both are derived entirely from the published Result stream, so nothing here
can influence a verdict. It is an observer, and deliberately only that.

Run ``python -m ppe.trials logs/trials`` over a finished trial to turn the
two files into the answer: which class, how often, and what window would
have covered it.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

from .tower import ClassState

ABSENT, SHORT, FIRED = "absent", "short", "fired"


@dataclass(slots=True)
class Gap:
    """A stretch during which a class was not fully in view."""

    klass: str
    kind: str          # ABSENT / SHORT for worn PPE, FIRED for a forbidden class
    started: float
    worst: int         # fewest seen at any point in the gap
    need: int
    window: float      # the hold or occlusion window that applied
    peak_conf: float = 0.0   # the best confidence reached — the number to tune
                             # a forbidden class's threshold against
    frames: int = 0          # cycles this gap has lasted
    off_subject: int = 0     # ...of which the class WAS detected, and thrown
                             # away by the containment test. A gap that is
                             # mostly this is not a detection problem at all.


class TrialLog:
    """Consumes published results; writes the two trial traces."""

    STOP_COLS = ["wall_time", "elapsed_s", "cause", "status", "missing", "banned",
                 "blamed", "blamed_gap_s", "open_gaps"]
    GAP_COLS = ["wall_time", "elapsed_s", "class", "kind", "duration_s",
                "worst_seen", "need", "window_s", "exceeded_window", "peak_conf",
                "frames", "off_subject_frames"]

    def __init__(self, directory: str | Path, subject: str = "") -> None:
        # The subject class is not "required" — it gates nothing directly —
        # but losing it takes the station to STANDBY, which stops the motor
        # just as hard. Tracked for exactly that reason.
        self.subject = subject
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self._stops, self._stop_fh = self._open(d / "stops.csv", self.STOP_COLS)
        self._gaps, self._gap_fh = self._open(d / "gaps.csv", self.GAP_COLS)
        self._open_gaps: dict[str, Gap] = {}
        # Gaps that have just closed, by class: (when, how long). A stop can
        # land a cycle or two after the item came back into view — the verdict
        # is debounced and the latch follows the verdict — and that late stop
        # is the purest nuisance stop there is. Blaming "nothing was open"
        # would hide exactly the case worth seeing.
        self._recent: dict[str, tuple[float, float]] = {}
        self._was_on = False
        self._t0: float | None = None

    @staticmethod
    def _open(path: Path, header: list[str]):
        fresh = not path.exists() or path.stat().st_size == 0
        fh = path.open("a", newline="")
        writer = csv.writer(fh)
        if fresh:
            writer.writerow(header)
        return writer, fh

    # -- recording ---------------------------------------------------------
    def observe(self, result, t: float) -> None:
        """Fold one cycle in. `t` is the monotonic clock the states carry."""
        if self._t0 is None:
            self._t0 = t
        # Detections the subject gate rejected. A class missing from the
        # verdict because it was never detected and one missing because it
        # was detected just off the person are the same gap from here, and
        # they have nothing in common as problems: one is the model or the
        # lighting, the other is a containment threshold.
        off = set()
        for rejected in getattr(result, "ignored", ()) or ():
            off.update(d.name for d in rejected)
        self._track_gaps(result.classes, t, off)
        if self._was_on and not result.grinder_on:
            self._record_stop(result, t)
        self._was_on = result.grinder_on

    def _track_gaps(self, classes: list[ClassState], t: float,
                    off_subject: set[str]) -> None:
        for c in classes:
            if not c.available or not (c.required or c.name == self.subject):
                continue
            if c.forbidden:
                # Absence is what this class is supposed to be, so a gap in it
                # is compliance, not a fault. What costs a stop here is the
                # opposite — a sighting — and what fixes a false one is the
                # confidence floor, not a hold window.
                kind, window = (FIRED, c.hold) if c.seen else ("", 0.0)
            elif c.seen == 0:
                kind, window = ABSENT, c.hold
            elif c.seen < c.need:
                kind, window = SHORT, c.occluded
            else:
                kind, window = "", 0.0

            gap = self._open_gaps.get(c.name)
            if not kind:
                if gap is not None:
                    self._close_gap(self._open_gaps.pop(c.name), t)
                continue
            if gap is not None and gap.kind == kind:
                gap.worst = min(gap.worst, c.seen)
                gap.peak_conf = max(gap.peak_conf, c.conf)
                gap.frames += 1
                gap.off_subject += c.name in off_subject
                continue
            if gap is not None:
                # absent -> short, or back: a different window applies, so the
                # old stretch is over on its own terms and a new one opens.
                self._close_gap(self._open_gaps.pop(c.name), t)
            self._open_gaps[c.name] = Gap(c.name, kind, t, c.seen, c.need,
                                          window, c.conf, frames=1,
                                          off_subject=int(c.name in off_subject))

    RECENT_S = 3.0   # how long a closed gap stays a candidate for blame

    def _close_gap(self, gap: Gap, t: float) -> None:
        secs = t - gap.started
        self._recent[gap.klass] = (t, secs)
        self._gaps.writerow([
            f"{time.time():.3f}", f"{gap.started - self._t0:.2f}", gap.klass,
            gap.kind, f"{secs:.3f}", gap.worst, gap.need,
            f"{gap.window:.3f}",
            1 if gap.kind == FIRED else int(secs > gap.window),
            f"{gap.peak_conf:.3f}", gap.frames, gap.off_subject,
        ])
        self._gap_fh.flush()

    def _record_stop(self, result, t: float) -> None:
        ages = sorted(((t - g.started, name, "open") for name, g in self._open_gaps.items()),
                      reverse=True)
        if not ages:
            ages = sorted(
                ((secs, name, "just closed")
                 for name, (closed, secs) in self._recent.items()
                 if t - closed <= self.RECENT_S),
                reverse=True)
        blamed = f"{ages[0][1]}" if ages else ""
        if ages and ages[0][2] == "just closed":
            blamed += " (recovered)"
        self._stops.writerow([
            f"{time.time():.3f}", f"{t - self._t0:.2f}",
            getattr(result, "stop_cause", "") or "",
            result.status.value,
            "; ".join(result.missing), "; ".join(result.banned),
            blamed,
            f"{ages[0][0]:.3f}" if ages else "",
            "; ".join(f"{n}={a:.2f}s {how}" for a, n, how in ages),
        ])
        self._stop_fh.flush()

    def close(self) -> None:
        for fh in (self._stop_fh, self._gap_fh):
            if fh is not None and not fh.closed:
                fh.close()


# --------------------------------------------------------------------------
# Reading the traces back
# --------------------------------------------------------------------------

def _pct(values: list[float], fraction: float) -> float:
    """The value below which `fraction` of the samples fall."""
    if not values:
        return 0.0
    ordered = sorted(values)
    i = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[i]


def report(directory: str | Path, confirm_s: float = 0.4) -> str:
    """Turn a finished trial into the answer to 'what do I change?'.

    `confirm_s` is ppe.confirm_sec.violation — the extra time a verdict must
    stand before it acts, which adds to every window below.
    """
    d = Path(directory)
    gaps, stops = [], []
    gap_path, stop_path = d / "gaps.csv", d / "stops.csv"
    if gap_path.exists():
        with gap_path.open(newline="") as fh:
            gaps = list(csv.DictReader(fh))
    if stop_path.exists():
        with stop_path.open(newline="") as fh:
            stops = list(csv.DictReader(fh))

    out: list[str] = []
    if not gaps and not stops:
        return f"no trial data in {d}/ — is telemetry.trials set, and has the station run?"

    span = max((float(r["elapsed_s"]) for r in gaps + stops), default=0.0)
    out.append(f"TRIAL  {d}    {span / 60:.1f} min observed    "
               f"{len(stops)} stop(s)    {len(gaps)} detection gap(s)")

    # -- what stopped the machine
    if stops:
        out.append("\nSTOPS, by what was out of sight at the time")
        by_blame: dict[str, list[float]] = {}
        for r in stops:
            key = r["blamed"] or f"(no gap open — cause: {r['cause'] or 'unknown'})"
            by_blame.setdefault(key, []).append(float(r["blamed_gap_s"] or 0))
        for name, ages in sorted(by_blame.items(), key=lambda kv: -len(kv[1])):
            share = 100 * len(ages) / len(stops)
            out.append(f"  {len(ages):3d}  ({share:3.0f}%)  {name:<16}"
                       f"  gap at the stop: {min(ages):.2f}–{max(ages):.2f}s")
        rate = len(stops) / (span / 3600) if span else 0.0
        out.append(f"  rate: {rate:.1f} stops/hour  ({rate * 8:.0f} per 8-hour shift)")

    # -- what the windows would have to cover
    if gaps:
        out.append("\nDETECTION GAPS, per class    (window = the hold or occlusion "
                   "window that applied)")
        out.append(f"  {'class':<16} {'kind':<7} {'n':>4} {'median':>8} {'p95':>8} "
                   f"{'max':>8} {'window':>8} {'over':>6}")
        rows: dict[tuple[str, str], list[dict]] = {}
        for r in gaps:
            rows.setdefault((r["class"], r["kind"]), []).append(r)
        for (name, kind), rs in sorted(rows.items(), key=lambda kv: -len(kv[1])):
            secs = [float(r["duration_s"]) for r in rs]
            over = sum(int(r["exceeded_window"]) for r in rs)
            window = float(rs[0]["window_s"])
            out.append(f"  {name:<16} {kind:<7} {len(rs):>4} {_pct(secs, 0.5):>7.2f}s "
                       f"{_pct(secs, 0.95):>7.2f}s {max(secs):>7.2f}s "
                       f"{window:>7.2f}s {over:>6}")

        seen_but_rejected = []
        for (name, kind), rs in sorted(rows.items()):
            if kind == FIRED:
                continue
            frames = sum(int(r.get("frames") or 0) for r in rs)
            off = sum(int(r.get("off_subject_frames") or 0) for r in rs)
            if frames and off / frames >= 0.2:
                seen_but_rejected.append((name, kind, off, frames))
        if seen_but_rejected:
            out.append("\nSEEN, BUT NOT CREDITED TO THE WORKER    (the detector found "
                       "it; the containment\n                                        "
                       "test put it off the subject and dropped it)")
            for name, kind, off, frames in seen_but_rejected:
                out.append(f"  {name:<16} {kind:<7} {100 * off / frames:3.0f}% of the gap "
                           f"frames had a rejected detection ({off} of {frames})")
            out.append("  No hold window fixes this — the item was in view the whole time. "
                       "Loosen that\n  class's `containment`, or look at why the person "
                       "box is missing the item:\n  a tight or jittery subject box clips "
                       "whatever sits at its edge.")

        fired = {k: v for k, v in rows.items() if k[1] == FIRED}
        if fired:
            out.append("\nFORBIDDEN-CLASS SIGHTINGS    (every one of these is a stop; "
                       "if they are false,\n                             the fix is the "
                       "confidence floor, not a window)")
            for (name, _), rs in sorted(fired.items(), key=lambda kv: -len(kv[1])):
                confs = [float(r["peak_conf"]) for r in rs]
                secs = [float(r["duration_s"]) for r in rs]
                out.append(f"  {name:<16} {len(rs):>4} sighting(s)   "
                           f"held {_pct(secs, 0.5):.2f}s median, {max(secs):.2f}s worst")
                out.append(f"  {'':16} peak confidence: median {_pct(confs, 0.5):.2f}, "
                           f"p95 {_pct(confs, 0.95):.2f}, max {max(confs):.2f}")
                out.append(f"  {'':16} raising conf above {max(confs):.2f} would remove "
                           f"all of them; above {_pct(confs, 0.95):.2f} removes "
                           f"{sum(c <= _pct(confs, 0.95) for c in confs)} of {len(confs)}."
                           f"\n  {'':16} Check the frames first: a real wrong sleeve that "
                           f"you threshold away is\n  {'':16} the one failure this class "
                           f"exists to catch.")

        out.append("\nWHAT WOULD HAVE HELD    (window needed to ride out 95% / 99% of "
                   "gaps, allowing for confirm)")
        rejected_share = {(n, k): off / frames
                          for n, k, off, frames in seen_but_rejected}
        for (name, kind), rs in sorted(rows.items()):
            if kind == FIRED:
                continue
            share = rejected_share.get((name, kind), 0.0)
            if share >= 0.8:
                # Recommending a window here would contradict the section
                # above: the item was in view, so no amount of holding the
                # last sighting helps when there was nothing to hold.
                out.append(f"  {name:<16} {'':12} no window would help — see "
                           f"'seen, but not credited' above")
                continue
            secs = [float(r["duration_s"]) for r in rs]
            over = sum(int(r["exceeded_window"]) for r in rs)
            if not over:
                continue
            window = float(rs[0]["window_s"])
            setting = "hold_ms" if kind == ABSENT else "occluded_ms"
            for label, frac in (("95%", 0.95), ("99%", 0.99)):
                need = max(0.0, _pct(secs, frac) - confirm_s)
                if share:
                    label += f" (but {100 * share:.0f}% were rejected, not unseen)"
                out.append(f"  {name:<16} {setting:<12} {window * 1000:>6.0f} -> "
                           f"{need * 1000:>6.0f} ms  covers {label} of gaps"
                           f"   (costs {need - window:+.2f}s before a real "
                           f"removal stops the machine)")
        out.append("\n  The cost column is the point: every millisecond added here is a "
                   "millisecond\n  a worker who has actually taken the item off keeps "
                   "the machine. Raise the\n  window where the gaps are the detector "
                   "blinking; fix the detector where they\n  are not.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m ppe.trials",
        description="Report on a commissioning trial: why the station stopped.")
    ap.add_argument("directory", nargs="?", default="logs/trials",
                    help="the telemetry.trials directory (default: logs/trials)")
    ap.add_argument("--confirm", type=float, default=0.4,
                    help="ppe.confirm_sec.violation, in seconds (default: 0.4)")
    args = ap.parse_args(argv)
    print(report(args.directory, args.confirm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
