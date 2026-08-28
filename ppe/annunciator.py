"""Spoken prompt for a worker who is not looking at the tower light.

A lamp only works on someone facing it. An audible prompt reaches a worker
turned away from the tower, which on a real floor is most of them.

The timing matters as much as the sound. Nagging the instant PPE is missing
trains people to ignore it, so there is a grace period first — long enough to
finish putting a glove on — and then a repeat interval, so a sustained
violation keeps prompting instead of announcing itself once and going quiet.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .latency import now
from .tower import Status

log = logging.getLogger(__name__)


class Annunciator:
    """Plays an audio prompt while a violation stands.

    Silent by default: with no audio file configured this does nothing at all,
    so a station with no speaker is not a special case anywhere else.
    """

    def __init__(
        self,
        path: str | Path | None,
        grace_sec: float = 3.0,
        repeat_sec: float = 3.0,
        base: Path | None = None,
        mute: bool = False,
    ) -> None:
        self.grace = max(0.0, grace_sec)
        self.repeat = max(0.5, repeat_sec)
        # Muted still runs the whole timing model — it just does not make a
        # sound. Skipping the evaluation instead would hide the behaviour you
        # opened the debug view to look at.
        self.mute = mute
        self.suppressed = 0
        self._sound = None
        self._since: float | None = None   # when the current violation began
        self._next: float = 0.0            # when the next prompt is due
        self.plays = 0

        resolved = self._resolve(path, base)
        if resolved is None:
            return
        try:
            import pygame

            pygame.mixer.init()
            pygame.mixer.music.load(str(resolved))
            self._sound = pygame.mixer.music
            log.info("annunciator: %s (grace %.1fs, repeat %.1fs)",
                     resolved.name, self.grace, self.repeat)
        except Exception as exc:  # noqa: BLE001 — a missing speaker is not fatal
            log.warning("annunciator disabled: %s", exc)
            self._sound = None

    @staticmethod
    def _resolve(path: str | Path | None, base: Path | None) -> Path | None:
        if not path:
            return None
        p = Path(path)
        if not p.is_absolute() and base is not None:
            p = base / p
        if not p.is_file():
            log.warning("annunciator: no such audio file %s", p)
            return None
        return p

    @property
    def enabled(self) -> bool:
        return self._sound is not None

    def update(self, status: Status, t: float | None = None) -> bool:
        """Follow the station status; returns True if a prompt was started."""
        t = now() if t is None else t

        if status is not Status.VIOLATION:
            # Compliant, empty, or unable to judge — nothing to announce, and
            # the grace period resets so the next violation gets one too.
            self._since = None
            return False

        if self._since is None:
            self._since = t
            self._next = t + self.grace
            return False

        if t < self._next:
            return False

        self._next = t + self.repeat
        return self._play()

    def due_in(self, t: float | None = None) -> float | None:
        """Seconds until the next prompt, or None when nothing is pending."""
        if self._since is None:
            return None
        return max(0.0, self._next - (now() if t is None else t))

    def _play(self) -> bool:
        if self.mute:
            self.suppressed += 1
            return False
        if self._sound is None:
            return False
        try:
            if self._sound.get_busy():
                return False  # already speaking; do not stack prompts
            self._sound.play()
        except Exception as exc:  # noqa: BLE001
            log.warning("annunciator playback failed: %s", exc)
            return False
        self.plays += 1
        return True

    def silence(self) -> None:
        try:
            if self._sound is not None:
                self._sound.stop()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        self.silence()
        try:
            import pygame

            pygame.mixer.quit()
        except Exception:  # noqa: BLE001
            pass
        self._sound = None
