"""Keep the station running: restart it after a crash, never after a clean exit.

    PPEStation.exe --supervise        # what the installed autostart runs
    python main.py --supervise        # the same from source

The supervisor is a small parent process. It starts the station as a child
with the same arguments (minus --supervise), waits, and starts it again if
it died with a non-zero exit code — a Python exception, a native crash, the
process being killed. A zero exit is someone closing the window on purpose,
and that is respected: the station stays closed until it is started again.

Restarting is safe for the machine because of what the station does on
start-up, not because of anything here: it blanks every relay channel when
it connects, and the motor latch only ever sets on a fresh press of the
start button with compliant PPE. A crash therefore always leaves the motor
stopped, and a restart never starts it.

Back-off doubles from 2 s to a 60 s ceiling, so a station that cannot start
(a broken config, a missing model) retries without spinning the CPU; a run
that lasted five minutes resets it, so one crash a week restarts in 2 s.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .paths import frozen

log = logging.getLogger("ppe.supervise")

FIRST_DELAY = 2.0
MAX_DELAY = 60.0
STABLE_RUN = 300.0   # seconds of uptime after which a crash counts as isolated


def child_command(argv: Sequence[str]) -> list[str]:
    """The command line for the station itself: this program, minus --supervise."""
    args = [a for a in argv if a != "--supervise"]
    if frozen():
        return [sys.executable, *args]
    main = Path(__file__).resolve().parent.parent / "main.py"
    return [sys.executable, str(main), *args]


def supervise(
    argv: Sequence[str],
    *,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run the station until it exits cleanly; returns 0 then."""
    cmd = child_command(argv)
    delay = FIRST_DELAY
    while True:
        started = clock()
        log.info("starting station: %s", subprocess.list2cmdline(cmd))
        proc = popen(cmd)
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            return 0
        if code == 0:
            log.info("station closed normally; supervisor exiting")
            return 0
        if clock() - started >= STABLE_RUN:
            delay = FIRST_DELAY
        log.warning("station exited with code %s after %.0f s; restarting in %.0f s",
                    code, clock() - started, delay)
        sleep(delay)
        delay = min(delay * 2, MAX_DELAY)
