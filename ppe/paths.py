"""Where things live: a source checkout, or an installed (frozen) build.

Run from source, everything sits in one tree and the working directory is
usually that tree, so relative paths in config.yaml just work. An installed
build splits the two kinds of file apart, because Windows does:

    app_dir()   C:\\Program Files\\PPE Station\\_internal   read-only, replaced
                models\\, assets\\, config.yaml (default)    on every upgrade
    data_dir()  C:\\ProgramData\\PPE Station                 writable, survives
                config.yaml (this machine's), logs\\,         upgrades
                datasets\\

`resolve()` lets the same config.yaml work in both: a relative path is
tried as given first (the old behaviour, unchanged), then next to the
config file, then in the program's own folder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "PPE Station"


def frozen() -> bool:
    """True inside a PyInstaller build."""
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """Read-only files shipped with the program."""
    if frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Writable, per-machine files: config, logs, captures.

    From source this is the working directory, as it always was. Installed,
    it is %ProgramData%\\PPE Station — the same for every Windows user who
    logs on to the station, and not wiped by an upgrade.
    """
    override = os.environ.get("PPE_STATION_DATA")
    if override:
        # Lets a build machine or CI self-test the frozen program without
        # leaving a config in the real ProgramData.
        return Path(override)
    if frozen():
        root = os.environ.get("PROGRAMDATA") or str(Path.home())
        return Path(root) / APP_NAME
    return Path.cwd()


def resolve(path: str, base: Path | None = None) -> str:
    """A path from the config, made to work wherever the program runs.

    Absolute paths, and relative ones that exist from the working directory,
    come back untouched — exactly what the code did before this existed.
    Otherwise the first of ``base/path`` and ``app_dir()/path`` that exists
    wins. If none does, the original string comes back, so a bare model name
    like ``yolo11s.pt`` still reaches ultralytics as it was written.

    Returns a string, keeping a trailing separator: ultralytics reads
    ``…_openvino_model/`` as a directory export.
    """
    if not path:
        return path
    p = Path(path)
    if p.is_absolute() or p.exists():
        return path
    trail = os.sep if path.endswith(("/", "\\")) else ""
    for root in (base, app_dir()):
        if root is not None and (root / p).exists():
            return str(root / p) + trail
    return path


class InstanceLock:
    """One station per PC: a second copy would fight the first for the
    cameras and the relay board, each blanking the other's outputs.

    An OS-level lock on a file in the data directory, not a PID file: the
    operating system releases it the moment the holder dies, so a crashed
    station never blocks its own restart.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or data_dir() / "logs" / "station.lock"
        self._fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")  # noqa: SIM115 — held open for the process lifetime
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is not None:
            self._fh.close()   # closing the handle drops the lock on both platforms
            self._fh = None
