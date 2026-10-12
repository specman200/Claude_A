"""Per-machine settings written at install time.

Every station runs the same program and the same config, except for the
address of its own relay module (see the IP plan). The installer calls

    PPEStation-console.exe --set-relay-ip 192.168.1.56

once, on a fresh install, and this edits that one value in place.

It is a line edit, not a YAML load-and-dump: config.yaml is the station's
documentation as much as its settings, and a round trip through a YAML
library would strip every comment from it.
"""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path

_SECTION = re.compile(r"^tower:\s*(#.*)?$")
_HOST = re.compile(r"^(\s+host:\s*)(\S+)(.*)$")


def set_relay_host(path: str | Path, ip: str) -> str:
    """Set tower.host in the config at ``path``; returns the previous value.

    Raises ValueError for anything that is not a plain IPv4 address, or if
    the config has no ``tower:`` block with a ``host:`` line to change.
    """
    ipaddress.IPv4Address(ip)   # raises ValueError on "192.168.1", hostnames, etc.
    path = Path(path)
    # Bytes, not text mode: the file keeps its own line endings on Windows.
    lines = path.read_bytes().decode("utf-8").splitlines(keepends=True)
    inside = False
    for i, line in enumerate(lines):
        if _SECTION.match(line):
            inside = True
            continue
        if inside and line.strip() and not line[0].isspace() and not line.lstrip().startswith("#"):
            break   # the next top-level key: tower: ended without a host line
        m = _HOST.match(line.rstrip("\r\n")) if inside else None
        if m:
            old = m.group(2)
            lines[i] = f"{m.group(1)}{ip}{m.group(3)}{_eol(line)}"
            path.write_bytes("".join(lines).encode("utf-8"))
            return old
    raise ValueError(f"{path}: no 'host:' line under 'tower:' to set")


def _eol(line: str) -> str:
    return line[len(line.rstrip("\r\n")):]
