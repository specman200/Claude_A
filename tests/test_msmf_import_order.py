"""The env var that fixes MSMF's hardware-transform stall must be set
before `import cv2` runs anywhere in the process, not merely before any
camera is opened.

Confirmed on real hardware: setting it in-process, after cv2 was already
imported, left os.environ correctly holding the value while the native
stall still happened — this build of OpenCV evidently caches the setting
at cv2's own load time, not lazily at first MSMF use. The only reliable
fix is an environment variable that exists before the Python process
itself starts, or — the best this codebase can do on its own — setting it
before this file's own `import cv2`, and before any other module gets a
chance to import cv2 first (Python caches the loaded module, so a second
`import cv2` elsewhere is a no-op that does not re-trigger whatever
native initialisation already happened).

None of this is observable by running the code here — it needs a Windows
process with a real MSMF-backed camera. What IS checkable, and what
actually failed before, is the textual order: the setdefault line must
precede `import cv2` in ppe/capture.py, and ppe/camcheck.py must import
.capture (which carries that setdefault) before it imports cv2 itself.
"""

from pathlib import Path

VAR = "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"


def _code_line_of(text: str, prefix: str) -> int:
    """The first line, ignoring comments, that starts with ``prefix`` once
    stripped of leading whitespace — so prose that happens to mention the
    same text (this file's own comments do, explaining exactly this
    ordering) is not mistaken for the code being checked."""
    for i, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith(prefix) and not stripped.startswith("#"):
            return i
    raise AssertionError(f"no code line starting with {prefix!r}")


def test_capture_sets_the_env_var_before_importing_cv2():
    text = Path("ppe/capture.py").read_text()
    setdefault_line = _code_line_of(text, f'os.environ.setdefault("{VAR}"')
    cv2_import_line = _code_line_of(text, "import cv2")
    assert setdefault_line < cv2_import_line, (
        "the setdefault must precede `import cv2` — after it is too late, "
        "even though os.environ ends up holding the right value either way"
    )


def test_camcheck_imports_capture_before_importing_cv2_itself():
    """camcheck.py has its own `import cv2` — if that ran first, .capture's
    setdefault (however early inside .capture) would already be too late,
    the same bug independently, in a different file."""
    text = Path("ppe/camcheck.py").read_text()
    capture_import_line = _code_line_of(text, "from .capture import")
    cv2_import_line = _code_line_of(text, "import cv2")
    assert capture_import_line < cv2_import_line
