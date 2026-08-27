"""CPU tuning: stop capture, inference and the UI fighting over the same cores.

On a CPU box every stage competes for the same few cores. OpenCV defaults to a
thread per core *per camera thread*, and the maths libraries behind the model do
the same, so a 4-core machine can end up with a dozen runnable threads and the
model running several times slower than it does in isolation.
"""

from __future__ import annotations

import logging
import os

import cv2

log = logging.getLogger(__name__)

# Read once when the maths libraries initialise, so they must be set before
# torch / onnxruntime / openvino are imported.
_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def cores() -> int:
    """Cores this process may actually use, honouring cgroup/affinity limits."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not Linux
        return max(1, os.cpu_count() or 1)


def configure(threads: int = 0) -> int:
    """Give the model the cores and keep everything else out of its way.

    Call before importing torch. ``threads`` of 0 means "use every core".
    Returns the thread count chosen.
    """
    n = threads if threads > 0 else cores()
    for var in _ENV:
        # Never override an operator who set these deliberately.
        os.environ.setdefault(var, str(n))
    # One decode thread per camera: the capture threads already run in
    # parallel, so a pool inside each of them only steals from inference.
    cv2.setNumThreads(1)
    log.info("cpu: %d inference threads, single-threaded capture", n)
    return n


def apply_torch(threads: int) -> None:
    """Pin torch's intra-op pool once it is imported."""
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(max(1, threads))
