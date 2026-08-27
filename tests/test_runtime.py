"""CPU tuning: the model gets the cores, everything else keeps out of the way."""

import cv2
import pytest

from ppe.config import Config, ModelCfg
from ppe.runtime import _ENV, apply_torch, configure, cores


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in _ENV:
        monkeypatch.delenv(var, raising=False)
    before = cv2.getNumThreads()
    yield
    cv2.setNumThreads(before)


def test_cores_is_at_least_one():
    assert cores() >= 1


def test_configure_gives_the_maths_libraries_a_thread_budget(monkeypatch):
    assert configure(3) == 3
    import os

    for var in _ENV:
        assert os.environ[var] == "3"


def test_capture_decode_is_pinned_to_one_thread():
    """Two camera threads each spawning a pool would starve inference."""
    cv2.setNumThreads(8)
    configure(2)
    assert cv2.getNumThreads() == 1


def test_zero_means_every_core():
    assert configure(0) == cores()


def test_an_operator_set_thread_budget_is_never_overridden(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    configure(4)
    import os

    assert os.environ["OMP_NUM_THREADS"] == "1"   # left alone
    assert os.environ["MKL_NUM_THREADS"] == "4"   # the rest still set


def test_apply_torch_is_safe_to_call():
    apply_torch(2)
    torch = pytest.importorskip("torch")
    assert torch.get_num_threads() == 2
    apply_torch(cores())


# -- batching policy -------------------------------------------------------


@pytest.mark.parametrize(
    "batch,device,expected",
    [
        ("auto", "cpu", False),      # batching costs ~2x on CPU
        ("auto", "cuda:0", True),    # and pays on a GPU
        ("auto", "mps", False),
        (True, "cpu", True),         # explicit wins
        (False, "cuda:0", False),
    ],
)
def test_batching_policy(batch, device, expected):
    assert ModelCfg(batch=batch).batches(device) is expected


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda c: setattr(c.model, "threads", -1), "threads must be >= 0"),
        (lambda c: setattr(c.model, "batch", "sometimes"), "must be true, false or 'auto'"),
    ],
)
def test_validate_rejects_bad_cpu_settings(mutate, message):
    cfg = Config.load("config.yaml")
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        cfg.validate()
