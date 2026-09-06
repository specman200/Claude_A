"""make_detector() just routes on model.arch — Detector and RFDetrDetector
are each tested for real elsewhere; this only pins that the right one gets
built, without paying either one's real construction cost (loading actual
weights, importing rfdetr) to prove it.
"""

from __future__ import annotations

from ppe.config import ClassCfg, ModelCfg, PPECfg


def ppe():
    return PPECfg(classes=[ClassCfg("glove")])


def test_yolo_arch_builds_a_detector(monkeypatch):
    import ppe.detector as detector_mod

    built = {}

    class SentinelDetector:
        def __init__(self, cfg, ppe_cfg):
            built["cfg"], built["ppe"] = cfg, ppe_cfg

    monkeypatch.setattr(detector_mod, "Detector", SentinelDetector)
    cfg = ModelCfg(arch="yolo")
    result = detector_mod.make_detector(cfg, ppe())
    assert isinstance(result, SentinelDetector)
    assert built["cfg"] is cfg


def test_rfdetr_arch_builds_an_rfdetr_detector(monkeypatch):
    import ppe.detector as detector_mod
    import ppe.rfdetr_detector as rfdetr_mod

    built = {}

    class SentinelRFDetr:
        def __init__(self, cfg, ppe_cfg):
            built["cfg"], built["ppe"] = cfg, ppe_cfg

    monkeypatch.setattr(rfdetr_mod, "RFDetrDetector", SentinelRFDetr)
    cfg = ModelCfg(arch="rfdetr")
    result = detector_mod.make_detector(cfg, ppe())
    assert isinstance(result, SentinelRFDetr)
    assert built["cfg"] is cfg
