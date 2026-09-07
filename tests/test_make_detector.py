"""make_detector() routes on model.arch, and then rfdetr_backend() routes
on the weights extension. Each backend is tested for real elsewhere; this
only pins that the right one gets built, without paying any of their real
construction costs (loading actual weights, importing rfdetr, compiling a
graph) to prove it.
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


def test_an_rfdetr_graph_builds_the_onnx_backend(monkeypatch):
    """A .onnx or .xml runs through OpenVINO/ONNX Runtime, not torch — and
    the extension is the only thing that says so."""
    import ppe.detector as detector_mod
    import ppe.rfdetr_onnx as onnx_mod

    built = {}

    class SentinelOnnx:
        def __init__(self, cfg, ppe_cfg):
            built["cfg"] = cfg

    monkeypatch.setattr(onnx_mod, "RFDetrOnnx", SentinelOnnx)
    for weights in ("best.onnx", "best.xml", "MODEL.ONNX"):
        built.clear()
        cfg = ModelCfg(arch="rfdetr", weights=weights)
        assert isinstance(detector_mod.make_detector(cfg, ppe()), SentinelOnnx), weights


def test_an_rfdetr_checkpoint_still_builds_the_torch_backend(monkeypatch):
    import ppe.detector as detector_mod
    import ppe.rfdetr_detector as rfdetr_mod

    class SentinelPth:
        def __init__(self, cfg, ppe_cfg):
            pass

    monkeypatch.setattr(rfdetr_mod, "RFDetrDetector", SentinelPth)
    cfg = ModelCfg(arch="rfdetr", weights="checkpoint_best_ema.pth")
    assert isinstance(detector_mod.make_detector(cfg, ppe()), SentinelPth)


def test_a_yolo_openvino_directory_is_not_mistaken_for_an_rfdetr_graph():
    """The YOLO path points at a *directory* of IR; only arch decides."""
    from ppe.config import ModelCfg as MC

    assert MC(weights="models/ppe-yolo11s_openvino_model/").is_graph is False
    assert MC(weights="models/ppe-yolo11s_openvino_model/model.xml").is_graph is True
    assert MC(weights="best.pth").is_graph is False
