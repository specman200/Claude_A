"""RFDetrOnnx against a real ONNX graph, run by real OpenVINO and real
ONNX Runtime — not a stubbed runtime.

There is no trained RF-DETR checkpoint for this station's classes yet, but
the *shape* of an rfdetr export is documented and small enough to build:
one NCHW input, and two raw outputs — boxes as normalized cxcywh and
un-activated logits. So these construct that graph with known constant
outputs and check what the backend makes of it, which exercises the parts
that were only ever going to be wrong from guessing: which output is
which, the sigmoid, the per-class floors, and cxcywh -> source-pixel xyxy.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppe.config import ClassCfg, ModelCfg, PPECfg

onnx = pytest.importorskip("onnx")
NAMES = ["person", "glove", "helmet"]


def build_graph(path, boxes, logits, res=64, boxes_first=True, names=("dets", "labels")):
    """An ONNX with RF-DETR's signature and constant outputs.

    The input is consumed (multiplied by zero and added on) purely so the
    graph really depends on it — an unused parameter is liable to be
    optimised away, and then there would be no input shape to read back.
    """
    from onnx import TensorProto, helper, numpy_helper

    b = numpy_helper.from_array(np.asarray(boxes, np.float32), "boxes_const")
    lg = numpy_helper.from_array(np.asarray(logits, np.float32), "logits_const")
    zero = numpy_helper.from_array(np.zeros((), np.float32), "zero_const")
    nodes = [
        helper.make_node("ReduceMean", ["input"], ["mean"], keepdims=0),
        helper.make_node("Mul", ["mean", "zero_const"], ["nil"]),
        helper.make_node("Add", ["boxes_const", "nil"], ["boxes_out"]),
        helper.make_node("Add", ["logits_const", "nil"], ["logits_out"]),
    ]
    outs = [
        helper.make_tensor_value_info(names[0], TensorProto.FLOAT, b.dims),
        helper.make_tensor_value_info(names[1], TensorProto.FLOAT, lg.dims),
    ]
    renames = [("boxes_out", names[0]), ("logits_out", names[1])]
    if not boxes_first:
        outs = outs[::-1]
        renames = [("boxes_out", names[1]), ("logits_out", names[0])]
        outs = [
            helper.make_tensor_value_info(names[0], TensorProto.FLOAT, lg.dims),
            helper.make_tensor_value_info(names[1], TensorProto.FLOAT, b.dims),
        ]
    for src, dst in renames:
        nodes.append(helper.make_node("Identity", [src], [dst]))

    graph = helper.make_graph(
        nodes, "rfdetrish",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, res, res])],
        outs, initializer=[b, lg, zero],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def cfg(weights, **kw):
    kw.setdefault("arch", "rfdetr")
    kw.setdefault("conf", 0.35)
    kw.setdefault("class_names", list(NAMES))
    return ModelCfg(weights=weights, **kw)


def ppe(*names, conf=None):
    return PPECfg(classes=[ClassCfg(n, conf=conf) for n in names])


def one_box(cx=0.5, cy=0.5, w=0.5, h=0.25, logits=(-9.0, 4.0, -9.0)):
    """A single query: one box, and logits over the three NAMES."""
    return np.array([[[cx, cy, w, h]]], np.float32), np.array([[list(logits)]], np.float32)


def detector(tmp_path, boxes, logits, ppe_cfg=None, name="m.onnx", **kw):
    from ppe.rfdetr_onnx import RFDetrOnnx

    path = build_graph(tmp_path / name, boxes, logits, **kw)
    return RFDetrOnnx(cfg(path), ppe_cfg or ppe(*NAMES))


# -- it really loads and runs, through the real runtimes -------------------


def test_a_real_graph_loads_and_reports_its_own_input_size(tmp_path):
    det = detector(tmp_path, *one_box())
    assert det.runtime.startswith("openvino")
    assert (det.height, det.width) == (64, 64)
    assert det.batches is False and det.batch_size == 1


def test_onnxruntime_runs_the_same_graph_to_the_same_answer(tmp_path):
    """The fallback path has to agree with OpenVINO, or a station that fell
    back would quietly detect different things."""
    from ppe.rfdetr_onnx import RFDetrOnnx

    boxes, logits = one_box()
    path = build_graph(tmp_path / "m.onnx", boxes, logits)

    ov_det = RFDetrOnnx(cfg(path), ppe(*NAMES))
    frame = np.zeros((480, 640, 3), np.uint8)
    ov_out, _ = ov_det.detect([frame])

    ort_det = RFDetrOnnx(cfg(path), ppe(*NAMES))
    ort_det.runtime = "onnxruntime"  # force the fallback branch
    import onnxruntime as ort
    ort_det._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ort_det._in_name = ort_det._session.get_inputs()[0].name
    ort_out, _ = ort_det.detect([frame])

    assert [(d.name, round(d.conf, 5), tuple(round(v, 3) for v in d.xyxy))
            for d in ov_out[0]] == \
           [(d.name, round(d.conf, 5), tuple(round(v, 3) for v in d.xyxy))
            for d in ort_out[0]]


# -- the decode ------------------------------------------------------------


def test_normalized_cxcywh_becomes_source_pixel_xyxy(tmp_path):
    """The graph sees a 64x64 tensor but the boxes must come back in the
    coordinates of the frame the camera actually delivered."""
    det = detector(tmp_path, *one_box(cx=0.5, cy=0.5, w=0.5, h=0.25))
    dets, _ = det.detect([np.zeros((480, 640, 3), np.uint8)])

    assert len(dets[0]) == 1
    x1, y1, x2, y2 = dets[0][0].xyxy
    assert (x1, x2) == pytest.approx((160.0, 480.0))  # 0.25..0.75 of 640
    assert (y1, y2) == pytest.approx((180.0, 300.0))  # 0.375..0.625 of 480


def test_the_logit_goes_through_a_sigmoid(tmp_path):
    det = detector(tmp_path, *one_box(logits=(-9.0, 2.0, -9.0)))
    dets, _ = det.detect([np.zeros((64, 64, 3), np.uint8)])
    assert dets[0][0].conf == pytest.approx(1 / (1 + np.exp(-2.0)), rel=1e-5)
    assert dets[0][0].name == "glove"  # the argmax class, id 1


def test_a_logit_below_the_floor_is_dropped(tmp_path):
    """sigmoid(-1) is 0.27, under the 0.35 global floor."""
    det = detector(tmp_path, *one_box(logits=(-9.0, -1.0, -9.0)))
    assert det.detect([np.zeros((64, 64, 3), np.uint8)])[0][0] == []


def test_per_class_floors_are_applied_over_the_global_one(tmp_path):
    boxes, logits = one_box(logits=(-9.0, 1.0, -9.0))  # sigmoid(1) = 0.73
    det = detector(tmp_path, boxes, logits, ppe_cfg=ppe("glove", conf=0.9))
    assert det.detect([np.zeros((64, 64, 3), np.uint8)])[0][0] == []

    det = detector(tmp_path, boxes, logits, ppe_cfg=ppe("glove", conf=0.5), name="b.onnx")
    assert len(det.detect([np.zeros((64, 64, 3), np.uint8)])[0][0]) == 1


def test_a_class_the_station_does_not_check_for_is_dropped(tmp_path):
    """No NMS to push a class filter into, so it happens on the way out."""
    boxes, logits = one_box(logits=(-9.0, -9.0, 5.0))  # 'helmet' fires
    det = detector(tmp_path, boxes, logits, ppe_cfg=ppe("person", "glove"))
    assert det.detect([np.zeros((64, 64, 3), np.uint8)])[0][0] == []


def test_every_query_above_the_floor_is_returned(tmp_path):
    boxes = np.array([[[0.25, 0.5, 0.1, 0.1], [0.75, 0.5, 0.1, 0.1]]], np.float32)
    logits = np.array([[[-9.0, 3.0, -9.0], [-9.0, 3.0, -9.0]]], np.float32)
    det = detector(tmp_path, boxes, logits)
    assert len(det.detect([np.zeros((64, 64, 3), np.uint8)])[0][0]) == 2


def test_stage_timings_are_reported(tmp_path):
    det = detector(tmp_path, *one_box())
    _dets, timings = det.detect([np.zeros((64, 64, 3), np.uint8)])
    assert set(timings) == {"preprocess", "inference", "postprocess"}


# -- telling the two outputs apart ----------------------------------------


def test_the_boxes_are_found_whichever_output_they_come_out_of(tmp_path):
    """Output names differ between exporter versions; the shape does not."""
    boxes, logits = one_box()
    swapped = detector(tmp_path, boxes, logits, boxes_first=False,
                       names=("pred_logits", "pred_boxes"))
    dets, _ = swapped.detect([np.zeros((480, 640, 3), np.uint8)])
    assert len(dets[0]) == 1
    assert dets[0][0].xyxy[0] == pytest.approx(160.0)


def test_a_four_class_model_is_refused_rather_than_guessed_at(tmp_path):
    """With 4 classes both outputs end in 4 and shape cannot decide. Saying
    so beats a coin flip that mislabels or mislocates everything."""
    from ppe.rfdetr_onnx import RFDetrOnnx

    boxes = np.zeros((1, 1, 4), np.float32)
    logits = np.zeros((1, 1, 4), np.float32)
    path = build_graph(tmp_path / "four.onnx", boxes, logits)
    with pytest.raises(RuntimeError, match="cannot tell the boxes from the logits"):
        RFDetrOnnx(cfg(path, class_names=["a", "b", "c", "d"]), ppe("a"))


# -- class names -----------------------------------------------------------


def test_missing_class_names_is_a_clear_error_not_a_silent_mislabel(tmp_path):
    """This is the failure that produced a DEGRADED station with
    'class0..class998' when an RF-DETR ONNX was fed to the YOLO backend."""
    from ppe.rfdetr_onnx import RFDetrOnnx

    path = build_graph(tmp_path / "m.onnx", *one_box())
    with pytest.raises(RuntimeError, match="set model.class_names"):
        RFDetrOnnx(cfg(path, class_names=[]), ppe(*NAMES))


def test_a_class_name_list_of_the_wrong_length_is_refused(tmp_path):
    from ppe.rfdetr_onnx import RFDetrOnnx

    path = build_graph(tmp_path / "m.onnx", *one_box())
    with pytest.raises(RuntimeError, match="predicts 3 classes"):
        RFDetrOnnx(cfg(path, class_names=["only", "two"]), ppe("only"))


def test_a_configured_class_the_model_lacks_is_reported(tmp_path):
    det = detector(tmp_path, *one_box(), ppe_cfg=ppe("glove", "hi_vis"))
    assert det.missing == ["hi_vis"]


# -- preprocessing ---------------------------------------------------------


def test_frames_are_rgb_imagenet_normalized_and_nchw(tmp_path):
    """Wrong channel order or a missing normalize is a silent accuracy loss,
    not a crash, so check the tensor that actually reaches the graph."""
    det = detector(tmp_path, *one_box())
    seen = []
    det._run = lambda t: (seen.append(t.copy()), [np.zeros((1, 1, 4), np.float32),
                                                  np.zeros((1, 1, 3), np.float32)])[1]

    bgr = np.zeros((480, 640, 3), np.uint8)
    bgr[..., 0], bgr[..., 1], bgr[..., 2] = 255, 0, 0  # pure blue in BGR
    det.detect([bgr])

    x = seen[-1]
    assert x.shape == (1, 3, 64, 64) and x.dtype == np.float32
    # Blue is channel 2 after the swap: R and G at 0, B at 255/255.
    assert x[0, 0].mean() == pytest.approx((0.0 - 0.485) / 0.229, rel=1e-4)
    assert x[0, 1].mean() == pytest.approx((0.0 - 0.456) / 0.224, rel=1e-4)
    assert x[0, 2].mean() == pytest.approx((1.0 - 0.406) / 0.225, rel=1e-4)


def test_a_frame_already_the_right_size_is_not_resized(tmp_path):
    det = detector(tmp_path, *one_box())
    x = det._preprocess(np.zeros((64, 64, 3), np.uint8))
    assert x.shape == (1, 3, 64, 64)
