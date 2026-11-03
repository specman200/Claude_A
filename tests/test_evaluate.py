"""The common-dataset model comparison: metrics, decisions, dataset loading, report filling."""

import cv2
import numpy as np
import pytest

from ppe import evaluate as ev
from ppe.config import Config
from ppe.detector import Detection

PERSON = (0.0, 0.0, 400.0, 800.0)


def box(name, i=0, conf=0.9):
    """A small box inside the person, offset so boxes of one class do not overlap."""
    x = 20 + 70 * i
    return Detection(name, conf, (x, 40.0, x + 50, 90.0))


def compliant(conf=0.99):
    """Everything the shipped config requires, on one person."""
    return [
        Detection("person", conf, PERSON),
        box("headnet", 0, conf),
        box("Safetyglasses", 1, conf),
        box("Mask", 2, conf),
        box("Gloves", 0, conf),
        box("Gloves", 1, conf),
        box("sleeves", 2, conf),
        box("sleeves", 3, conf),
    ]


@pytest.fixture
def cfg():
    return Config.load("config.yaml")


# -- detection metrics -----------------------------------------------------------------


# The COCO / Ultralytics 101-point interpolation tops out at 0.995, not 1.0 (the last half
# step to recall 1.0 interpolates towards the appended zero). Matching it keeps these numbers
# on the same scale as the training metrics in the report.
PERFECT = pytest.approx(0.995, abs=1e-3)


def test_perfect_predictions_score_the_coco_maximum():
    truth = [Detection("Mask", 1.0, (0, 0, 10, 10)), Detection("Mask", 1.0, (20, 20, 30, 30))]
    tp = ev.match(truth, truth)
    assert tp.all()
    [s] = ev.score_classes(["Mask"], {"Mask"}, [tp], [truth], [truth])
    assert s.ap50 == PERFECT and s.ap == PERFECT
    assert s.recall == pytest.approx(1.0) and s.precision == pytest.approx(1.0)


def test_a_missed_object_halves_recall():
    truth = [Detection("Mask", 1.0, (0, 0, 10, 10)), Detection("Mask", 1.0, (20, 20, 30, 30))]
    preds = truth[:1]
    [s] = ev.score_classes(["Mask"], {"Mask"}, [ev.match(preds, truth)], [preds], [truth])
    assert s.recall == pytest.approx(0.5)
    # Ultralytics interpolates linearly from the last recall point to (1, 0), so a model that
    # stops at recall 0.5 scores 0.75 rather than COCO's step-function 0.5. Same as training.
    assert s.ap50 == pytest.approx(0.75, abs=0.01)


def test_a_duplicate_box_is_a_false_positive():
    truth = [Detection("Mask", 1.0, (0, 0, 10, 10))]
    preds = [Detection("Mask", 0.9, (0, 0, 10, 10)), Detection("Mask", 0.8, (0, 0, 10, 10))]
    tp = ev.match(preds, truth)
    assert tp[:, 0].tolist() == [True, False]


def test_a_box_of_the_wrong_class_never_matches():
    truth = [Detection("Gloves", 1.0, (0, 0, 10, 10))]
    preds = [Detection("sleeves", 0.9, (0, 0, 10, 10))]
    assert not ev.match(preds, truth).any()


def test_loose_boxes_count_at_iou_05_but_not_at_095():
    truth = [Detection("Mask", 1.0, (0, 0, 10, 10))]
    preds = [Detection("Mask", 0.9, (0, 0, 10, 13))]  # IoU ~0.77
    tp = ev.match(preds, truth)
    assert tp[0, 0] and not tp[0, -1]


def test_a_class_the_model_lacks_is_not_scored_against_it():
    truth = [Detection("Apron", 1.0, (0, 0, 10, 10)), Detection("Mask", 1.0, (20, 20, 30, 30))]
    preds = [truth[1]]
    scores = {
        s.name: s
        for s in ev.score_classes(
            ["Mask", "Apron"], {"Mask"}, [ev.match(preds, truth)], [preds], [truth]
        )
    }
    assert scores["Apron"].instances == 1 and not scores["Apron"].detectable
    assert scores["Apron"].ap50 != scores["Apron"].ap50  # NaN: not a zero
    assert scores["Mask"].ap50 == PERFECT


# -- decisions ------------------------------------------------------------------------------


def test_the_labels_of_a_compliant_worker_read_ok(cfg):
    assert ev.verdict(compliant(), cfg) == ev.OK


def test_nobody_in_view_is_standby(cfg):
    assert ev.verdict([box("Mask")], cfg) == ev.STANDBY


def test_a_wrong_sleeve_is_a_violation_however_complete_the_rest(cfg):
    assert ev.verdict(compliant() + [box("Wrong Sleeve", 4)], cfg) == ev.VIOLATION


def test_floors_drop_what_the_station_would_drop(cfg):
    weak = [Detection("Wrong Sleeve", 0.5, (0, 0, 10, 10))]  # Wrong Sleeve floor is 0.90
    assert ev.with_floors(weak, cfg) == []


def test_missed_gloves_are_a_false_stop_and_invented_gloves_a_false_permit(cfg):
    d = ev.Decisions()
    no_gloves = [x for x in compliant() if x.name != "Gloves"]
    d.add(ev.verdict(compliant(), cfg), ev.verdict(no_gloves, cfg))  # model missed the gloves
    d.add(ev.verdict(no_gloves, cfg), ev.verdict(compliant(), cfg))  # model invented them
    assert (d.false_stop, d.false_permit, d.agree) == (1, 1, 0)
    assert d.false_stop_rate == 1.0 and d.false_permit_rate == 1.0


def test_a_model_without_an_apron_class_is_judged_on_sleeves(cfg):
    sleeveless = [x for x in compliant() if x.name != "sleeves"] + [box("Apron", 4)]
    assert ev.verdict(sleeveless, cfg) == ev.OK
    # the same image, judged by a model that cannot see aprons
    no_apron = [x for x in sleeveless if x.name != "Apron"]
    assert ev.verdict(no_apron, cfg, unavailable=["Apron"]) == ev.VIOLATION


# -- dataset loading -------------------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path):
    """Two test images in YOLO layout, with names in best-apron's order."""
    root = tmp_path / "ppe"
    for name, lines in (("a", ["0 0.5 0.5 0.5 1.0", "3 0.25 0.1 0.1 0.05"]), ("b", [])):
        (root / "images" / "test").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "test").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(root / "images" / "test" / f"{name}.jpg"), np.zeros((800, 800, 3), np.uint8)
        )
        (root / "labels" / "test" / f"{name}.txt").write_text("\n".join(lines))
    data = tmp_path / "data.yaml"
    data.write_text(
        "path: C:\\somewhere\\else\ntest: images/test\nnames:\n"
        "  0: person\n  1: headnet\n  2: Gloves\n  3: Mask\n"
    )
    return data, root


def test_root_overrides_the_yaml_path_and_labels_are_read_in_pixels(dataset):
    data, root = dataset
    ds = ev.load_dataset(data, root=root)
    assert ds.names == ["person", "headnet", "Gloves", "Mask"]
    a = next(s for s in ds.samples if s.image.stem == "a")
    assert [t.name for t in a.truth] == ["person", "Mask"]
    assert a.truth[0].xyxy == pytest.approx((200, 0, 600, 800))
    assert next(s for s in ds.samples if s.image.stem == "b").truth == []


def test_a_missing_split_says_which_splits_exist(dataset):
    data, root = dataset
    with pytest.raises(KeyError, match="test"):
        ev.load_dataset(data, root=root, split="val")


# -- one model end to end, and the report ------------------------------------------------------


def test_a_fake_model_is_scored_by_name_whatever_its_class_order(dataset, cfg):
    data, root = dataset
    ds = ev.load_dataset(data, root=root)

    def predict(image):
        truth = next(s for s in ds.samples if s.image == image).truth
        return [Detection(t.name, 0.95, t.xyxy) for t in truth], 5.0

    r = ev.evaluate_model("fake", "fake.pt", ["Mask", "person"], ds, cfg, predict)
    scores = {s.name: s for s in r.scores}
    assert scores["Mask"].ap50 == PERFECT and scores["person"].ap50 == PERFECT
    assert not scores["Gloves"].detectable
    assert r.ms == 5.0 and r.decisions.images == 2


def test_tables_and_report_blocks(dataset, cfg, tmp_path):
    data, root = dataset
    ds = ev.load_dataset(data, root=root)
    r = ev.evaluate_model("fake", "fake.pt", ["Mask", "person"], ds, cfg, lambda image: ([], 1.0))
    summary = ev.summary_table([r], ds)
    assert summary.startswith("**Table 5.**") and "| fake |" in summary
    md = ev.write_outputs([r], ds, tmp_path / "out", {"date": "today", "machine": "test"})
    assert md.exists() and (tmp_path / "out" / "model-evaluation.json").exists()
    report = tmp_path / "report.md"
    report.write_text(
        "before\n<!-- evaluation:summary -->\nold\n<!-- /evaluation:summary -->\nafter\n"
    )
    assert ev.fill_report(report, {"summary": summary, "detail": "x"}) == ["summary"]
    text = report.read_text()
    assert (
        "old" not in text
        and "| fake |" in text
        and text.startswith("before")
        and text.endswith("after\n")
    )


def test_discovery_prefers_the_openvino_export(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "a.pt").write_bytes(b"")
    (tmp_path / "m" / "a_openvino_model").mkdir()
    (tmp_path / "m" / "b.pt").write_bytes(b"")
    found = ev.discover([str(tmp_path / "m")])
    assert found == [f"{tmp_path / 'm' / 'a_openvino_model'}/", str(tmp_path / "m" / "b.pt")]
    assert ev.label_for("models/ppe-yolo11s_openvino_model/") == "v2 (ppe-yolo11s)"


def test_an_apron_wearer_is_a_false_stop_the_seven_class_model_cannot_avoid(dataset, cfg):
    """Labels: compliant through an apron. A model without Apron must stop it, however good."""
    data, root = dataset
    ds = ev.load_dataset(data, root=root)
    apron = [x for x in compliant() if x.name != "sleeves"] + [box("Apron", 4)]
    seven_classes = ["person", "headnet", "Gloves", "Mask", "Safetyglasses", "sleeves"]
    seven_classes.append("Wrong Sleeve")
    names = [*seven_classes, "Apron"]
    ds.samples = [ev.Sample(ds.samples[0].image, apron)]
    ds.names = names

    def perfect(image, lacks=()):
        return [Detection(t.name, 0.99, t.xyxy) for t in apron if t.name not in lacks], 1.0

    seven = ev.evaluate_model(
        "v6", "v6.pt", seven_classes, ds, cfg,
        lambda image: perfect(image, lacks=("Apron",)),
    )
    d = seven.decisions
    assert (d.false_stop, d.false_stop_unseeable, d.truth_ok_unseeable) == (1, 1, 1)
    assert d.false_stop_rate == 1.0 and d.false_stop_rate_seeable != d.false_stop_rate_seeable
    assert d.disagreements == [(ds.samples[0].image.name, ev.OK, ev.VIOLATION)]

    eight = ev.evaluate_model("apron", "apron.pt", names, ds, cfg, perfect)
    assert eight.decisions.agree == 1 and eight.decisions.truth_ok_unseeable == 0
    assert "| 0.0 % (0/1) |" in ev.summary_table([eight], ds)
