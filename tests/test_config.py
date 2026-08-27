"""Config load, save and validation."""

import pytest
import yaml

from ppe.config import ClassCfg, Config


def test_loads_the_shipped_config():
    cfg = Config.load("config.yaml")
    cfg.validate()
    assert len(cfg.cameras) == 2
    assert cfg.tower.coils["red"] == 2
    from pathlib import Path

    assert Path(cfg.model.weights).exists(), "config points at weights that are not there"


def test_the_shipped_classes_match_the_fine_tuned_model():
    """Guards the config against drifting from the weights it points at."""
    cfg = Config.load("config.yaml")
    assert {c.name for c in cfg.ppe.classes} == {
        "Gloves", "Mask", "Safetyglasses", "Wrong Sleeve", "headnet", "person", "sleeves",
    }


def test_the_violation_class_is_shipped_as_forbidden_not_as_required_ppe():
    """If this ever flips, the tower goes green on the exact fault it watches for."""
    cfg = Config.load("config.yaml")
    by_name = {c.name: c for c in cfg.ppe.classes}
    assert by_name["Wrong Sleeve"].forbidden
    assert by_name["Wrong Sleeve"].required          # it still gates the light
    assert not by_name["sleeves"].forbidden          # the correct-sleeve class does not
    assert not by_name["person"].required            # context, not equipment


def test_labels_default_to_a_readable_name():
    assert ClassCfg("safety_boots").label == "Safety Boots"


def test_unknown_keys_are_ignored(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"model": {"imgsz": 320, "future_option": 1}}))
    assert Config.load(path).model.imgsz == 320


def test_save_round_trips(tmp_path):
    cfg = Config.load("config.yaml")
    cfg.ppe.classes.append(ClassCfg("apron", required=False, conf=0.6))
    out = cfg.save(tmp_path / "out.yaml")

    again = Config.load(out)
    assert [c.name for c in again.ppe.classes] == [c.name for c in cfg.ppe.classes]
    assert again.ppe.classes[-1].conf == 0.6
    assert again.ppe.classes[-1].required is False
    assert again.tower.coils == cfg.tower.coils


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda c: c.cameras.clear(), "camera"),
        (lambda c: setattr(c.model, "imgsz", 641), "multiple of 32"),
        (lambda c: c.ppe.classes.clear(), "empty"),
        (lambda c: c.ppe.classes.append(ClassCfg(c.ppe.classes[0].name)), "duplicate"),
        (lambda c: c.tower.coils.update(strobe=9), "unknown tower coils"),
        (lambda c: setattr(c.ppe.classes[0], "expect", "maybe"), "expect must be one of"),
    ],
)
def test_validate_rejects_broken_configs(mutate, message):
    cfg = Config.load("config.yaml")
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_the_shipped_config_gates_on_the_person_class():
    cfg = Config.load("config.yaml")
    assert cfg.ppe.subject == "person"
    assert cfg.ppe.subject in {c.name for c in cfg.ppe.classes}, (
        "the subject must be listed, or NMS would filter it out before we see it"
    )
    assert 0.0 <= cfg.ppe.containment <= 1.0


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda c: setattr(c.ppe, "subject", "nobody"), "must also appear in ppe.classes"),
        (lambda c: setattr(c.ppe, "containment", 1.5), "between 0 and 1"),
        (lambda c: setattr(c.ppe, "containment", -0.1), "between 0 and 1"),
    ],
)
def test_validate_rejects_a_broken_subject_setup(mutate, message):
    cfg = Config.load("config.yaml")
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        cfg.validate()
