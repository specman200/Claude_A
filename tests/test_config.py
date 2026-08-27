"""Config load, save and validation."""

import pytest
import yaml

from ppe.config import ClassCfg, Config


def test_loads_the_shipped_config():
    cfg = Config.load("config.yaml")
    cfg.validate()
    assert len(cfg.cameras) == 2
    assert [c.name for c in cfg.ppe.required] == ["helmet", "vest", "gloves", "goggles"]
    assert cfg.tower.coils["red"] == 2


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
        (lambda c: c.ppe.classes.append(ClassCfg("helmet")), "duplicate"),
        (lambda c: c.tower.coils.update(strobe=9), "unknown tower coils"),
    ],
)
def test_validate_rejects_broken_configs(mutate, message):
    cfg = Config.load("config.yaml")
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        cfg.validate()


# -- branding --------------------------------------------------------------


def test_branding_loads_from_the_shipped_config():
    cfg = Config.load("config.yaml")
    assert cfg.branding.name and cfg.branding.tagline
    assert cfg.branding.logo_path(cfg.base_dir) is not None


def test_a_relative_logo_resolves_from_the_config_file_not_the_cwd(tmp_path):
    from ppe.config import BrandingCfg

    (tmp_path / "brand").mkdir()
    logo = tmp_path / "brand" / "mine.svg"
    logo.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")

    cfg = BrandingCfg(name="Me", logo="brand/mine.svg")
    assert cfg.logo_path(tmp_path) == logo
    assert cfg.logo_path(tmp_path / "elsewhere") is None


def test_an_absolute_logo_path_is_used_as_given(tmp_path):
    from ppe.config import BrandingCfg

    logo = tmp_path / "mark.png"
    logo.write_bytes(b"x")
    assert BrandingCfg(logo=str(logo)).logo_path(tmp_path / "ignored") == logo


@pytest.mark.parametrize("logo", ["", "does/not/exist.svg"])
def test_an_unset_or_missing_logo_resolves_to_nothing(tmp_path, logo):
    from ppe.config import BrandingCfg

    assert BrandingCfg(logo=logo).logo_path(tmp_path) is None


def test_a_directory_is_not_mistaken_for_a_logo(tmp_path):
    from ppe.config import BrandingCfg

    (tmp_path / "assets").mkdir()
    assert BrandingCfg(logo="assets").logo_path(tmp_path) is None


def test_branding_survives_a_save_round_trip(tmp_path):
    cfg = Config.load("config.yaml")
    cfg.branding.name = "A. Engineer"
    cfg.branding.logo = "assets/mine.png"
    out = cfg.save(tmp_path / "out.yaml")

    again = Config.load(out)
    assert again.branding.name == "A. Engineer"
    assert again.branding.logo == "assets/mine.png"
    assert again.branding.tagline == cfg.branding.tagline


def test_a_config_without_a_branding_block_still_loads(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"model": {"imgsz": 320}}))
    cfg = Config.load(path)
    assert cfg.branding.name == "" and cfg.branding.logo_path(tmp_path) is None
