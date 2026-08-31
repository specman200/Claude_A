"""Resolving which file the exporter actually reads.

Only a PyTorch checkpoint can be exported, but `model.weights` normally points
at the *export* — that is what the station runs. These pin that `python -m
ppe.export` follows that back to the .pt instead of handing ultralytics a
format it rejects with a TypeError about "not a PyTorch model".
"""

import pytest
import yaml

from ppe.export import describe, main, source_weights, versions


def test_a_checkpoint_is_used_as_given(tmp_path):
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"")
    assert source_weights(str(pt)) == pt


def test_a_checkpoint_that_is_not_there_is_still_returned_for_the_caller_to_report(tmp_path):
    """Existence is main()'s to report, with the path in the message."""
    missing = tmp_path / "nope.pt"
    assert source_weights(str(missing)) == missing


@pytest.mark.parametrize(
    "export_name",
    ["best_openvino_model", "best_ncnn_model", "best_saved_model", "best_paddle_model"],
)
def test_an_export_directory_resolves_to_the_checkpoint_beside_it(tmp_path, export_name):
    (tmp_path / "best.pt").write_bytes(b"")
    (tmp_path / export_name).mkdir()
    assert source_weights(str(tmp_path / export_name)) == tmp_path / "best.pt"


def test_an_export_directory_resolves_even_before_it_has_been_generated(tmp_path):
    """A fresh clone has the .pt but not the IR — that is the case you export in."""
    (tmp_path / "best.pt").write_bytes(b"")
    assert source_weights(str(tmp_path / "best_openvino_model")) == tmp_path / "best.pt"


def test_an_exported_file_resolves_to_the_checkpoint_beside_it(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"")
    for name in ("best.onnx", "best.engine", "best.tflite"):
        assert source_weights(str(tmp_path / name)) == tmp_path / "best.pt"


def test_an_export_with_no_checkpoint_beside_it_resolves_to_nothing(tmp_path):
    (tmp_path / "orphan_openvino_model").mkdir()
    assert source_weights(str(tmp_path / "orphan_openvino_model")) is None
    assert source_weights(str(tmp_path / "orphan.onnx")) is None


def config_at(tmp_path, weights):
    """The shipped config with model.weights repointed — nothing else changed."""
    with open("config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["model"]["weights"] = str(weights)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_export_refuses_an_orphaned_export_with_an_actionable_message(tmp_path, caplog):
    """The ultralytics TypeError names the format, not the fix. This says the fix."""
    (tmp_path / "orphan_openvino_model").mkdir()
    cfg = config_at(tmp_path, tmp_path / "orphan_openvino_model")
    assert main(["-c", cfg]) == 1
    assert "-w path/to/best.pt" in caplog.text


def test_export_reports_a_checkpoint_that_is_not_there_by_name(tmp_path, caplog):
    cfg = config_at(tmp_path, tmp_path / "gone.pt")
    assert main(["-c", cfg]) == 1
    assert "no such weights" in caplog.text
    assert "gone.pt" in caplog.text


def test_int8_without_calibration_data_is_refused(tmp_path, caplog):
    """Calibrating on someone else's images is how int8 quietly loses recall."""
    (tmp_path / "best.pt").write_bytes(b"")
    cfg = config_at(tmp_path, tmp_path / "best.pt")
    assert main(["-c", cfg, "--int8"]) == 1
    assert "--data" in caplog.text


# -- what a refused file tells you -----------------------------------------


def test_describe_names_the_resolved_path_and_the_size(tmp_path):
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"x" * 2_000_000)
    said = describe(pt)
    assert str(pt.resolve()) in said
    assert "2.0 MB" in said


def test_describe_survives_a_file_that_is_not_there(tmp_path):
    assert "unreadable" in describe(tmp_path / "gone.pt")


def test_versions_names_every_runtime_an_export_depends_on():
    said = versions()
    assert all(name in said for name in ("ultralytics", "torch", "openvino", "onnx"))


def test_a_refused_checkpoint_reports_which_file_and_which_versions(
    tmp_path, caplog, monkeypatch
):
    """Ultralytics raises TypeError for a TorchScript archive, a bare
    state_dict, a YOLOv5-era checkpoint and a truncated download alike, each
    time describing the format. Which file was read, and on which versions,
    is what separates them — and is exactly what a bug report needs."""
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"not really a checkpoint")

    def refuse(*_args, **_kwargs):
        raise TypeError("ERROR is a TorchScript archive, not an Ultralytics PyTorch checkpoint.")

    monkeypatch.setattr("ultralytics.YOLO", refuse)
    assert main(["-c", config_at(tmp_path, pt)]) == 1
    assert str(pt.resolve()) in caplog.text
    assert "ultralytics=" in caplog.text
    assert "torch=" in caplog.text
    assert "TorchScript archive" in caplog.text
