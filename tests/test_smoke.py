"""Smoke tests: package imports, configs load, and contract values have not drifted.

CI-safe: needs only numpy + pyyaml (no mediapipe / torch / opencv).
"""

from __future__ import annotations

import logging

import pytest

import asl
from asl.config import cfg_get, load_config, load_yaml
from asl.utils.logging import ROOT_LOGGER_NAME, setup_logging

CONFIG_NAMES = ["landmarks", "features", "train_fingerspell", "train_sign", "runtime"]


def test_package_version() -> None:
    assert isinstance(asl.__version__, str) and asl.__version__


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_config_loads(name: str) -> None:
    cfg = load_config(name)
    assert isinstance(cfg, dict) and cfg


def test_landmark_contract() -> None:
    cfg = load_config("landmarks")
    assert cfg["hands"]["num_points"] == 21
    assert cfg["pose"]["indices"] == [0, 11, 12, 13, 14, 15, 16]
    assert len(cfg["pose"]["names"]) == len(cfg["pose"]["indices"])
    assert cfg["part_order"] == ["left_hand", "right_hand", "pose", "lips"]
    assert cfg["coordinates"]["use_z"] is False


def test_feature_contract() -> None:
    cfg = load_config("features")
    assert cfg["sequence"]["length"] == 64
    assert cfg["stats"]["source_split"] == "train"
    # shoulder names used for normalization must exist in the pose list
    pose_names = load_config("landmarks")["pose"]["names"]
    assert all(s in pose_names for s in cfg["body"]["shoulders"])


def test_runtime_camera_contract() -> None:
    cam = load_config("runtime")["camera"]
    assert (cam["width"], cam["height"]) == (640, 480)
    assert cam["fourcc"] == "MJPG"


def test_runtime_tracking_contract() -> None:
    rt = load_config("runtime")
    assert rt["tracking"]["model_path"].endswith("holistic_landmarker.task")
    assert rt["tracking"]["mediapipe_version"] == "1.1.0"
    assert rt["onnx"]["providers"] == ["CPUExecutionProvider"]


def test_cfg_get_dotted() -> None:
    assert cfg_get({"a": {"b": 3}}, "a.b") == 3


def test_cfg_get_default() -> None:
    assert cfg_get({"a": {}}, "a.b", default=None) is None


def test_cfg_get_missing_raises() -> None:
    with pytest.raises(KeyError, match="a.b"):
        cfg_get({"a": {}}, "a.b")


def test_load_yaml_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nope.yaml")


def test_load_yaml_rejects_non_mapping(tmp_path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(p)


def test_load_yaml_empty_is_empty_dict(tmp_path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_yaml(p) == {}


def test_load_config_unknown_name() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")


def test_setup_logging_idempotent() -> None:
    setup_logging("INFO")
    n = len(logging.getLogger(ROOT_LOGGER_NAME).handlers)
    setup_logging("DEBUG")
    assert len(logging.getLogger(ROOT_LOGGER_NAME).handlers) == n