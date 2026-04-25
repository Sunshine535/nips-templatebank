"""Validate ablation config integrity (GPT-5.5 R3 Task 3).

Catches duplicate keys, library path collisions, and required fields.
"""
import os
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ABLATION_DIR = REPO_ROOT / "configs" / "ablations"


class _DuplicateKeyDetector(yaml.SafeLoader):
    """Custom loader that raises on duplicate keys at the same mapping level."""


def _construct_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate key '{key}' at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_DuplicateKeyDetector.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


@pytest.mark.parametrize("config_name", [
    "full_gift_step.yaml",
    "old_fragment_only.yaml",
    "gift_no_call_output.yaml",
    "gift_no_active_gate.yaml",
])
def test_no_duplicate_keys(config_name):
    path = ABLATION_DIR / config_name
    with open(path) as f:
        try:
            yaml.load(f, Loader=_DuplicateKeyDetector)
        except ValueError as e:
            pytest.fail(f"{config_name}: {e}")


@pytest.mark.parametrize("config_name,expects_library", [
    ("full_gift_step.yaml", True),
    ("gift_no_active_gate.yaml", False),
    ("gift_no_call_output.yaml", False),
    ("old_fragment_only.yaml", False),
])
def test_library_path_naming(config_name, expects_library):
    """training section must not use ambiguous 'library' key."""
    path = ABLATION_DIR / config_name
    with open(path) as f:
        cfg = yaml.safe_load(f)
    training = cfg.get("training", {})
    assert "library" not in training, (
        f"{config_name}: training.library is ambiguous; "
        "use library_path (string) or library_config (mapping) at top level"
    )
    if expects_library:
        assert "library_path" in training, (
            f"{config_name}: should specify training.library_path"
        )


def test_required_fields():
    """All ablation configs must declare method + planner.model."""
    for cfg_path in ABLATION_DIR.glob("*.yaml"):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert "method" in cfg, f"{cfg_path.name}: missing 'method'"
        assert "planner" in cfg, f"{cfg_path.name}: missing 'planner'"
        assert "model" in cfg["planner"], f"{cfg_path.name}: missing planner.model"
