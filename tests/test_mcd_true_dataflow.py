"""Tests for true_dataflow compound extraction (GPT-5.5 Task 6)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mcd_split import (
    build_mcd_split,
    extract_compounds,
    extract_compounds_true_dataflow,
)


def test_legacy_creates_flow_for_any_adjacency():
    """Legacy: any consecutive call pair gets a flow: compound."""
    plan = {
        "plan": [
            {"sub_id": "L00", "bindings": {"a": 1, "b": 2}},
            {"sub_id": "L01", "bindings": {"x": 3, "y": 4}},
        ]
    }
    compounds = extract_compounds(plan)
    has_flow = any(c.startswith("flow:") for c in compounds)
    assert has_flow, "Legacy should create flow compounds for adjacent calls"


def test_true_dataflow_requires_explicit_call_output_ref():
    """True_dataflow: flow compound only if BindingRef references call_output."""
    plan_no_flow = {
        "calls": [
            {
                "call_id": "c0", "sub_id": "L00",
                "bindings": {
                    "a": {"source": "quantity", "value": 1},
                    "b": {"source": "quantity", "value": 2},
                },
            },
            {
                "call_id": "c1", "sub_id": "L01",
                "bindings": {
                    "x": {"source": "quantity", "value": 3},
                    "y": {"source": "quantity", "value": 4},
                },
            },
        ]
    }
    compounds = extract_compounds_true_dataflow(plan_no_flow)
    has_flow = any(c.startswith("true_flow:") for c in compounds)
    assert not has_flow, (
        "True_dataflow must NOT create flow compounds "
        "without explicit call_output ref"
    )


def test_true_dataflow_creates_flow_with_call_output_ref():
    """True_dataflow: call_output ref → true_flow compound."""
    plan_with_flow = {
        "calls": [
            {
                "call_id": "c0", "sub_id": "L00",
                "bindings": {
                    "a": {"source": "quantity", "value": 1},
                    "b": {"source": "quantity", "value": 2},
                },
            },
            {
                "call_id": "c1", "sub_id": "L01",
                "bindings": {
                    "x": {"source": "call_output", "call_id": "c0"},
                    "y": {"source": "quantity", "value": 4},
                },
            },
        ]
    }
    compounds = extract_compounds_true_dataflow(plan_with_flow)
    has_flow = any(c.startswith("true_flow:") for c in compounds)
    assert has_flow, "True_dataflow must create flow compounds for call_output refs"
    assert "true_flow:L00>L01" in compounds


def test_build_mcd_split_compound_mode_switch():
    """build_mcd_split must respect compound_mode parameter."""
    plans_legacy = []
    plans_gift = []
    for i in range(20):
        plans_legacy.append({
            "plan_data": {
                "plan": [
                    {"sub_id": f"L{i%3:02d}", "bindings": {"x": i}},
                    {"sub_id": f"L{(i+1)%3:02d}", "bindings": {"y": i+1}},
                ]
            }
        })
        plans_gift.append({
            "plan_data": {
                "calls": [
                    {"call_id": "c0", "sub_id": f"L{i%3:02d}",
                     "bindings": {"x": {"source": "quantity", "value": i}}},
                    {"call_id": "c1", "sub_id": f"L{(i+1)%3:02d}",
                     "bindings": {
                         "y": {"source": "call_output", "call_id": "c0"}
                     }},
                ]
            }
        })

    split_legacy = build_mcd_split(plans_legacy, num_trials=5, compound_mode="legacy", seed=42)
    split_gift = build_mcd_split(plans_gift, num_trials=5, compound_mode="true_dataflow", seed=42)

    assert "stats" in split_legacy
    assert "stats" in split_gift
