"""Tests for V-GIFT: ValueAnnotatedDataflowPlan + ConsistencyExecutor."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataflow_plan import (
    BindingRef,
    ConsistencyExecutor,
    DataflowExecutor,
    DataflowPlan,
    PlanCall,
    ValueAnnotatedDataflowPlan,
    ValueAnnotatedPlanCall,
    ValueHint,
)
from src.template_dsl import DType, Op, Program, Slot, Step, Subroutine, SubroutineLibrary


def _make_lib():
    lib = SubroutineLibrary()
    lib.add(Subroutine("L_ADD", Program("add", [
        Slot("a", DType.FLOAT, ""), Slot("b", DType.FLOAT, ""),
    ], [
        Step(Op.COMPUTE, "r", "a + b", ["a", "b"], DType.FLOAT),
        Step(Op.OUTPUT, "__output__", "r", ["r"], DType.FLOAT),
    ]), support=1))
    lib.add(Subroutine("L_MUL", Program("mul", [
        Slot("x", DType.FLOAT, ""), Slot("y", DType.FLOAT, ""),
    ], [
        Step(Op.COMPUTE, "r", "x * y", ["x", "y"], DType.FLOAT),
        Step(Op.OUTPUT, "__output__", "r", ["r"], DType.FLOAT),
    ]), support=1))
    return lib


class TestValueHint:
    def test_round_trip(self):
        vh = ValueHint(value=5.0, dtype="float", confidence=0.9)
        d = vh.to_dict()
        vh2 = ValueHint.from_dict(d)
        assert vh2.value == 5.0
        assert vh2.confidence == 0.9

    def test_from_scalar(self):
        vh = ValueHint.from_dict(42)
        assert vh.value == 42


class TestValueAnnotatedPlanCall:
    def test_with_hint(self):
        call = ValueAnnotatedPlanCall(
            call_id="c0", sub_id="L_ADD",
            bindings={"a": BindingRef(source="quantity", value=2.0),
                       "b": BindingRef(source="quantity", value=3.0)},
            value_hint=ValueHint(value=5.0),
        )
        d = call.to_dict()
        assert d["value_hint"]["value"] == 5.0
        restored = ValueAnnotatedPlanCall.from_dict(d)
        assert restored.value_hint.value == 5.0

    def test_without_hint(self):
        call = ValueAnnotatedPlanCall(
            call_id="c0", sub_id="L_ADD",
            bindings={"a": BindingRef(source="quantity", value=2.0),
                       "b": BindingRef(source="quantity", value=3.0)},
        )
        d = call.to_dict()
        assert "value_hint" not in d


class TestConsistencyExecutor:
    def test_consistent_hints(self):
        """Correct value hints → agreement."""
        lib = _make_lib()
        plan = ValueAnnotatedDataflowPlan(
            calls=[
                ValueAnnotatedPlanCall(
                    call_id="c0", sub_id="L_ADD",
                    bindings={"a": BindingRef(source="quantity", value=2.0),
                               "b": BindingRef(source="quantity", value=3.0)},
                    value_hint=ValueHint(value=5.0),
                ),
                ValueAnnotatedPlanCall(
                    call_id="c1", sub_id="L_MUL",
                    bindings={"x": BindingRef(source="call_output", call_id="c0"),
                               "y": BindingRef(source="quantity", value=4.0)},
                    value_hint=ValueHint(value=20.0),
                ),
            ],
            final=BindingRef(source="call_output", call_id="c1"),
        )
        executor = ConsistencyExecutor(lib)
        result = executor.execute(plan)
        assert result["symbolic_exec_ok"]
        assert result["symbolic_result"] == 20.0
        assert result["value_hint_result"] == 20.0
        assert result["final_agreement"]
        assert result["consistency_errors"] == 0
        assert result["value_hints_consistent"] == 2

    def test_inconsistent_hint(self):
        """Wrong value hint → consistency error."""
        lib = _make_lib()
        plan = ValueAnnotatedDataflowPlan(
            calls=[
                ValueAnnotatedPlanCall(
                    call_id="c0", sub_id="L_ADD",
                    bindings={"a": BindingRef(source="quantity", value=2.0),
                               "b": BindingRef(source="quantity", value=3.0)},
                    value_hint=ValueHint(value=999.0),
                ),
            ],
            final=BindingRef(source="call_output", call_id="c0"),
        )
        executor = ConsistencyExecutor(lib)
        result = executor.execute(plan)
        assert result["symbolic_result"] == 5.0
        assert result["value_hint_result"] == 999.0
        assert not result["final_agreement"]
        assert result["consistency_errors"] == 1

    def test_no_hints_still_executes(self):
        """Plan without value hints executes normally."""
        lib = _make_lib()
        plan = ValueAnnotatedDataflowPlan(
            calls=[
                PlanCall(call_id="c0", sub_id="L_ADD",
                         bindings={"a": BindingRef(source="quantity", value=2.0),
                                    "b": BindingRef(source="quantity", value=3.0)}),
            ],
            final=BindingRef(source="call_output", call_id="c0"),
        )
        executor = ConsistencyExecutor(lib)
        result = executor.execute(plan)
        assert result["symbolic_exec_ok"]
        assert result["symbolic_result"] == 5.0
        assert result["value_hints_present"] == 0

    def test_existing_dataflow_tests_still_pass(self):
        """Old DataflowExecutor still works unchanged."""
        lib = _make_lib()
        plan = DataflowPlan(
            calls=[
                PlanCall(call_id="c0", sub_id="L_ADD",
                         bindings={"a": BindingRef(source="quantity", value=2.0),
                                    "b": BindingRef(source="quantity", value=3.0)}),
                PlanCall(call_id="c1", sub_id="L_MUL",
                         bindings={"x": BindingRef(source="call_output", call_id="c0"),
                                    "y": BindingRef(source="quantity", value=4.0)}),
            ],
            final=BindingRef(source="call_output", call_id="c1"),
        )
        executor = DataflowExecutor(lib)
        ok, result, stats = executor.execute(plan)
        assert ok and result == 20.0
