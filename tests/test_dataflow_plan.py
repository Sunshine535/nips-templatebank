"""Tests for explicit dataflow composition (GIFT).

These tests define the required semantics for GIFT:
- Every subroutine input must have an explicit BindingRef
- Outputs flow through call_output refs, not implicit env
- Plans are typed DAGs, not flat call lists with empty bindings
- Executor rejects missing/implicit bindings
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.template_dsl import DType, Executor, Op, Program, Slot, Step, Subroutine, SubroutineLibrary


def _make_add_sub():
    """Subroutine: result = a + b"""
    return Subroutine(
        sub_id="L_ADD",
        program=Program(
            program_id="add",
            slots=[Slot("a", DType.FLOAT, "first"), Slot("b", DType.FLOAT, "second")],
            steps=[
                Step(Op.COMPUTE, "result", "a + b", ["a", "b"], DType.FLOAT),
                Step(Op.OUTPUT, "__output__", "result", ["result"], DType.FLOAT),
            ],
        ),
        support=1,
    )


def _make_mul_sub():
    """Subroutine: result = x * y"""
    return Subroutine(
        sub_id="L_MUL",
        program=Program(
            program_id="mul",
            slots=[Slot("x", DType.FLOAT, "first"), Slot("y", DType.FLOAT, "second")],
            steps=[
                Step(Op.COMPUTE, "result", "x * y", ["x", "y"], DType.FLOAT),
                Step(Op.OUTPUT, "__output__", "result", ["result"], DType.FLOAT),
            ],
        ),
        support=1,
    )


class TestDataflowPlanImport:
    """Test that GIFT modules can be imported."""

    def test_import_binding_ref(self):
        from src.dataflow_plan import BindingRef

    def test_import_plan_call(self):
        from src.dataflow_plan import PlanCall

    def test_import_dataflow_plan(self):
        from src.dataflow_plan import DataflowPlan

    def test_import_dataflow_executor(self):
        from src.dataflow_plan import DataflowExecutor


class TestExplicitDataflow:
    """Core test: ADD output must feed MUL input via explicit binding."""

    def test_add_then_mul_explicit_binding(self):
        """(2 + 3) * 4 = 20, with explicit call_output binding."""
        from src.dataflow_plan import (
            BindingRef,
            DataflowExecutor,
            DataflowPlan,
            PlanCall,
        )

        lib = SubroutineLibrary()
        lib.add(_make_add_sub())
        lib.add(_make_mul_sub())

        plan = DataflowPlan(
            calls=[
                PlanCall(
                    call_id="c0",
                    sub_id="L_ADD",
                    bindings={
                        "a": BindingRef(source="quantity", value=2.0),
                        "b": BindingRef(source="quantity", value=3.0),
                    },
                ),
                PlanCall(
                    call_id="c1",
                    sub_id="L_MUL",
                    bindings={
                        "x": BindingRef(source="call_output", call_id="c0"),
                        "y": BindingRef(source="quantity", value=4.0),
                    },
                ),
            ],
            final=BindingRef(source="call_output", call_id="c1"),
        )

        executor = DataflowExecutor(lib)
        success, result, stats = executor.execute(plan)

        assert success, f"Execution failed: {stats}"
        assert result == 20.0, f"Expected 20.0, got {result}"
        assert stats.get("calls_made", 0) == 2
        assert stats.get("calls_succeeded", 0) == 2

    def test_rejects_empty_bindings(self):
        """Plans with empty bindings must fail, not silently use env."""
        from src.dataflow_plan import (
            BindingRef,
            DataflowExecutor,
            DataflowPlan,
            PlanCall,
        )

        lib = SubroutineLibrary()
        lib.add(_make_add_sub())

        plan = DataflowPlan(
            calls=[
                PlanCall(call_id="c0", sub_id="L_ADD", bindings={}),
            ],
            final=BindingRef(source="call_output", call_id="c0"),
        )

        executor = DataflowExecutor(lib)
        success, result, stats = executor.execute(plan)

        assert not success, "Should reject empty bindings"

    def test_rejects_missing_binding(self):
        """Partial bindings (one slot unbound) must fail."""
        from src.dataflow_plan import (
            BindingRef,
            DataflowExecutor,
            DataflowPlan,
            PlanCall,
        )

        lib = SubroutineLibrary()
        lib.add(_make_add_sub())

        plan = DataflowPlan(
            calls=[
                PlanCall(
                    call_id="c0",
                    sub_id="L_ADD",
                    bindings={"a": BindingRef(source="quantity", value=5.0)},
                ),
            ],
            final=BindingRef(source="call_output", call_id="c0"),
        )

        executor = DataflowExecutor(lib)
        success, result, stats = executor.execute(plan)

        assert not success, "Should reject missing binding for slot 'b'"

    def test_rejects_invalid_call_output_ref(self):
        """Referencing a non-existent call_id must fail."""
        from src.dataflow_plan import (
            BindingRef,
            DataflowExecutor,
            DataflowPlan,
            PlanCall,
        )

        lib = SubroutineLibrary()
        lib.add(_make_mul_sub())

        plan = DataflowPlan(
            calls=[
                PlanCall(
                    call_id="c0",
                    sub_id="L_MUL",
                    bindings={
                        "x": BindingRef(source="call_output", call_id="c_nonexistent"),
                        "y": BindingRef(source="quantity", value=4.0),
                    },
                ),
            ],
            final=BindingRef(source="call_output", call_id="c0"),
        )

        executor = DataflowExecutor(lib)
        success, result, stats = executor.execute(plan)

        assert not success, "Should reject ref to nonexistent call"


class TestDataflowSerialization:
    """Test JSON serialization/deserialization of GIFT plans."""

    def test_round_trip(self):
        from src.dataflow_plan import BindingRef, DataflowPlan, PlanCall

        plan = DataflowPlan(
            calls=[
                PlanCall(
                    call_id="c0",
                    sub_id="L_ADD",
                    bindings={
                        "a": BindingRef(source="quantity", value=2.0),
                        "b": BindingRef(source="quantity", value=3.0),
                    },
                ),
            ],
            final=BindingRef(source="call_output", call_id="c0"),
        )

        data = plan.to_dict()
        json_str = json.dumps(data)
        restored = DataflowPlan.from_dict(json.loads(json_str))

        assert len(restored.calls) == 1
        assert restored.calls[0].sub_id == "L_ADD"
        assert restored.final.source == "call_output"
        assert restored.final.call_id == "c0"


class TestActiveBidingPerturbation:
    """Test that changing a bound quantity changes the output (active binding)."""

    def test_perturbation_changes_output(self):
        from src.dataflow_plan import (
            BindingRef,
            DataflowExecutor,
            DataflowPlan,
            PlanCall,
        )

        lib = SubroutineLibrary()
        lib.add(_make_add_sub())

        def run_with_a(val):
            plan = DataflowPlan(
                calls=[
                    PlanCall(
                        call_id="c0",
                        sub_id="L_ADD",
                        bindings={
                            "a": BindingRef(source="quantity", value=val),
                            "b": BindingRef(source="quantity", value=3.0),
                        },
                    ),
                ],
                final=BindingRef(source="call_output", call_id="c0"),
            )
            executor = DataflowExecutor(lib)
            ok, result, _ = executor.execute(plan)
            return result

        r1 = run_with_a(2.0)
        r2 = run_with_a(10.0)

        assert r1 == 5.0
        assert r2 == 13.0
        assert r1 != r2, "Active binding: changing input must change output"
