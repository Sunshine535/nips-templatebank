"""Reproducibility tests for nips-templatebank core components."""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.template_dsl import (
    CompositionExecutor,
    CompositionPlan,
    DType,
    Executor,
    Op,
    Program,
    Slot,
    Step,
    Subroutine,
    SubroutineLibrary,
)
from src.mcd_split import build_mcd_split, extract_atoms, extract_compounds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_add_program(prog_id: str = "add") -> Program:
    """a + b program."""
    return Program(
        program_id=prog_id,
        slots=[Slot("a", DType.FLOAT, "first"), Slot("b", DType.FLOAT, "second")],
        steps=[
            Step(Op.COMPUTE, "result", "a + b", ["a", "b"], DType.FLOAT),
            Step(Op.OUTPUT, "__output__", "result", ["result"], DType.FLOAT),
        ],
    )


def _make_mul_program(prog_id: str = "mul") -> Program:
    """a * b program."""
    return Program(
        program_id=prog_id,
        slots=[Slot("a", DType.FLOAT, "first"), Slot("b", DType.FLOAT, "second")],
        steps=[
            Step(Op.COMPUTE, "result", "a * b", ["a", "b"], DType.FLOAT),
            Step(Op.OUTPUT, "__output__", "result", ["result"], DType.FLOAT),
        ],
    )


def _make_library() -> SubroutineLibrary:
    lib = SubroutineLibrary()
    lib.add(Subroutine("L00", _make_add_program("L00_prog"), support=10, mdl_gain=5.0))
    lib.add(Subroutine("L01", _make_mul_program("L01_prog"), support=8, mdl_gain=4.0))
    return lib


# ---------------------------------------------------------------------------
# Executor round-trip
# ---------------------------------------------------------------------------

class TestExecutor:
    def test_simple_add(self):
        prog = _make_add_program()
        ok, result, env = Executor().execute(prog, {"a": 3.0, "b": 5.0})
        assert ok is True
        assert result == pytest.approx(8.0)

    def test_simple_mul(self):
        prog = _make_mul_program()
        ok, result, env = Executor().execute(prog, {"a": 4.0, "b": 7.0})
        assert ok is True
        assert result == pytest.approx(28.0)

    def test_missing_binding_fails(self):
        prog = _make_add_program()
        ok, result, env = Executor().execute(prog, {"a": 1.0})
        assert ok is False
        assert result is None
        assert "error" in env

    def test_multi_step_program(self):
        """(a + b) * c with intermediate variable."""
        prog = Program(
            program_id="multi",
            slots=[
                Slot("a", DType.FLOAT), Slot("b", DType.FLOAT), Slot("c", DType.FLOAT),
            ],
            steps=[
                Step(Op.COMPUTE, "s", "a + b", ["a", "b"], DType.FLOAT),
                Step(Op.COMPUTE, "result", "s * c", ["s", "c"], DType.FLOAT),
                Step(Op.OUTPUT, "__output__", "result", ["result"], DType.FLOAT),
            ],
        )
        ok, result, _ = Executor().execute(prog, {"a": 2.0, "b": 3.0, "c": 4.0})
        assert ok and result == pytest.approx(20.0)

    def test_type_coercion(self):
        prog = Program(
            program_id="int_out",
            slots=[Slot("n", DType.INT)],
            steps=[
                Step(Op.COMPUTE, "result", "n * 2", ["n"], DType.INT),
                Step(Op.OUTPUT, "__output__", "result", ["result"], DType.INT),
            ],
        )
        ok, result, _ = Executor().execute(prog, {"n": 5})
        assert ok and result == 10


# ---------------------------------------------------------------------------
# CompositionExecutor
# ---------------------------------------------------------------------------

class TestCompositionExecutor:
    def test_single_call(self):
        lib = _make_library()
        plan = CompositionPlan(calls=[{"sub_id": "L00", "bindings": {"a": 10.0, "b": 20.0}}])
        comp = CompositionExecutor(lib)
        ok, result, stats = comp.execute(plan, {})
        assert ok is True
        assert result == pytest.approx(30.0)
        assert stats["calls_succeeded"] == 1

    def test_chained_calls(self):
        """L00(a=2,b=3) -> env has a=2; L01 picks up env[a]=2, bindings b=4 -> 8."""
        lib = _make_library()
        plan = CompositionPlan(calls=[
            {"sub_id": "L00", "bindings": {"a": 2.0, "b": 3.0}},
            {"sub_id": "L01", "bindings": {"b": 4.0}},
        ])
        comp = CompositionExecutor(lib)
        ok, result, stats = comp.execute(plan, {})
        assert ok is True
        assert result == pytest.approx(8.0)
        assert stats["calls_succeeded"] == 2

    def test_unknown_subroutine_fails(self):
        lib = _make_library()
        plan = CompositionPlan(calls=[{"sub_id": "NONEXIST", "bindings": {}}])
        comp = CompositionExecutor(lib)
        ok, result, stats = comp.execute(plan, {})
        assert ok is False
        assert "error" in stats

    def test_initial_bindings_propagate(self):
        lib = _make_library()
        plan = CompositionPlan(calls=[{"sub_id": "L00", "bindings": {}}])
        comp = CompositionExecutor(lib)
        ok, result, _ = comp.execute(plan, {"a": 100.0, "b": 200.0})
        assert ok is True
        assert result == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Library save / load round-trip
# ---------------------------------------------------------------------------

class TestLibrarySaveLoad:
    def test_round_trip(self, tmp_path):
        lib = _make_library()
        path = str(tmp_path / "lib.json")
        lib.save(path)

        loaded = SubroutineLibrary.load(path)
        assert loaded.size == lib.size
        assert set(loaded.subroutines.keys()) == set(lib.subroutines.keys())
        for sid in lib.subroutines:
            orig = lib.subroutines[sid]
            copy = loaded.subroutines[sid]
            assert orig.sub_id == copy.sub_id
            assert orig.support == copy.support
            assert len(orig.program.steps) == len(copy.program.steps)

    def test_loaded_library_executes(self, tmp_path):
        lib = _make_library()
        path = str(tmp_path / "lib.json")
        lib.save(path)
        loaded = SubroutineLibrary.load(path)

        plan = CompositionPlan(calls=[{"sub_id": "L01", "bindings": {"a": 6.0, "b": 7.0}}])
        ok, result, _ = CompositionExecutor(loaded).execute(plan, {})
        assert ok and result == pytest.approx(42.0)

    def test_saved_json_is_valid(self, tmp_path):
        lib = _make_library()
        path = str(tmp_path / "lib.json")
        lib.save(path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert len(data) == 2


# ---------------------------------------------------------------------------
# MCD split invariants
# ---------------------------------------------------------------------------

class TestMCDSplit:
    @staticmethod
    def _make_plans(n: int = 60):
        """Generate diverse plans for MCD split testing."""
        import random
        rng = random.Random(123)
        sub_ids = [f"S{i}" for i in range(6)]
        plans = []
        for i in range(n):
            n_calls = rng.randint(1, 3)
            calls = [{"sub_id": rng.choice(sub_ids), "bindings": {"x": float(i)}} for _ in range(n_calls)]
            plans.append({"plan_data": {"plan": calls}})
        return plans

    def test_split_covers_all_indices(self):
        plans = self._make_plans(60)
        split = build_mcd_split(plans, num_trials=50, seed=42)
        all_idx = sorted(split["train"] + split["dev"] + split["test"])
        assert all_idx == list(range(60))

    def test_no_overlap(self):
        plans = self._make_plans(60)
        split = build_mcd_split(plans, num_trials=50, seed=42)
        train_set = set(split["train"])
        dev_set = set(split["dev"])
        test_set = set(split["test"])
        assert not (train_set & dev_set)
        assert not (train_set & test_set)
        assert not (dev_set & test_set)

    def test_stats_present(self):
        plans = self._make_plans(60)
        split = build_mcd_split(plans, num_trials=50, seed=42)
        assert "stats" in split
        assert "atom_tvd" in split["stats"]
        assert "compound_divergence" in split["stats"]


# ---------------------------------------------------------------------------
# Tiny end-to-end smoke: extract -> library -> plan -> execute
# ---------------------------------------------------------------------------

class TestEndToEndSmoke:
    def test_full_pipeline_roundtrip(self, tmp_path):
        prog_add = _make_add_program("smoke_add")
        prog_mul = _make_mul_program("smoke_mul")

        lib = SubroutineLibrary()
        lib.add(Subroutine("S0", prog_add, support=5, mdl_gain=2.0))
        lib.add(Subroutine("S1", prog_mul, support=3, mdl_gain=1.5))

        lib_path = str(tmp_path / "lib.json")
        lib.save(lib_path)
        loaded_lib = SubroutineLibrary.load(lib_path)

        plan = CompositionPlan(calls=[
            {"sub_id": "S0", "bindings": {"a": 10.0, "b": 5.0}},
            {"sub_id": "S1", "bindings": {"b": 3.0}},
        ])
        ok, result, stats = CompositionExecutor(loaded_lib).execute(plan, {})
        assert ok is True
        assert result == pytest.approx(30.0)
        assert stats["calls_succeeded"] == 2

        plan_json = plan.to_json()
        plan_loaded = CompositionPlan.from_json(plan_json)
        assert plan_loaded.num_calls == 2
        assert plan_loaded.subroutine_ids == ["S0", "S1"]
