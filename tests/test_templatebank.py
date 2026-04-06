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
    inline_program,
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


# ---------------------------------------------------------------------------
# inline_program correctness
# ---------------------------------------------------------------------------

class TestInlineProgram:
    """Verify that inline_program rewrites variable names correctly and
    produces the same result as CompositionExecutor."""

    @staticmethod
    def _make_two_sub_library():
        """Library with two subroutines that share an intermediate name 'result'."""
        # sub_add: a + b -> result -> output
        prog_add = Program(
            program_id="add_prog",
            slots=[Slot("a", DType.FLOAT), Slot("b", DType.FLOAT)],
            steps=[
                Step(Op.COMPUTE, "result", "a + b", ["a", "b"], DType.FLOAT),
                Step(Op.OUTPUT, "__output__", "result", ["result"], DType.FLOAT),
            ],
        )
        # sub_mul: a * b -> result -> output
        prog_mul = Program(
            program_id="mul_prog",
            slots=[Slot("a", DType.FLOAT), Slot("b", DType.FLOAT)],
            steps=[
                Step(Op.COMPUTE, "result", "a * b", ["a", "b"], DType.FLOAT),
                Step(Op.OUTPUT, "__output__", "result", ["result"], DType.FLOAT),
            ],
        )
        lib = SubroutineLibrary()
        lib.add(Subroutine("ADD", prog_add, support=5, mdl_gain=2.0))
        lib.add(Subroutine("MUL", prog_mul, support=5, mdl_gain=2.0))
        return lib

    def test_single_call_inline_matches_compose(self):
        lib = self._make_two_sub_library()
        plan = CompositionPlan(calls=[
            {"sub_id": "ADD", "bindings": {"a": 3.0, "b": 7.0}},
        ])
        # Compose execution
        ok_c, result_c, _ = CompositionExecutor(lib).execute(plan, {})
        assert ok_c and result_c == pytest.approx(10.0)

        # Inline execution
        inlined = inline_program(plan, lib)
        assert inlined is not None
        ok_i, result_i, _ = Executor().execute(inlined, {"a": 3.0, "b": 7.0})
        assert ok_i and result_i == pytest.approx(10.0)

    def test_two_call_inline_matches_compose(self):
        """Two calls that share variable name 'result' internally."""
        lib = self._make_two_sub_library()
        plan = CompositionPlan(calls=[
            {"sub_id": "ADD", "bindings": {"a": 2.0, "b": 3.0}},
            {"sub_id": "MUL", "bindings": {"b": 4.0}},
        ])
        # Compose: ADD(2,3)=5 -> env has a=2; MUL(a=2, b=4)=8
        ok_c, result_c, _ = CompositionExecutor(lib).execute(plan, {})
        assert ok_c and result_c == pytest.approx(8.0)

        # Inline: should produce same result
        inlined = inline_program(plan, lib)
        assert inlined is not None
        ok_i, result_i, _ = Executor().execute(inlined, {"a": 2.0, "b": 4.0})
        assert ok_i and result_i == pytest.approx(8.0)

    def test_intermediate_output_demoted_to_compute(self):
        """First subroutine's OUTPUT should become COMPUTE in multi-call plan."""
        lib = self._make_two_sub_library()
        plan = CompositionPlan(calls=[
            {"sub_id": "ADD", "bindings": {"a": 1.0, "b": 1.0}},
            {"sub_id": "MUL", "bindings": {"a": 2.0, "b": 2.0}},
        ])
        inlined = inline_program(plan, lib)
        assert inlined is not None
        output_steps = [s for s in inlined.steps if s.op == Op.OUTPUT]
        assert len(output_steps) == 1, "Only the last subroutine should have OUTPUT"

    def test_no_stale_variable_references(self):
        """After inlining, expr and inputs must reference renamed targets."""
        lib = self._make_two_sub_library()
        plan = CompositionPlan(calls=[
            {"sub_id": "ADD", "bindings": {"a": 5.0, "b": 5.0}},
        ])
        inlined = inline_program(plan, lib)
        assert inlined is not None
        slot_names = {s.name for s in inlined.slots}
        # Every input reference should either be a slot name or a step target
        step_targets = {s.target for s in inlined.steps}
        valid_names = slot_names | step_targets
        for step in inlined.steps:
            for inp in step.inputs:
                assert inp in valid_names, (
                    f"Step targeting '{step.target}' references undefined input '{inp}'"
                )

    def test_unknown_subroutine_returns_none(self):
        lib = self._make_two_sub_library()
        plan = CompositionPlan(calls=[
            {"sub_id": "NONEXIST", "bindings": {}},
        ])
        assert inline_program(plan, lib) is None


# ---------------------------------------------------------------------------
# TestPostSplitAlignment
# ---------------------------------------------------------------------------

class TestPostSplitAlignment:
    """Verify that MCD split indices correctly partition programs."""

    @staticmethod
    def _make_diverse_plans(n: int = 20):
        import random
        rng = random.Random(77)
        sub_ids = [f"S{i}" for i in range(4)]
        plans = []
        for i in range(n):
            n_calls = rng.randint(1, 3)
            calls = [{"sub_id": rng.choice(sub_ids), "bindings": {"x": float(i)}} for _ in range(n_calls)]
            plans.append({"plan_data": {"plan": calls}, "index": i})
        return plans

    def test_train_indices_select_correct_subset(self):
        plans = self._make_diverse_plans(20)
        split = build_mcd_split(plans, num_trials=30, seed=42)
        train_idx = set(split["train"])
        train_plans = [plans[i] for i in sorted(train_idx) if i < len(plans)]
        # Each selected plan should match original
        for plan in train_plans:
            assert plan["index"] in train_idx

    def test_all_indices_covered(self):
        plans = self._make_diverse_plans(20)
        split = build_mcd_split(plans, num_trials=30, seed=42)
        all_idx = sorted(split["train"] + split["dev"] + split["test"])
        assert all_idx == list(range(20))

    def test_train_size_reasonable(self):
        plans = self._make_diverse_plans(20)
        split = build_mcd_split(plans, train_ratio=0.6, dev_ratio=0.2, test_ratio=0.2,
                                num_trials=30, seed=42)
        # Train should be roughly 60% of 20 = 12, allow tolerance
        assert 6 <= len(split["train"]) <= 16


# ---------------------------------------------------------------------------
# TestInlineProgramCorrectness
# ---------------------------------------------------------------------------

class TestInlineProgramCorrectness:
    """Create a 2-subroutine library, build a 2-call plan, inline it,
    and verify both composition execution and inlined execution produce
    the same result."""

    @staticmethod
    def _make_sub_add_mul_library():
        prog_add = Program(
            program_id="sub_add",
            slots=[Slot("a", DType.FLOAT), Slot("b", DType.FLOAT)],
            steps=[
                Step(Op.COMPUTE, "sum_ab", "a + b", ["a", "b"], DType.FLOAT),
                Step(Op.OUTPUT, "__output__", "sum_ab", ["sum_ab"], DType.FLOAT),
            ],
        )
        prog_mul = Program(
            program_id="sub_mul",
            slots=[Slot("a", DType.FLOAT), Slot("b", DType.FLOAT)],
            steps=[
                Step(Op.COMPUTE, "prod_ab", "a * b", ["a", "b"], DType.FLOAT),
                Step(Op.OUTPUT, "__output__", "prod_ab", ["prod_ab"], DType.FLOAT),
            ],
        )
        lib = SubroutineLibrary()
        lib.add(Subroutine("SADD", prog_add, support=5, mdl_gain=2.0))
        lib.add(Subroutine("SMUL", prog_mul, support=5, mdl_gain=2.0))
        return lib

    def test_inline_matches_compose_execution(self):
        lib = self._make_sub_add_mul_library()
        plan = CompositionPlan(calls=[
            {"sub_id": "SADD", "bindings": {"a": 3.0, "b": 4.0}},
            {"sub_id": "SMUL", "bindings": {"a": 2.0, "b": 5.0}},
        ])
        # Composition execution
        ok_c, result_c, stats_c = CompositionExecutor(lib).execute(plan, {})
        assert ok_c is True
        assert result_c == pytest.approx(10.0)

        # Inline and execute
        inlined = inline_program(plan, lib)
        assert inlined is not None
        ok_i, result_i, _ = Executor().execute(inlined, {"a": 2.0, "b": 5.0})
        assert ok_i is True
        assert result_i == pytest.approx(result_c)


# ---------------------------------------------------------------------------
# TestMultiCallPlanFaithfulness
# ---------------------------------------------------------------------------

class TestMultiCallPlanFaithfulness:
    """Build multi-call plans and verify execution succeeds for most."""

    def test_multi_call_plans_execute(self):
        lib = _make_library()  # L00=add, L01=mul
        plans = [
            CompositionPlan(calls=[
                {"sub_id": "L00", "bindings": {"a": float(i), "b": float(i + 1)}},
                {"sub_id": "L01", "bindings": {"b": 2.0}},
            ])
            for i in range(10)
        ]
        comp = CompositionExecutor(lib)
        successes = 0
        for plan in plans:
            ok, result, stats = comp.execute(plan, {})
            if ok and result is not None:
                successes += 1
        # Most plans should succeed
        assert successes >= 8, f"Only {successes}/10 plans succeeded"

    def test_single_call_faithfulness(self):
        lib = _make_library()
        plan = CompositionPlan(calls=[
            {"sub_id": "L00", "bindings": {"a": 10.0, "b": 20.0}},
        ])
        ok, result, _ = CompositionExecutor(lib).execute(plan, {})
        assert ok and result == pytest.approx(30.0)

    def test_three_call_chain(self):
        lib = _make_library()
        plan = CompositionPlan(calls=[
            {"sub_id": "L00", "bindings": {"a": 1.0, "b": 2.0}},
            {"sub_id": "L01", "bindings": {"b": 3.0}},
            {"sub_id": "L00", "bindings": {"b": 10.0}},
        ])
        ok, result, stats = CompositionExecutor(lib).execute(plan, {})
        assert ok is True
        assert stats["calls_succeeded"] == 3


# ---------------------------------------------------------------------------
# TestMCDRichCompounds
# ---------------------------------------------------------------------------

class TestMCDRichCompounds:
    """Verify that extract_compounds produces the expected compound types
    for a 3-call plan."""

    def test_all_compound_types_present(self):
        plan_data = {
            "plan": [
                {"sub_id": "L00", "bindings": {"x": 1.0, "y": 2.0}},
                {"sub_id": "L01", "bindings": {"x": 3.0}},
                {"sub_id": "L02", "bindings": {"x": 4.0, "z": 5.0}},
            ]
        }
        compounds = extract_compounds(plan_data)

        # Check for each expected compound type
        flow_compounds = [c for c in compounds if c.startswith("flow:")]
        assert len(flow_compounds) >= 1, f"Missing flow compounds: {compounds}"

        bind_flow_compounds = [c for c in compounds if c.startswith("bind_flow:")]
        assert len(bind_flow_compounds) >= 1, f"Missing bind_flow compounds: {compounds}"

        type_sig_compounds = [c for c in compounds if c.startswith("type_sig:")]
        assert len(type_sig_compounds) >= 1, f"Missing type_sig compounds: {compounds}"

        arity_pair_compounds = [c for c in compounds if c.startswith("arity_pair:")]
        assert len(arity_pair_compounds) >= 1, f"Missing arity_pair compounds: {compounds}"

        total_calls_compounds = [c for c in compounds if c.startswith("total_calls:")]
        assert len(total_calls_compounds) == 1
        assert "total_calls:3" in compounds

    def test_bigram_and_trigram_present(self):
        plan_data = {
            "plan": [
                {"sub_id": "A", "bindings": {"x": 1.0}},
                {"sub_id": "B", "bindings": {"x": 2.0}},
                {"sub_id": "C", "bindings": {"x": 3.0}},
            ]
        }
        compounds = extract_compounds(plan_data)
        assert "bigram:A>B" in compounds
        assert "bigram:B>C" in compounds
        assert "trigram:A>B>C" in compounds

    def test_depth_trigram_present(self):
        plan_data = {
            "plan": [
                {"sub_id": "X", "bindings": {"a": 1.0}},
                {"sub_id": "Y", "bindings": {"b": 2.0}},
                {"sub_id": "Z", "bindings": {"c": 3.0}},
            ]
        }
        compounds = extract_compounds(plan_data)
        depth_trigrams = [c for c in compounds if c.startswith("depth_trigram:")]
        assert len(depth_trigrams) >= 1, f"Missing depth_trigram compounds: {compounds}"
