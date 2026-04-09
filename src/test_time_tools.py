"""Test-Time Tool Building Module (SEVAL Phase 2).

When library compositions fail at inference time, this module:
  1. Analyzes failure patterns to identify missing operations
  2. Generates candidate new tools from failed attempt fragments
  3. Verifies candidates via execution on synthetic cases
  4. Adds verified tools to a per-problem cache
  5. Re-attempts composition with the expanded tool set

Budget-controlled: max M new tools, max T verification attempts per problem.
Inspired by T3RL (Tool Verification for Test-Time RL) but extends from
verification to BUILDING.
"""

import copy
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

logger = logging.getLogger(__name__)


@dataclass
class ToolCandidate:
    """A candidate tool generated from failure analysis."""
    program: Program
    source: str  # how it was generated
    verification_score: float = 0.0
    verified: bool = False


@dataclass
class FailureAnalysis:
    """Analysis of why a composition plan failed."""
    plan: CompositionPlan
    failure_step: int  # which call failed (-1 if parse failure)
    failure_type: str  # "missing_sub" | "type_mismatch" | "execution_error" | "wrong_answer"
    missing_operation: Optional[str] = None  # what operation seems needed
    available_env: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildStats:
    """Budget and result tracking for test-time tool building."""
    candidates_generated: int = 0
    candidates_verified: int = 0
    candidates_accepted: int = 0
    verification_attempts: int = 0
    compositions_retried: int = 0
    recovered: bool = False
    final_result: Any = None
    budget_exhausted: bool = False


class TestTimeToolBuilder:
    """Builds and verifies new tools at inference time.

    Given a problem where existing library tools fail, this module:
    1. Tries all top-K compositions with existing library
    2. If all fail, analyzes failure patterns
    3. Generates candidate new tools from fragments
    4. Verifies candidates via execution
    5. Re-attempts with expanded library
    """

    def __init__(
        self,
        library: SubroutineLibrary,
        model=None,
        tokenizer=None,
        max_new_tools: int = 3,
        max_verify_attempts: int = 10,
        max_retry_compositions: int = 5,
        answer_tolerance: float = 1e-3,
    ):
        self.base_library = library
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tools = max_new_tools
        self.max_verify_attempts = max_verify_attempts
        self.max_retry_compositions = max_retry_compositions
        self.answer_tolerance = answer_tolerance
        self.executor = Executor()

    def solve_with_building(
        self,
        problem: str,
        bindings: Dict[str, Any],
        failed_plans: List[CompositionPlan],
        gold_answer: Optional[float] = None,
    ) -> Tuple[Optional[CompositionPlan], BuildStats]:
        """Main entry: try to solve by building new tools.

        Args:
            problem: problem text
            bindings: slot bindings
            failed_plans: plans that already failed with the base library
            gold_answer: if available (for training), used for verification

        Returns:
            (successful_plan_or_None, build_stats)
        """
        stats = BuildStats()

        # Create a per-problem library copy
        local_library = self._copy_library(self.base_library)

        # Analyze failures
        analyses = []
        for plan in failed_plans:
            analysis = self._analyze_failure(plan, bindings, local_library)
            analyses.append(analysis)

        # Generate candidate tools from failure patterns
        candidates = self._generate_candidates(analyses, problem, bindings)
        stats.candidates_generated = len(candidates)

        # Verify and add candidates
        for candidate in candidates:
            if stats.candidates_accepted >= self.max_new_tools:
                break
            if stats.verification_attempts >= self.max_verify_attempts:
                stats.budget_exhausted = True
                break

            verified, score = self._verify_candidate(
                candidate, bindings, gold_answer,
            )
            stats.verification_attempts += 1
            stats.candidates_verified += 1
            candidate.verification_score = score
            candidate.verified = verified

            if verified:
                sub = local_library.mint_subroutine(
                    program=candidate.program,
                    support=1,
                    mdl_gain=score,
                )
                if sub is not None:
                    stats.candidates_accepted += 1
                    logger.info(
                        f"Test-time tool accepted: {sub.sub_id} "
                        f"(source={candidate.source}, score={score:.2f})"
                    )

        if stats.candidates_accepted == 0:
            return None, stats

        # Re-attempt composition with expanded library
        comp_exec = CompositionExecutor(local_library)

        # Re-try failed plans with new tools
        for plan in failed_plans[:self.max_retry_compositions]:
            stats.compositions_retried += 1
            # Try the original plan (might work now with new tools available)
            success, result, _ = comp_exec.execute(plan, bindings)
            if success and result is not None:
                if gold_answer is not None:
                    try:
                        if abs(float(result) - float(gold_answer)) < self.answer_tolerance * max(abs(float(gold_answer)), 1.0):
                            stats.recovered = True
                            stats.final_result = result
                            return plan, stats
                    except (ValueError, TypeError):
                        pass
                else:
                    try:
                        if math.isfinite(float(result)):
                            stats.recovered = True
                            stats.final_result = result
                            return plan, stats
                    except (ValueError, TypeError):
                        pass

        # Try generating new plans using the expanded library
        if self.model is not None and self.tokenizer is not None:
            new_plans = self._generate_new_plans(problem, local_library)
            for plan in new_plans[:self.max_retry_compositions]:
                stats.compositions_retried += 1
                success, result, _ = comp_exec.execute(plan, bindings)
                if success and result is not None:
                    if gold_answer is not None:
                        try:
                            if abs(float(result) - float(gold_answer)) < self.answer_tolerance * max(abs(float(gold_answer)), 1.0):
                                stats.recovered = True
                                stats.final_result = result
                                return plan, stats
                        except (ValueError, TypeError):
                            pass
                    else:
                        try:
                            if math.isfinite(float(result)):
                                stats.recovered = True
                                stats.final_result = result
                                return plan, stats
                        except (ValueError, TypeError):
                            pass

        return None, stats

    def _copy_library(self, library: SubroutineLibrary) -> SubroutineLibrary:
        """Deep copy a library for per-problem modification."""
        new_lib = SubroutineLibrary()
        for sid, sub in library.subroutines.items():
            new_sub = Subroutine(
                sub_id=sub.sub_id,
                program=copy.deepcopy(sub.program),
                support=sub.support,
                mdl_gain=sub.mdl_gain,
            )
            new_lib.subroutines[sid] = new_sub
            new_lib._fp_index[sub.program.fingerprint()] = sid
        return new_lib

    def _analyze_failure(
        self,
        plan: CompositionPlan,
        bindings: Dict[str, Any],
        library: SubroutineLibrary,
    ) -> FailureAnalysis:
        """Analyze why a composition plan failed."""
        env = dict(bindings)

        for i, call in enumerate(plan.calls):
            sub_id = call.get("sub_id", "")
            sub = library.get(sub_id)

            if sub is None:
                return FailureAnalysis(
                    plan=plan, failure_step=i, failure_type="missing_sub",
                    missing_operation=sub_id, available_env=dict(env),
                )

            # Try to build bindings
            call_bindings = {}
            type_mismatch = False
            for slot in sub.program.slots:
                if slot.name in call.get("bindings", {}):
                    val = call["bindings"][slot.name]
                    if not DType.check(val, slot.dtype):
                        type_mismatch = True
                    call_bindings[slot.name] = val
                elif slot.name in env:
                    call_bindings[slot.name] = env[slot.name]
                else:
                    candidates = [(k, v) for k, v in env.items()
                                  if not k.startswith("_") and DType.check(v, slot.dtype)]
                    if candidates:
                        call_bindings[slot.name] = candidates[-1][1]
                    else:
                        return FailureAnalysis(
                            plan=plan, failure_step=i, failure_type="type_mismatch",
                            missing_operation=f"need {slot.dtype.value} for {slot.name}",
                            available_env=dict(env),
                        )

            if type_mismatch:
                return FailureAnalysis(
                    plan=plan, failure_step=i, failure_type="type_mismatch",
                    available_env=dict(env),
                )

            # Try execution
            success, result, call_env = self.executor.execute(sub.program, call_bindings)
            if not success:
                return FailureAnalysis(
                    plan=plan, failure_step=i, failure_type="execution_error",
                    available_env=dict(env),
                )

            env.update(call_env)
            if result is not None:
                env["__last_output__"] = result

        return FailureAnalysis(
            plan=plan, failure_step=-1, failure_type="wrong_answer",
            available_env=dict(env),
        )

    def _generate_candidates(
        self,
        analyses: List[FailureAnalysis],
        problem: str,
        bindings: Dict[str, Any],
    ) -> List[ToolCandidate]:
        """Generate candidate tools from failure analyses."""
        candidates = []

        # Strategy 1: Combine partial successes from different failed plans
        partial_envs = [a.available_env for a in analyses if a.failure_step > 0]
        if len(partial_envs) >= 2:
            candidate = self._combine_partial_results(partial_envs, problem, bindings)
            if candidate is not None:
                candidates.append(candidate)

        # Strategy 2: Create bridge tool between failure point and goal
        for analysis in analyses:
            if analysis.failure_type == "wrong_answer":
                candidate = self._create_bridge_tool(analysis, problem, bindings)
                if candidate is not None:
                    candidates.append(candidate)

        # Strategy 3: Create type-conversion tool for type mismatches
        for analysis in analyses:
            if analysis.failure_type == "type_mismatch" and analysis.missing_operation:
                candidate = self._create_type_adapter(analysis)
                if candidate is not None:
                    candidates.append(candidate)

        # Strategy 4: Model-based tool generation (if model available)
        if self.model is not None and self.tokenizer is not None:
            model_candidates = self._model_generate_tools(analyses, problem, bindings)
            candidates.extend(model_candidates)

        return candidates

    def _combine_partial_results(
        self,
        partial_envs: List[Dict[str, Any]],
        problem: str,
        bindings: Dict[str, Any],
    ) -> Optional[ToolCandidate]:
        """Try to create a tool that combines partial results."""
        # Find numeric values that appear in partial environments
        values = set()
        for env in partial_envs:
            for k, v in env.items():
                if k.startswith("_"):
                    continue
                try:
                    values.add(float(v))
                except (ValueError, TypeError):
                    pass

        if len(values) < 2:
            return None

        # Create a simple aggregation tool
        vals = sorted(values)
        slots = [Slot(name=f"v{i}", dtype=DType.FLOAT) for i in range(min(len(vals), 3))]
        slot_names = [s.name for s in slots]

        # Try common aggregation operations
        for op_name, expr in [
            ("sum", " + ".join(slot_names)),
            ("product", " * ".join(slot_names)),
            ("difference", f"{slot_names[0]} - {slot_names[1]}" if len(slot_names) >= 2 else None),
        ]:
            if expr is None:
                continue
            steps = [Step(
                op=Op.OUTPUT, target="result", expr=expr,
                inputs=slot_names, target_dtype=DType.FLOAT,
            )]
            program = Program(
                program_id=f"tt_combine_{op_name}",
                slots=slots, steps=steps,
            )
            candidate = ToolCandidate(program=program, source=f"combine_{op_name}")
            return candidate

        return None

    def _create_bridge_tool(
        self,
        analysis: FailureAnalysis,
        problem: str,
        bindings: Dict[str, Any],
    ) -> Optional[ToolCandidate]:
        """Create a tool that bridges from available values to the answer."""
        last_output = analysis.available_env.get("__last_output__")
        if last_output is None:
            return None

        # Extract numbers from problem for potential computation targets
        numbers = re.findall(r'[\d,]+\.?\d*', problem)
        nums = []
        for n in numbers:
            cleaned = n.replace(",", "")
            try:
                nums.append(float(cleaned))
            except ValueError:
                continue

        if not nums:
            return None

        # Create a parameterized correction tool
        slots = [
            Slot(name="base_value", dtype=DType.FLOAT),
            Slot(name="factor", dtype=DType.FLOAT),
        ]
        steps = [
            Step(op=Op.COMPUTE, target="adjusted",
                 expr="base_value * factor", inputs=["base_value", "factor"],
                 target_dtype=DType.FLOAT),
            Step(op=Op.OUTPUT, target="result",
                 expr="adjusted", inputs=["adjusted"],
                 target_dtype=DType.FLOAT),
        ]
        program = Program(
            program_id="tt_bridge_multiply",
            slots=slots, steps=steps,
        )
        return ToolCandidate(program=program, source="bridge_multiply")

    def _create_type_adapter(self, analysis: FailureAnalysis) -> Optional[ToolCandidate]:
        """Create a type conversion tool."""
        if not analysis.missing_operation:
            return None

        # Parse the needed type
        match = re.search(r'need (\w+)', analysis.missing_operation)
        if not match:
            return None

        target_type = match.group(1)
        dtype_map = {"int": DType.INT, "float": DType.FLOAT, "string": DType.STRING}
        target_dtype = dtype_map.get(target_type, DType.FLOAT)

        slots = [Slot(name="input_val", dtype=DType.FLOAT)]
        steps = [Step(
            op=Op.OUTPUT, target="converted",
            expr=f"{target_type}(input_val)" if target_type in ("int", "float") else "str(input_val)",
            inputs=["input_val"], target_dtype=target_dtype,
        )]
        program = Program(
            program_id=f"tt_convert_to_{target_type}",
            slots=slots, steps=steps,
        )
        return ToolCandidate(program=program, source=f"type_adapter_{target_type}")

    def _model_generate_tools(
        self,
        analyses: List[FailureAnalysis],
        problem: str,
        bindings: Dict[str, Any],
    ) -> List[ToolCandidate]:
        """Use the planner model to generate candidate tools."""
        if self.model is None or self.tokenizer is None:
            return []

        # Summarize failures for the model
        failure_summary = []
        for a in analyses[:3]:
            failure_summary.append(
                f"- Plan failed at step {a.failure_step}: {a.failure_type}"
            )

        lib_sigs = "\n".join(self.base_library.signatures())
        prompt = (
            f"The following subroutines are available:\n{lib_sigs}\n\n"
            f"Problem: {problem}\n\n"
            f"Previous composition attempts failed:\n"
            + "\n".join(failure_summary) + "\n\n"
            f"Design a NEW subroutine that could help solve this problem. "
            f"Output as JSON with fields: slots (list of {{name, dtype}}), "
            f"steps (list of {{op, target, expr, inputs, target_dtype}}).\n"
            f"New subroutine:"
        )

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        import torch
        candidates = []
        with torch.no_grad():
            for _ in range(min(3, self.max_new_tools)):
                output = self.model.generate(
                    **inputs, max_new_tokens=256,
                    do_sample=True, temperature=0.7,
                )
                completion = self.tokenizer.decode(
                    output[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                program = self._parse_tool_output(completion)
                if program is not None:
                    candidates.append(ToolCandidate(
                        program=program, source="model_generated",
                    ))

        return candidates

    def _parse_tool_output(self, text: str) -> Optional[Program]:
        """Parse a model-generated tool specification."""
        # Try to find JSON in the output
        for start in ['{"slots"', '{"program"']:
            idx = text.find(start)
            if idx >= 0:
                candidate = text[idx:]
                try:
                    end = candidate.rindex("}") + 1
                    obj = json.loads(candidate[:end])
                    if "program" in obj:
                        return Program.from_dict(obj["program"])
                    # Direct format
                    slots = [Slot(name=s["name"], dtype=DType(s["dtype"]))
                             for s in obj.get("slots", [])]
                    steps = [Step.from_dict(s) for s in obj.get("steps", [])]
                    if slots and steps:
                        return Program(
                            program_id="tt_model_gen",
                            slots=slots, steps=steps,
                        )
                except (json.JSONDecodeError, ValueError, KeyError):
                    pass
        return None

    def _verify_candidate(
        self,
        candidate: ToolCandidate,
        bindings: Dict[str, Any],
        gold_answer: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """Verify a candidate tool via execution.

        Tests:
        1. Does it parse and type-check?
        2. Does it execute without errors on given bindings?
        3. Does it produce a finite numeric result?
        4. (If gold available) Does combining it help reach the answer?
        """
        program = candidate.program
        score = 0.0

        # Test 1: Can it execute at all?
        test_bindings = {}
        for slot in program.slots:
            if slot.name in bindings:
                test_bindings[slot.name] = bindings[slot.name]
            else:
                # Use a default value matching the type
                defaults = {DType.INT: 1, DType.FLOAT: 1.0, DType.STRING: "x",
                            DType.BOOL: True, DType.LIST_INT: [1], DType.LIST_FLOAT: [1.0]}
                test_bindings[slot.name] = defaults.get(slot.dtype, 0)

        success, result, env = self.executor.execute(program, test_bindings)
        if not success:
            return False, 0.0
        score += 0.3

        # Test 2: Finite numeric result?
        if result is not None:
            try:
                val = float(result)
                if math.isfinite(val):
                    score += 0.3
            except (ValueError, TypeError):
                pass

        # Test 3: Deterministic? (run twice, same result)
        success2, result2, _ = self.executor.execute(program, test_bindings)
        if success2 and result == result2:
            score += 0.2

        # Test 4: Perturbation test (change irrelevant variable)
        perturbed = dict(test_bindings)
        if len(perturbed) > 1:
            first_key = list(perturbed.keys())[0]
            try:
                perturbed[first_key] = float(perturbed[first_key]) + 1.0
            except (ValueError, TypeError):
                pass
            success3, result3, _ = self.executor.execute(program, perturbed)
            if success3 and result3 != result:
                score += 0.2  # The tool actually uses its inputs

        return score >= 0.5, score

    def _generate_new_plans(
        self, problem: str, library: SubroutineLibrary,
    ) -> List[CompositionPlan]:
        """Generate new composition plans using the expanded library."""
        if self.model is None or self.tokenizer is None:
            return []

        lib_sigs = "\n".join(library.signatures())
        prompt = (
            f"Available subroutines:\n{lib_sigs}\n\n"
            f"Problem: {problem}\n\n"
            f"Output a composition plan as JSON:\n"
            f"Plan:"
        )

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        import torch
        plans = []
        with torch.no_grad():
            for _ in range(self.max_retry_compositions):
                output = self.model.generate(
                    **inputs, max_new_tokens=384,
                    do_sample=True, temperature=0.6,
                )
                completion = self.tokenizer.decode(
                    output[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                plan = self._parse_plan(completion)
                if plan is not None:
                    plans.append(plan)

        return plans

    @staticmethod
    def _parse_plan(text: str) -> Optional[CompositionPlan]:
        """Parse composition plan from text."""
        text = text.strip()
        for start_tok in ['{"plan"', '[{"sub_id"']:
            idx = text.find(start_tok)
            if idx >= 0:
                candidate = text[idx:]
                try:
                    if candidate.startswith("["):
                        obj = json.loads(candidate[:candidate.rindex("]") + 1])
                        return CompositionPlan(calls=obj)
                    else:
                        obj = json.loads(candidate[:candidate.rindex("}") + 1])
                        return CompositionPlan.from_dict(obj)
                except (json.JSONDecodeError, ValueError):
                    pass
        return None
