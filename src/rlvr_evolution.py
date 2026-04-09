"""RLVR-Driven Library Evolution Engine (SEVAL Core).

Implements the Self-Evolving Verified Abstraction Libraries (SEVAL) training loop:
  1. GRPO trains a composition planner with verifiable execution rewards
  2. Periodically, successful composition patterns are analyzed
  3. Recurring verified patterns are abstracted into new library subroutines
  4. The library grows: L₀ → L₁ → L₂ → ... (the "abstraction flywheel")

Uses TRL's GRPOTrainer with a custom reward function based on
CompositionExecutor execution success.
"""

import json
import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.template_dsl import (
    CompositionExecutor,
    CompositionPlan,
    DType,
    Executor,
    Program,
    Slot,
    Step,
    Subroutine,
    SubroutineLibrary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward function: verifiable composition execution
# ---------------------------------------------------------------------------

@dataclass
class RewardResult:
    reward: float
    plan_parsed: bool
    execution_success: bool
    answer_correct: bool
    plan: Optional[CompositionPlan] = None
    result: Any = None


class CompositionReward:
    """Verifiable reward for GRPO: execute plan, check answer.

    reward = 1.0 if plan parses, executes, and produces correct answer
    reward = 0.5 if plan parses and executes but wrong answer
    reward = 0.1 if plan parses but fails execution
    reward = 0.0 if plan doesn't parse
    """

    def __init__(
        self,
        library: SubroutineLibrary,
        max_calls: int = 10,
        answer_tolerance: float = 1e-3,
    ):
        self.library = library
        self.comp_exec = CompositionExecutor(library, max_calls=max_calls)
        self.answer_tolerance = answer_tolerance

    def update_library(self, library: SubroutineLibrary):
        """Update the reward function's library (after evolution step)."""
        self.library = library
        self.comp_exec = CompositionExecutor(library, max_calls=self.comp_exec.max_calls)

    def __call__(
        self,
        plan_text: str,
        bindings: Dict[str, Any],
        gold_answer: Optional[float] = None,
    ) -> RewardResult:
        # 1. Parse plan from model output
        plan = self._parse_plan(plan_text)
        if plan is None:
            return RewardResult(reward=0.0, plan_parsed=False,
                                execution_success=False, answer_correct=False)

        # 2. Execute
        success, result, stats = self.comp_exec.execute(plan, bindings)
        if not success:
            return RewardResult(reward=0.1, plan_parsed=True,
                                execution_success=False, answer_correct=False,
                                plan=plan)

        # 3. Check answer
        if gold_answer is not None and result is not None:
            try:
                pred = float(result)
                gold = float(gold_answer)
                if math.isfinite(pred) and math.isfinite(gold):
                    if abs(pred - gold) < self.answer_tolerance * max(abs(gold), 1.0):
                        return RewardResult(
                            reward=1.0, plan_parsed=True,
                            execution_success=True, answer_correct=True,
                            plan=plan, result=result,
                        )
            except (ValueError, TypeError):
                pass

        # Execution succeeded but answer wrong (or no gold to check)
        return RewardResult(
            reward=0.5 if gold_answer is None else 0.0,
            plan_parsed=True, execution_success=True,
            answer_correct=False, plan=plan, result=result,
        )

    def _parse_plan(self, text: str) -> Optional[CompositionPlan]:
        """Extract a composition plan JSON from model output text."""
        # Try direct JSON parse
        text = text.strip()
        for start_tok in ['{"plan"', '[{"sub_id"']:
            idx = text.find(start_tok)
            if idx >= 0:
                candidate = text[idx:]
                # Find matching brace/bracket
                try:
                    if candidate.startswith("["):
                        obj = json.loads(candidate[:candidate.rindex("]") + 1])
                        return CompositionPlan(calls=obj)
                    else:
                        obj = json.loads(candidate[:candidate.rindex("}") + 1])
                        return CompositionPlan.from_dict(obj)
                except (json.JSONDecodeError, ValueError):
                    pass

        # Try extracting from code block
        match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(1))
                if isinstance(obj, list):
                    return CompositionPlan(calls=obj)
                return CompositionPlan.from_dict(obj)
            except (json.JSONDecodeError, ValueError):
                pass

        return None


# ---------------------------------------------------------------------------
# GRPO reward wrapper for TRL
# ---------------------------------------------------------------------------

def make_reward_fn(
    reward: CompositionReward,
    problem_map: Dict[str, Dict[str, Any]],
):
    """Create a reward function compatible with TRL's GRPOTrainer.

    Args:
        reward: CompositionReward instance
        problem_map: mapping from problem_id to {bindings, gold_answer}
    """
    def reward_fn(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
        rewards = []
        for i, completion in enumerate(completions):
            # Extract problem_id from prompt or kwargs
            problem_id = kwargs.get("problem_ids", [None] * len(completions))[i]
            info = problem_map.get(problem_id, {})
            bindings = info.get("bindings", {})
            gold = info.get("gold_answer")

            result = reward(completion, bindings, gold)
            rewards.append(result.reward)
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Library Evolver: the "abstraction flywheel"
# ---------------------------------------------------------------------------

@dataclass
class CompositionPattern:
    """A recurring composition subpattern observed in successful plans."""
    sub_ids: Tuple[str, ...]  # ordered subroutine IDs in the pattern
    count: int = 0
    success_rate: float = 0.0
    examples: List[CompositionPlan] = field(default_factory=list)
    mdl_gain: float = 0.0


class LibraryEvolver:
    """Evolves the subroutine library from successful GRPO compositions.

    Watches successful composition plans, identifies recurring subpatterns,
    abstracts them into new verified subroutines, and adds them to the library.
    """

    def __init__(
        self,
        library: SubroutineLibrary,
        min_pattern_count: int = 5,
        min_success_rate: float = 0.7,
        min_mdl_gain: float = 0.5,
        max_library_size: int = 64,
        verify_holdout_n: int = 10,
    ):
        self.library = library
        self.min_pattern_count = min_pattern_count
        self.min_success_rate = min_success_rate
        self.min_mdl_gain = min_mdl_gain
        self.max_library_size = max_library_size
        self.verify_holdout_n = verify_holdout_n
        self.executor = Executor()

        # Buffers
        self.successful_plans: List[Tuple[CompositionPlan, Dict[str, Any]]] = []
        self.failed_plans: List[Tuple[CompositionPlan, Dict[str, Any]]] = []
        self.evolution_history: List[Dict[str, Any]] = []

    def record(self, plan: CompositionPlan, bindings: Dict[str, Any], success: bool):
        """Record a composition attempt for later pattern analysis."""
        if success:
            self.successful_plans.append((plan, bindings))
        else:
            self.failed_plans.append((plan, bindings))

    def clear_buffers(self):
        self.successful_plans.clear()
        self.failed_plans.clear()

    def find_patterns(self) -> List[CompositionPattern]:
        """Find recurring subpatterns in successful compositions."""
        # Count subroutine bigrams and trigrams
        bigram_counter: Counter = Counter()
        trigram_counter: Counter = Counter()
        bigram_examples: Dict[Tuple[str, ...], List] = defaultdict(list)

        total_plans = len(self.successful_plans)
        if total_plans < self.min_pattern_count:
            return []

        for plan, bindings in self.successful_plans:
            ids = plan.subroutine_ids
            for i in range(len(ids) - 1):
                bg = (ids[i], ids[i + 1])
                bigram_counter[bg] += 1
                bigram_examples[bg].append(plan)
            for i in range(len(ids) - 2):
                tg = (ids[i], ids[i + 1], ids[i + 2])
                trigram_counter[tg] += 1

        # Also count in failed plans (for success rate)
        fail_bigram_counter: Counter = Counter()
        for plan, bindings in self.failed_plans:
            ids = plan.subroutine_ids
            for i in range(len(ids) - 1):
                bg = (ids[i], ids[i + 1])
                fail_bigram_counter[bg] += 1

        patterns = []
        for bg, count in bigram_counter.most_common(20):
            if count < self.min_pattern_count:
                break
            total_attempts = count + fail_bigram_counter.get(bg, 0)
            success_rate = count / total_attempts if total_attempts > 0 else 0
            if success_rate < self.min_success_rate:
                continue

            # Estimate MDL gain: how much does this bigram compress solutions?
            mdl_gain = self._estimate_mdl_gain(bg)
            if mdl_gain < self.min_mdl_gain:
                continue

            patterns.append(CompositionPattern(
                sub_ids=bg, count=count, success_rate=success_rate,
                examples=bigram_examples[bg][:5], mdl_gain=mdl_gain,
            ))

        # Sort by MDL gain descending
        patterns.sort(key=lambda p: p.mdl_gain, reverse=True)
        return patterns

    def _estimate_mdl_gain(self, sub_ids: Tuple[str, ...]) -> float:
        """Estimate MDL gain of abstracting a subroutine sequence."""
        total_steps = 0
        for sid in sub_ids:
            sub = self.library.get(sid)
            if sub is None:
                return 0.0
            total_steps += len(sub.program.steps)

        # MDL gain ≈ (steps_saved_per_use * usage_count) - description_cost
        # A composed subroutine saves (total_steps - 1) steps per use
        # Description cost = total_steps of the new subroutine
        steps_saved_per_use = total_steps - 1  # one call replaces many steps
        # Conservative: count in buffer only
        usage_count = sum(
            1 for plan, _ in self.successful_plans
            if self._contains_subsequence(plan.subroutine_ids, sub_ids)
        )
        description_cost = total_steps

        return steps_saved_per_use * usage_count - description_cost

    @staticmethod
    def _contains_subsequence(seq: List[str], subseq: Tuple[str, ...]) -> bool:
        n, m = len(seq), len(subseq)
        for i in range(n - m + 1):
            if tuple(seq[i:i + m]) == subseq:
                return True
        return False

    def abstract_pattern(self, pattern: CompositionPattern) -> Optional[Program]:
        """Convert a composition pattern into a PARAMETERIZED new subroutine.

        Key insight (addresses reviewer W1): simple concatenation of two
        subroutines is equivalent to what the planner already does. True
        evolution requires GENERALIZATION — extracting the common structure
        across multiple usage instances and abstracting varying constants
        into new typed slots.

        Algorithm:
        1. Collect all successful uses of this bigram pattern
        2. For each binding that VARIES across uses → create a new slot
        3. For each binding that is CONSTANT → inline the value
        4. Chain the constituent subroutines with the new slot structure
        5. This creates a genuinely new operation (parameterized composition)
           that the original library cannot express in a single call
        """
        if not pattern.examples:
            return self._fallback_chain(pattern)

        # Step 1: Collect binding values across all uses of this pattern
        # For each subroutine call in the pattern, collect what values
        # were bound to each slot across different successful uses
        binding_values: Dict[str, List[Any]] = defaultdict(list)

        for plan in pattern.examples:
            for call in plan.calls:
                sub_id = call.get("sub_id", "")
                if sub_id not in pattern.sub_ids:
                    continue
                for slot_name, value in call.get("bindings", {}).items():
                    key = f"{sub_id}_{slot_name}"
                    binding_values[key].append(value)

        # Step 2: Classify each binding as VARYING or CONSTANT
        new_slots: Dict[str, Slot] = {}
        constant_bindings: Dict[str, Any] = {}
        slot_counter = 0

        for key, values in binding_values.items():
            if not values:
                continue
            unique_vals = set()
            for v in values:
                try:
                    unique_vals.add(float(v))
                except (ValueError, TypeError):
                    unique_vals.add(str(v))

            if len(unique_vals) > 1:
                # VARYING: abstract into a new slot
                slot_name = f"p{slot_counter}"
                slot_counter += 1
                # Infer type from values
                all_int = all(isinstance(v, (int,)) or (isinstance(v, float) and v == int(v))
                              for v in values)
                dtype = DType.INT if all_int else DType.FLOAT
                new_slots[key] = Slot(name=slot_name, dtype=dtype,
                                      description=f"abstracted from {key}")
            else:
                # CONSTANT: inline the value
                constant_bindings[key] = values[0]

        # Step 3: Build the new program with parameterized slots
        all_steps: List[Step] = []
        final_slots: Dict[str, Slot] = {}

        # Map from original (sub_id, slot_name) → new variable name
        var_map: Dict[str, str] = {}

        for key, slot in new_slots.items():
            var_map[key] = slot.name
            final_slots[slot.name] = slot

        for key, value in constant_bindings.items():
            # Create an assignment step for constants
            const_var = f"_const_{len(all_steps)}"
            all_steps.append(Step(
                op=Op.ASSIGN, target=const_var,
                expr=repr(value), inputs=[],
                target_dtype=DType.FLOAT,
            ))
            var_map[key] = const_var

        # Step 4: Chain the subroutine computations
        for idx, sid in enumerate(pattern.sub_ids):
            sub = self.library.get(sid)
            if sub is None:
                return self._fallback_chain(pattern)

            # Build per-step rename map
            rename_map: Dict[str, str] = {}
            for slot in sub.program.slots:
                key = f"{sid}_{slot.name}"
                if key in var_map:
                    rename_map[slot.name] = var_map[key]
                elif slot.name not in final_slots:
                    # Slot not seen in examples — keep as parameter
                    final_slots[slot.name] = slot
                    rename_map[slot.name] = slot.name

            for step in sub.program.steps:
                # Rename internal targets to avoid collision
                internal = step.target not in {s.name for s in sub.program.slots}
                new_target = f"_e{idx}_{step.target}" if internal else rename_map.get(step.target, step.target)

                if internal:
                    rename_map[step.target] = new_target

                new_expr = step.expr
                new_inputs = list(step.inputs)

                # Apply all renames to expression and inputs
                for old_name, new_name in rename_map.items():
                    if old_name != new_name:
                        new_expr = re.sub(
                            r'\b' + re.escape(old_name) + r'\b',
                            new_name, new_expr,
                        )
                        new_inputs = [new_name if inp == old_name else inp
                                      for inp in new_inputs]

                all_steps.append(Step(
                    op=step.op, target=new_target, expr=new_expr,
                    inputs=new_inputs, target_dtype=step.target_dtype,
                ))

                # Track output for chaining
                if step.op == Op.OUTPUT:
                    var_map[f"_output_{idx}"] = new_target

        if not all_steps or not final_slots:
            return self._fallback_chain(pattern)

        new_program = Program(
            program_id=f"evolved_{'-'.join(pattern.sub_ids)}",
            slots=list(final_slots.values()),
            steps=all_steps,
        )
        return new_program

    def _fallback_chain(self, pattern: CompositionPattern) -> Optional[Program]:
        """Simple chain abstraction when parameterized generalization fails."""
        all_slots: Dict[str, Slot] = {}
        all_steps: List[Step] = []

        for idx, sid in enumerate(pattern.sub_ids):
            sub = self.library.get(sid)
            if sub is None:
                return None
            for slot in sub.program.slots:
                if slot.name not in all_slots:
                    all_slots[slot.name] = slot
            for step in sub.program.steps:
                new_target = f"_e{idx}_{step.target}" if step.target not in all_slots else step.target
                new_expr = step.expr
                new_inputs = list(step.inputs)
                for s in sub.program.steps:
                    if s.target not in all_slots:
                        old_name = s.target
                        new_name = f"_e{idx}_{old_name}"
                        new_expr = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, new_expr)
                        new_inputs = [new_name if inp == old_name else inp for inp in new_inputs]
                all_steps.append(Step(
                    op=step.op, target=new_target, expr=new_expr,
                    inputs=new_inputs, target_dtype=step.target_dtype,
                ))

        return Program(
            program_id=f"evolved_{'-'.join(pattern.sub_ids)}",
            slots=list(all_slots.values()),
            steps=all_steps,
        )

    def verify_candidate(
        self,
        candidate: Program,
        holdout_examples: List[Dict[str, Any]],
    ) -> Tuple[bool, float]:
        """Verify a candidate subroutine on held-out examples.

        Returns (verified, success_rate).
        """
        if not holdout_examples:
            return False, 0.0

        successes = 0
        tested = 0
        for example in holdout_examples[:self.verify_holdout_n]:
            bindings = example.get("bindings", {})
            gold = example.get("gold_answer")
            if not bindings:
                continue
            tested += 1
            success, result, env = self.executor.execute(candidate, bindings)
            if success and result is not None and gold is not None:
                try:
                    if abs(float(result) - float(gold)) < 1e-3 * max(abs(float(gold)), 1.0):
                        successes += 1
                except (ValueError, TypeError):
                    pass

        rate = successes / tested if tested > 0 else 0.0
        return rate >= 0.8, rate

    def evolve(
        self,
        holdout_examples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run one evolution step: find patterns → abstract → verify → add.

        Returns evolution stats.
        """
        if self.library.size >= self.max_library_size:
            return {"evolved": False, "reason": "max_library_size reached"}

        patterns = self.find_patterns()
        if not patterns:
            return {"evolved": False, "reason": "no qualifying patterns",
                    "buffer_size": len(self.successful_plans)}

        new_subs = []
        for pattern in patterns:
            if self.library.size >= self.max_library_size:
                break

            candidate = self.abstract_pattern(pattern)
            if candidate is None:
                continue

            verified, rate = self.verify_candidate(candidate, holdout_examples)
            if not verified:
                logger.info(f"Pattern {pattern.sub_ids} failed verification (rate={rate:.2f})")
                continue

            sub = self.library.mint_subroutine(
                program=candidate,
                support=pattern.count,
                mdl_gain=pattern.mdl_gain,
            )
            if sub is not None:
                new_subs.append({
                    "sub_id": sub.sub_id,
                    "source_pattern": list(pattern.sub_ids),
                    "count": pattern.count,
                    "success_rate": pattern.success_rate,
                    "mdl_gain": pattern.mdl_gain,
                    "verification_rate": rate,
                })
                logger.info(
                    f"Evolved new subroutine {sub.sub_id} from "
                    f"{pattern.sub_ids} (count={pattern.count}, "
                    f"mdl_gain={pattern.mdl_gain:.2f}, "
                    f"verify_rate={rate:.2f})"
                )

        step_record = {
            "evolved": len(new_subs) > 0,
            "new_subroutines": new_subs,
            "library_size_before": self.library.size - len(new_subs),
            "library_size_after": self.library.size,
            "patterns_found": len(patterns),
            "buffer_size": len(self.successful_plans),
            "library_snapshot": self.library.snapshot(),
        }
        self.evolution_history.append(step_record)
        return step_record


# ---------------------------------------------------------------------------
# CoT-Pass@K evaluation metric
# ---------------------------------------------------------------------------

def cot_pass_at_k(
    model,
    tokenizer,
    problems: List[Dict[str, Any]],
    library: SubroutineLibrary,
    k_values: List[int] = [1, 4, 16, 64],
    max_samples: int = 64,
    temperature: float = 0.8,
    max_new_tokens: int = 512,
) -> Dict[str, float]:
    """Compute CoT-Pass@K: fraction of problems solved with K samples.

    This measures whether RLVR EXPANDS reasoning capability (higher pass@K
    means the model can reach more solutions in its sampling distribution).
    """
    comp_exec = CompositionExecutor(library)
    reward = CompositionReward(library)

    results = {f"pass@{k}": 0.0 for k in k_values}
    problem_results = []

    for prob in problems:
        question = prob["question"]
        bindings = prob.get("bindings", {})
        gold = prob.get("gold_answer")

        # Build prompt
        lib_sigs = "\n".join(library.signatures())
        prompt = (
            f"Available subroutines:\n{lib_sigs}\n\n"
            f"Problem: {question}\n\n"
            f"Output a composition plan as JSON:"
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Sample max_samples completions
        solved_at = []  # which sample indices solved the problem
        with torch.no_grad():
            for sample_idx in range(max_samples):
                output = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=temperature,
                    top_p=0.95,
                )
                completion = tokenizer.decode(
                    output[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                r = reward(completion, bindings, gold)
                if r.answer_correct:
                    solved_at.append(sample_idx)

        # Compute pass@k for each k
        n = max_samples
        c = len(solved_at)
        prob_results = {}
        for k in k_values:
            if k > n:
                continue
            # pass@k = 1 - C(n-c, k) / C(n, k)
            if c >= k:
                prob_results[f"pass@{k}"] = 1.0
            elif c == 0:
                prob_results[f"pass@{k}"] = 0.0
            else:
                # Exact computation using log to avoid overflow
                log_numerator = sum(math.log(n - c - i) for i in range(k) if n - c - i > 0)
                log_denominator = sum(math.log(n - i) for i in range(k))
                if log_denominator == 0:
                    prob_results[f"pass@{k}"] = 1.0
                else:
                    prob_results[f"pass@{k}"] = 1.0 - math.exp(log_numerator - log_denominator)

        problem_results.append(prob_results)

    # Average across problems
    for k in k_values:
        key = f"pass@{k}"
        vals = [pr.get(key, 0.0) for pr in problem_results]
        results[key] = sum(vals) / len(vals) if vals else 0.0

    return results


# ---------------------------------------------------------------------------
# SEVAL Training Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class SEVALConfig:
    """Configuration for SEVAL training."""
    # GRPO
    grpo_num_generations: int = 8
    grpo_temperature: float = 0.8
    grpo_max_new_tokens: int = 512
    grpo_learning_rate: float = 5e-6
    grpo_num_steps: int = 2000
    grpo_batch_size: int = 4

    # Evolution
    evolution_interval: int = 200  # evolve every N GRPO steps
    evolution_rounds: int = 5  # max evolution rounds
    min_pattern_count: int = 5
    min_success_rate: float = 0.7
    min_mdl_gain: float = 0.5
    max_library_size: int = 64

    # CoT-Pass@K eval
    eval_interval: int = 500
    eval_k_values: List[int] = field(default_factory=lambda: [1, 4, 16, 64])
    eval_max_samples: int = 64
    eval_problems: int = 100

    # Checkpointing
    output_dir: str = "results/seval"
    save_interval: int = 500


class SEVALTrainer:
    """Main SEVAL training loop: GRPO + periodic library evolution.

    This orchestrates:
    1. GRPO training with composition-execution rewards
    2. Periodic library evolution (abstraction flywheel)
    3. CoT-Pass@K evaluation for capability expansion measurement
    """

    def __init__(
        self,
        model,
        tokenizer,
        library: SubroutineLibrary,
        train_data: List[Dict[str, Any]],
        eval_data: List[Dict[str, Any]],
        holdout_data: List[Dict[str, Any]],
        config: SEVALConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.library = library
        self.train_data = train_data
        self.eval_data = eval_data
        self.holdout_data = holdout_data
        self.config = config

        self.reward_fn = CompositionReward(library)
        self.evolver = LibraryEvolver(
            library=library,
            min_pattern_count=config.min_pattern_count,
            min_success_rate=config.min_success_rate,
            min_mdl_gain=config.min_mdl_gain,
            max_library_size=config.max_library_size,
        )

        self.step = 0
        self.evolution_round = 0
        self.metrics_history: List[Dict[str, Any]] = []

        os.makedirs(config.output_dir, exist_ok=True)

    def format_prompt(self, problem: Dict[str, Any]) -> str:
        """Format a problem into the composition planner prompt."""
        lib_sigs = "\n".join(self.library.signatures())
        question = problem["question"]
        return (
            f"Available subroutines:\n{lib_sigs}\n\n"
            f"Problem: {question}\n\n"
            f"Output a composition plan as JSON. Use the format: "
            f'{{"plan": [{{"sub_id": "L00", "bindings": {{"x": 5}}}}]}}\n'
            f"Plan:"
        )

    def compute_rewards(
        self, completions: List[str], problems: List[Dict[str, Any]],
    ) -> List[float]:
        """Compute verifiable rewards for a batch of completions."""
        rewards = []
        for completion, problem in zip(completions, problems):
            bindings = problem.get("bindings", {})
            gold = problem.get("gold_answer")
            result = self.reward_fn(completion, bindings, gold)

            # Record for evolution
            if result.plan is not None:
                self.evolver.record(result.plan, bindings, result.answer_correct)

            rewards.append(result.reward)
        return rewards

    def maybe_evolve(self) -> Optional[Dict[str, Any]]:
        """Check if it's time for library evolution and run if so."""
        if self.step % self.config.evolution_interval != 0:
            return None
        if self.evolution_round >= self.config.evolution_rounds:
            return None
        if self.step == 0:
            return None

        logger.info(f"=== Evolution Round {self.evolution_round + 1} at step {self.step} ===")

        # Run evolution
        evo_result = self.evolver.evolve(self.holdout_data)

        if evo_result["evolved"]:
            self.evolution_round += 1
            # Update reward function with new library
            self.reward_fn.update_library(self.library)
            # Save evolved library
            lib_path = os.path.join(
                self.config.output_dir,
                f"library_L{self.evolution_round}.json",
            )
            self.library.save(lib_path)
            logger.info(
                f"Library evolved: {evo_result['library_size_before']} → "
                f"{evo_result['library_size_after']} subroutines"
            )

        # Clear buffers for next round
        self.evolver.clear_buffers()

        return evo_result

    def maybe_eval(self) -> Optional[Dict[str, float]]:
        """Run CoT-Pass@K evaluation if interval reached."""
        if self.step % self.config.eval_interval != 0:
            return None

        logger.info(f"=== Evaluating CoT-Pass@K at step {self.step} ===")
        eval_problems = self.eval_data[:self.config.eval_problems]
        metrics = cot_pass_at_k(
            model=self.model,
            tokenizer=self.tokenizer,
            problems=eval_problems,
            library=self.library,
            k_values=self.config.eval_k_values,
            max_samples=self.config.eval_max_samples,
        )
        metrics["step"] = self.step
        metrics["library_size"] = self.library.size
        metrics["evolution_round"] = self.evolution_round
        self.metrics_history.append(metrics)

        # Save metrics
        metrics_path = os.path.join(self.config.output_dir, "cot_passk_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        logger.info(f"CoT-Pass@K: {metrics}")
        return metrics

    def save_checkpoint(self):
        """Save training state."""
        ckpt_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save library
        self.library.save(os.path.join(ckpt_dir, "library.json"))

        # Save evolution history
        with open(os.path.join(ckpt_dir, "evolution_history.json"), "w") as f:
            json.dump(self.evolver.evolution_history, f, indent=2)

        # Save metrics
        with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        # Save model (LoRA adapter)
        self.model.save_pretrained(os.path.join(ckpt_dir, "model"))
        self.tokenizer.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))

        logger.info(f"Checkpoint saved at step {self.step}")

    def get_grpo_reward_fn(self):
        """Return a reward function compatible with TRL GRPOTrainer."""
        reward_fn = self.reward_fn
        evolver = self.evolver
        train_data = self.train_data

        def _reward(completions, prompts=None, **kwargs):
            rewards = []
            for i, completion in enumerate(completions):
                # Map prompt back to problem
                problem_idx = kwargs.get("problem_indices", list(range(len(completions))))[i]
                problem = train_data[problem_idx % len(train_data)]
                bindings = problem.get("bindings", {})
                gold = problem.get("gold_answer")

                result = reward_fn(completion, bindings, gold)
                if result.plan is not None:
                    evolver.record(result.plan, bindings, result.answer_correct)
                rewards.append(result.reward)
            return rewards

        return _reward
