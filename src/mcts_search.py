"""MCTS over Composition Plans for Test-Time Search.

Searches the space of subroutine compositions at inference time.
Each node represents a partial composition plan; actions are
(subroutine_id, binding) pairs. Uses the planner model as policy
prior and execution-based rewards.

This is the key innovation: instead of greedily sampling one plan,
we search the exponential composition space guided by neural policy.
"""

import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.template_dsl import (
    CompositionExecutor,
    CompositionPlan,
    DType,
    Executor,
    SubroutineLibrary,
)


@dataclass
class MCTSNode:
    """A node in the MCTS tree over composition plans."""
    partial_plan: List[Dict[str, Any]]
    env: Dict[str, Any]  # current variable environment
    parent: Optional["MCTSNode"] = None
    children: Dict[str, "MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0
    prior: float = 0.0  # policy prior from model
    is_terminal: bool = False
    result: Any = None

    @property
    def q_value(self) -> float:
        return self.value / max(self.visits, 1)

    def ucb1(self, c_puct: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.q_value
        exploration = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploitation + exploration


class MCTSPlanner:
    """Monte Carlo Tree Search over composition plans.

    Given a problem, searches the space of subroutine call sequences
    to find a composition plan that produces the correct answer.
    """

    def __init__(
        self,
        library: SubroutineLibrary,
        model=None,
        tokenizer=None,
        max_calls: int = 5,
        max_simulations: int = 50,
        c_puct: float = 1.41,
        rollout_depth: int = 3,
    ):
        self.library = library
        self.model = model
        self.tokenizer = tokenizer
        self.executor = Executor()
        self.comp_exec = CompositionExecutor(library, max_calls=max_calls)
        self.max_calls = max_calls
        self.max_simulations = max_simulations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth

    def search(
        self,
        problem: str,
        initial_bindings: Dict[str, Any],
    ) -> Tuple[CompositionPlan, Dict[str, Any]]:
        """Run MCTS to find the best composition plan.

        Uses LABEL-FREE reward only (execution success + plan structure).
        Returns (best_plan, search_stats).
        """
        root = MCTSNode(
            partial_plan=[],
            env=dict(initial_bindings),
        )

        stats = {
            "simulations": 0,
            "solutions_found": 0,
            "max_depth": 0,
            "best_value": 0.0,
        }

        for sim in range(self.max_simulations):
            node = self._select(root)

            if not node.is_terminal and node.visits > 0:
                self._expand(node, problem)

            reward = self._rollout(node, problem)

            # 4. Backpropagate: update values
            self._backpropagate(node, reward)

            stats["simulations"] = sim + 1
            if reward > 0:
                stats["solutions_found"] += 1

        # Select best plan from root's children
        best_plan, best_value = self._best_plan(root)
        stats["best_value"] = best_value
        stats["max_depth"] = self._max_depth(root)

        return best_plan, stats

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select leaf node using UCB1."""
        while node.children and not node.is_terminal:
            node = max(node.children.values(), key=lambda n: n.ucb1(self.c_puct))
        return node

    def _expand(self, node: MCTSNode, problem: str):
        """Expand node by adding children for each possible subroutine call."""
        if len(node.partial_plan) >= self.max_calls:
            node.is_terminal = True
            return

        # Get policy priors from model (if available)
        priors = self._get_priors(node, problem)

        for sub_id, sub in self.library.subroutines.items():
            action_key = sub_id

            # Build bindings from current environment
            bindings = {}
            for slot in sub.program.slots:
                if slot.name in node.env:
                    bindings[slot.name] = node.env[slot.name]
                elif "__last_output__" in node.env:
                    bindings[slot.name] = node.env["__last_output__"]
                else:
                    # Use type-matched variable from env
                    candidates = [(k, v) for k, v in node.env.items()
                                  if not k.startswith("_") and DType.check(v, slot.dtype)]
                    if candidates:
                        bindings[slot.name] = candidates[-1][1]
                    else:
                        bindings[slot.name] = 0.0

            # Try executing this subroutine call
            success, result, call_env = self.executor.execute(sub.program, bindings)
            if not success:
                continue  # skip invalid calls

            new_env = dict(node.env)
            new_env.update(call_env)
            if result is not None:
                new_env["__last_output__"] = result

            new_plan = node.partial_plan + [{"sub_id": sub_id, "bindings": bindings}]

            child = MCTSNode(
                partial_plan=new_plan,
                env=new_env,
                parent=node,
                prior=priors.get(sub_id, 1.0 / max(self.library.size, 1)),
                result=result,
            )
            node.children[action_key] = child

        # Also add a "STOP" action (terminate the plan here)
        if node.partial_plan:  # must have at least one call
            stop_key = "STOP"
            child = MCTSNode(
                partial_plan=list(node.partial_plan),
                env=dict(node.env),
                parent=node,
                prior=priors.get("STOP", 0.1),
                is_terminal=True,
                result=node.env.get("__last_output__"),
            )
            node.children[stop_key] = child

    def _get_priors(self, node: MCTSNode, problem: str) -> Dict[str, float]:
        """Get policy priors from the planner model.

        Falls back to uniform distribution if model is unavailable.
        """
        if self.model is None or self.tokenizer is None:
            # Uniform prior
            uniform = 1.0 / max(self.library.size + 1, 1)
            priors = {sid: uniform for sid in self.library.subroutines}
            priors["STOP"] = uniform
            return priors

        # Use model to score each possible next action
        lib_sigs = "\n".join(self.library.signatures())
        current_plan = [c["sub_id"] for c in node.partial_plan]
        plan_so_far = " -> ".join(current_plan) if current_plan else "(empty)"

        prompt = (
            f"Available subroutines:\n{lib_sigs}\n\n"
            f"Problem: {problem}\n\n"
            f"Plan so far: {plan_so_far}\n\n"
            f"What subroutine should be called next? Output just the ID (e.g., L00) or STOP:"
        )

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=10, do_sample=True,
                temperature=0.5, top_k=self.library.size + 1,
                return_dict_in_generate=True, output_scores=True,
            )

        response = self.tokenizer.decode(
            output.sequences[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Parse response into priors
        priors = {}
        base_prior = 0.5 / max(self.library.size, 1)
        for sid in self.library.subroutines:
            priors[sid] = base_prior

        # Boost the predicted subroutine
        predicted = re.search(r'L\d+', response)
        if predicted and predicted.group() in self.library.subroutines:
            priors[predicted.group()] = 0.5
        if "STOP" in response.upper():
            priors["STOP"] = 0.3
        else:
            priors["STOP"] = 0.05

        # Normalize
        total = sum(priors.values())
        return {k: v / total for k, v in priors.items()}

    def _rollout(self, node: MCTSNode, problem: str) -> float:
        """Simulate random completion from this node and evaluate."""
        env = dict(node.env)
        plan = list(node.partial_plan)

        # Random rollout: add random subroutine calls
        subs = list(self.library.subroutines.values())
        for _ in range(self.rollout_depth):
            if len(plan) >= self.max_calls:
                break

            sub = random.choice(subs)
            bindings = {}
            for slot in sub.program.slots:
                if slot.name in env:
                    bindings[slot.name] = env[slot.name]
                elif "__last_output__" in env:
                    bindings[slot.name] = env["__last_output__"]
                else:
                    candidates = [(k, v) for k, v in env.items()
                                  if not k.startswith("_") and DType.check(v, slot.dtype)]
                    bindings[slot.name] = candidates[-1][1] if candidates else 0.0

            success, result, call_env = self.executor.execute(sub.program, bindings)
            if success:
                env.update(call_env)
                if result is not None:
                    env["__last_output__"] = result
                plan.append({"sub_id": sub.sub_id, "bindings": bindings})

        # Evaluate the completed plan using LABEL-FREE criteria only.
        # We NEVER use the gold answer during search — that would be cheating.
        # Instead, reward plans that: (1) execute successfully, (2) produce
        # a finite numeric result, (3) use multiple subroutine calls.
        result = env.get("__last_output__")
        if result is None:
            return 0.0

        reward = 0.0
        try:
            val = float(result)
            if not math.isfinite(val):
                return 0.0
            # Successfully produced a finite number
            reward += 0.4
            # Prefer plans with multiple calls (actual composition)
            reward += 0.3 * min(len(plan) / 3.0, 1.0)
            # Prefer non-trivial results (not 0 or 1)
            if abs(val) > 1e-6 and abs(val - 1.0) > 1e-6:
                reward += 0.2
            # Prefer results in a reasonable numeric range for math problems
            if 0.01 < abs(val) < 1e8:
                reward += 0.1
        except (ValueError, TypeError):
            return 0.0

        return reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Update values up the tree."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _best_plan(self, root: MCTSNode) -> Tuple[CompositionPlan, float]:
        """Select the best plan from root's children by visit count."""
        if not root.children:
            return CompositionPlan(calls=[]), 0.0

        # Find the most-visited child path
        best_child = max(root.children.values(), key=lambda n: n.visits)

        # If the best child has children, recurse
        plan_calls = list(best_child.partial_plan)

        # Walk down the tree following most-visited children
        node = best_child
        while node.children:
            node = max(node.children.values(), key=lambda n: n.visits)
            if node.partial_plan != plan_calls:
                plan_calls = list(node.partial_plan)

        return CompositionPlan(calls=plan_calls), best_child.q_value

    def _max_depth(self, node: MCTSNode, depth: int = 0) -> int:
        """Find maximum depth of the tree."""
        if not node.children:
            return depth
        return max(self._max_depth(c, depth + 1) for c in node.children.values())


def mcts_solve(
    problem: str,
    library: SubroutineLibrary,
    model=None,
    tokenizer=None,
    max_simulations: int = 50,
    max_calls: int = 5,
) -> Tuple[CompositionPlan, Any, Dict]:
    """High-level API: solve a problem using MCTS over compositions.

    Uses label-free reward (no gold answer). Returns (plan, result, stats).
    """
    # Extract numbers from problem for initial bindings
    raw_numbers = re.findall(r'[\d,]+\.?\d*', problem)
    numbers = []
    for n in raw_numbers:
        cleaned = n.replace(",", "").strip()
        if cleaned:
            try:
                numbers.append(float(cleaned))
            except ValueError:
                continue

    # Build initial bindings using library's most common slot names
    all_slot_names = []
    for sub in library.subroutines.values():
        for slot in sub.program.slots:
            if slot.name not in all_slot_names:
                all_slot_names.append(slot.name)

    initial_bindings = {}
    for j, name in enumerate(all_slot_names):
        if j < len(numbers):
            initial_bindings[name] = numbers[j]

    # Also add positional bindings
    for j, num in enumerate(numbers):
        initial_bindings[f"x{j}"] = num

    planner = MCTSPlanner(
        library=library,
        model=model,
        tokenizer=tokenizer,
        max_calls=max_calls,
        max_simulations=max_simulations,
    )

    plan, stats = planner.search(problem, initial_bindings)

    # Execute the best plan
    comp_exec = CompositionExecutor(library, max_calls=max_calls)
    success, result, exec_stats = comp_exec.execute(plan, initial_bindings)

    stats["execution_success"] = success
    stats["result"] = result

    return plan, result, stats
