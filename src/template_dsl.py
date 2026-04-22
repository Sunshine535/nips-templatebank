"""Typed DSL for mathematical reasoning programs.

Programs are JSON-AST trees over a fixed set of primitive operators.
Templates are programs with typed variable slots that can be composed
via a subroutine library.
"""

import copy
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class DType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    LIST_INT = "list[int]"
    LIST_FLOAT = "list[float]"

    @staticmethod
    def coerce(value: Any, dtype: "DType") -> Any:
        try:
            if dtype == DType.INT:
                return int(float(value))
            elif dtype == DType.FLOAT:
                return float(value)
            elif dtype == DType.STRING:
                return str(value)
            elif dtype == DType.BOOL:
                return bool(value)
            elif dtype in (DType.LIST_INT, DType.LIST_FLOAT):
                if isinstance(value, list):
                    elem_fn = int if dtype == DType.LIST_INT else float
                    return [elem_fn(v) for v in value]
                return [DType.coerce(value, DType.INT if dtype == DType.LIST_INT else DType.FLOAT)]
        except (ValueError, TypeError):
            raise TypeError(f"Cannot coerce {value!r} to {dtype.value}")

    @staticmethod
    def check(value: Any, dtype: "DType") -> bool:
        try:
            DType.coerce(value, dtype)
            return True
        except TypeError:
            return False


class Op(str, Enum):
    ASSIGN = "assign"
    COMPUTE = "compute"
    COMPARE = "compare"
    AGGREGATE = "aggregate"
    CONDITION = "condition"
    OUTPUT = "output"


@dataclass
class Slot:
    name: str
    dtype: DType
    description: str = ""


@dataclass
class Step:
    op: Op
    target: str
    expr: str
    inputs: List[str] = field(default_factory=list)
    target_dtype: DType = DType.FLOAT

    def to_dict(self) -> dict:
        return {
            "op": self.op.value,
            "target": self.target,
            "expr": self.expr,
            "inputs": self.inputs,
            "target_dtype": self.target_dtype.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(
            op=Op(d["op"]),
            target=d["target"],
            expr=d["expr"],
            inputs=d.get("inputs", []),
            target_dtype=DType(d.get("target_dtype", "float")),
        )


@dataclass
class Program:
    """An executable program in the DSL."""
    program_id: str
    slots: List[Slot] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    source: str = ""

    @property
    def slot_names(self) -> Set[str]:
        return {s.name for s in self.slots}

    @property
    def slot_map(self) -> Dict[str, Slot]:
        return {s.name: s for s in self.slots}

    def fingerprint(self) -> str:
        sig = "|".join(f"{s.op.value}:{s.expr}" for s in self.steps)
        return hashlib.md5(sig.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "program_id": self.program_id,
            "slots": [{"name": s.name, "dtype": s.dtype.value, "description": s.description} for s in self.slots],
            "steps": [s.to_dict() for s in self.steps],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Program":
        return cls(
            program_id=d["program_id"],
            slots=[Slot(name=s["name"], dtype=DType(s["dtype"]), description=s.get("description", "")) for s in d.get("slots", [])],
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            source=d.get("source", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class Executor:
    """Deterministic executor for DSL programs."""

    SAFE_BUILTINS = {
        # Basic
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "len": len, "int": int, "float": float,
        "bool": bool, "str": str,
        # Arithmetic
        "pow": pow, "divmod": divmod,
        # Math (algebra, precalculus)
        "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor,
        "log": math.log, "log2": math.log2, "log10": math.log10, "exp": math.exp,
        # Trigonometry (precalculus, geometry)
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
        "pi": math.pi, "e": math.e,
        # Combinatorics (counting_and_probability)
        "factorial": math.factorial, "comb": math.comb, "perm": math.perm,
        # Number theory
        "gcd": math.gcd,
        # Constants
        "True": True, "False": False, "inf": math.inf,
    }

    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps

    def execute(self, program: Program, bindings: Dict[str, Any]) -> Tuple[bool, Any, Dict[str, Any]]:
        """Execute a program with given slot bindings.

        Returns (success, output_value, full_env).
        """
        env: Dict[str, Any] = {}
        for slot in program.slots:
            if slot.name not in bindings:
                return False, None, {"error": f"Missing binding for slot '{slot.name}'"}
            try:
                env[slot.name] = DType.coerce(bindings[slot.name], slot.dtype)
            except TypeError as e:
                return False, None, {"error": str(e)}

        output_value = None
        for i, step in enumerate(program.steps):
            if i >= self.max_steps:
                return False, None, {"error": "Max steps exceeded"}

            try:
                safe_env = {**self.SAFE_BUILTINS, **env}
                result = eval(step.expr, {"__builtins__": {}}, safe_env)  # noqa: S307
                result = DType.coerce(result, step.target_dtype)
                env[step.target] = result
                if step.op == Op.OUTPUT:
                    output_value = result
            except Exception as e:
                return False, None, {"error": f"Step {i} ({step.target}): {e}"}

        return True, output_value, env


@dataclass
class Subroutine:
    """A reusable subroutine in the library (opaque ID)."""
    sub_id: str
    program: Program
    support: int = 0
    mdl_gain: float = 0.0

    @property
    def signature(self) -> str:
        inputs = ", ".join(f"{s.name}: {s.dtype.value}" for s in self.program.slots)
        return f"{self.sub_id}({inputs})"


class SubroutineLibrary:
    """Collection of subroutines with opaque IDs."""

    def __init__(self):
        self.subroutines: Dict[str, Subroutine] = {}
        self._fp_index: Dict[str, str] = {}

    def add(self, sub: Subroutine) -> bool:
        fp = sub.program.fingerprint()
        if fp in self._fp_index:
            self.subroutines[self._fp_index[fp]].support += 1
            return False
        self.subroutines[sub.sub_id] = sub
        self._fp_index[fp] = sub.sub_id
        return True

    def get(self, sub_id: str) -> Optional[Subroutine]:
        return self.subroutines.get(sub_id)

    @property
    def size(self) -> int:
        return len(self.subroutines)

    def signatures(self) -> List[str]:
        return [s.signature for s in sorted(self.subroutines.values(), key=lambda x: x.sub_id)]

    def save(self, path: str):
        data = {}
        for sid, sub in self.subroutines.items():
            data[sid] = {
                "sub_id": sub.sub_id,
                "program": sub.program.to_dict(),
                "support": sub.support,
                "mdl_gain": sub.mdl_gain,
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SubroutineLibrary":
        lib = cls()
        with open(path) as f:
            data = json.load(f)
        for sid, d in data.items():
            sub = Subroutine(
                sub_id=d["sub_id"],
                program=Program.from_dict(d["program"]),
                support=d.get("support", 0),
                mdl_gain=d.get("mdl_gain", 0.0),
            )
            lib.subroutines[sid] = sub
            lib._fp_index[sub.program.fingerprint()] = sid
        return lib

    def stats(self) -> dict:
        if not self.subroutines:
            return {"size": 0}
        supports = [s.support for s in self.subroutines.values()]
        steps = [len(s.program.steps) for s in self.subroutines.values()]
        return {
            "size": len(self.subroutines),
            "avg_support": sum(supports) / len(supports),
            "avg_steps": sum(steps) / len(steps),
            "total_support": sum(supports),
        }

    # -- Library Evolution API (SEVAL) ------------------------------------

    def next_id(self) -> str:
        """Generate next available subroutine ID."""
        existing = [int(s.sub_id[1:]) for s in self.subroutines.values()
                    if s.sub_id.startswith("L") and s.sub_id[1:].isdigit()]
        next_num = max(existing, default=-1) + 1
        return f"L{next_num:02d}"

    def mint_subroutine(
        self, program: "Program", support: int = 0, mdl_gain: float = 0.0,
    ) -> Optional[Subroutine]:
        """Create and add a new verified subroutine to the library.

        Returns the subroutine if successfully added (not a duplicate), else None.
        """
        fp = program.fingerprint()
        if fp in self._fp_index:
            self.subroutines[self._fp_index[fp]].support += support
            return None
        sub = Subroutine(
            sub_id=self.next_id(), program=program,
            support=support, mdl_gain=mdl_gain,
        )
        self.subroutines[sub.sub_id] = sub
        self._fp_index[fp] = sub.sub_id
        return sub

    def snapshot(self) -> Dict[str, Any]:
        """Serializable snapshot for evolution logging."""
        return {
            "size": self.size,
            "ids": sorted(self.subroutines.keys()),
            "total_support": sum(s.support for s in self.subroutines.values()),
            "avg_mdl_gain": (
                sum(s.mdl_gain for s in self.subroutines.values()) / self.size
                if self.size > 0 else 0.0
            ),
        }

    def diversity_score(self) -> float:
        """Measure library diversity via fingerprint entropy."""
        if self.size <= 1:
            return 0.0
        fps = [s.program.fingerprint() for s in self.subroutines.values()]
        unique = len(set(fps))
        return unique / len(fps)

    def prune_low_quality(self, min_support: int = 2, min_mdl_gain: float = 0.0) -> int:
        """Remove subroutines below quality thresholds. Returns count removed."""
        to_remove = [
            sid for sid, sub in self.subroutines.items()
            if sub.support < min_support or sub.mdl_gain < min_mdl_gain
        ]
        for sid in to_remove:
            fp = self.subroutines[sid].program.fingerprint()
            del self.subroutines[sid]
            self._fp_index.pop(fp, None)
        return len(to_remove)


@dataclass
class CompositionPlan:
    """A plan that calls subroutines from the library."""
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"plan": self.calls}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "CompositionPlan":
        return cls(calls=d.get("plan", []))

    @classmethod
    def from_json(cls, s: str) -> "CompositionPlan":
        return cls.from_dict(json.loads(s))

    @property
    def num_calls(self) -> int:
        return len(self.calls)

    @property
    def subroutine_ids(self) -> List[str]:
        return [c.get("sub_id", "") for c in self.calls]

    @property
    def subroutine_bigrams(self) -> List[Tuple[str, str]]:
        ids = self.subroutine_ids
        return [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]


class CompositionExecutor:
    """Execute a composition plan against a subroutine library."""

    def __init__(self, library: SubroutineLibrary, max_calls: int = 10):
        self.library = library
        self.executor = Executor()
        self.max_calls = max_calls

    def execute(self, plan: CompositionPlan, initial_bindings: Dict[str, Any]) -> Tuple[bool, Any, Dict]:
        env = dict(initial_bindings)
        output = None
        stats = {"calls_made": 0, "calls_succeeded": 0, "calls_failed": 0}

        for i, call in enumerate(plan.calls):
            if i >= self.max_calls:
                return False, None, {**stats, "error": "Max calls exceeded"}

            sub_id = call.get("sub_id", "")
            bindings = call.get("bindings", {})

            sub = self.library.get(sub_id)
            if sub is None:
                stats["calls_failed"] += 1
                return False, None, {**stats, "error": f"Unknown subroutine '{sub_id}'"}

            call_bindings = {}
            for slot in sub.program.slots:
                if slot.name in bindings:
                    call_bindings[slot.name] = bindings[slot.name]
                elif slot.name in env:
                    call_bindings[slot.name] = env[slot.name]
                else:
                    candidates = [(k, v) for k, v in env.items()
                                  if DType.check(v, slot.dtype) and not k.startswith("__")]
                    if len(candidates) == 1:
                        call_bindings[slot.name] = candidates[0][1]
                    else:
                        stats["calls_failed"] += 1
                        return False, None, {**stats, "error": f"Cannot bind slot '{slot.name}' in '{sub_id}': {len(candidates)} candidates"}

            stats["calls_made"] += 1
            success, result, call_env = self.executor.execute(sub.program, call_bindings)
            if not success:
                stats["calls_failed"] += 1
                return False, None, {**stats, "error": f"Execution failed in '{sub_id}': {call_env.get('error', '')}"}

            stats["calls_succeeded"] += 1
            env.update(call_env)
            if result is not None:
                output = result
                env["__last_output__"] = result

        return True, output, stats


def inline_program(plan: CompositionPlan, library: SubroutineLibrary) -> Optional[Program]:
    """Convert a composition plan to a flat (inlined) program.

    This is used for the flat-program baseline: same DSL, no library calls.
    Each subroutine's internal variables are prefixed with ``_s{counter}_``
    to avoid collisions, and all references in ``expr`` and ``inputs`` are
    rewritten accordingly.  Slot names (the external interface) are **not**
    renamed.  Intermediate ``OUTPUT`` steps are demoted to ``COMPUTE`` so
    that only the final subroutine's output is emitted.
    """
    all_slots: Dict[str, Slot] = {}
    all_steps: List[Step] = []
    step_counter = 0

    num_calls = len(plan.calls)
    for call_idx, call in enumerate(plan.calls):
        sub_id = call.get("sub_id", "")
        sub = library.get(sub_id)
        if sub is None:
            return None

        # Collect slots (shared across calls, not renamed)
        slot_names = {slot.name for slot in sub.program.slots}
        for slot in sub.program.slots:
            if slot.name not in all_slots:
                all_slots[slot.name] = slot

        # Build rename map for this subroutine's internal step targets.
        # Slot names are NOT renamed — they are the external interface.
        rename_map: Dict[str, str] = {}
        for i, step in enumerate(sub.program.steps):
            if step.target not in slot_names:
                rename_map[step.target] = f"_s{step_counter + i}_{step.target}"

        for i, step in enumerate(sub.program.steps):
            new_target = rename_map.get(step.target, step.target)

            # Rename inputs
            new_inputs = [rename_map.get(inp, inp) for inp in step.inputs]

            # Rename variable references in expr
            new_expr = step.expr
            for old_name, new_name in rename_map.items():
                new_expr = re.sub(
                    r'\b' + re.escape(old_name) + r'\b', new_name, new_expr
                )

            # Convert intermediate OUTPUT to COMPUTE
            new_op = step.op
            if step.op == Op.OUTPUT and call_idx < num_calls - 1:
                new_op = Op.COMPUTE

            inlined = Step(
                op=new_op,
                target=new_target,
                expr=new_expr,
                inputs=new_inputs,
                target_dtype=step.target_dtype,
            )
            all_steps.append(inlined)
            step_counter += 1

    return Program(
        program_id="flat_inlined",
        slots=list(all_slots.values()),
        steps=all_steps,
    )
