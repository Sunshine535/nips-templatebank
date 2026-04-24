"""GIFT: Grounded Interface-Flow Template Composition.

Core dataflow plan representation and executor. Every subroutine input
must have an explicit BindingRef — no implicit env/name/type fallback.

Plans are typed DAGs:
- Nodes are subroutine calls
- Edges are explicit binding refs (quantity or call_output)
- Final answer is a selected call output
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from src.template_dsl import DType, Executor, SubroutineLibrary


@dataclass
class BindingRef:
    """Explicit reference for a subroutine slot binding.

    source="quantity": value is a number from the problem text
    source="call_output": value comes from a previous call's output
    source="constant": a literal constant (e.g., 0, 1, pi)
    """
    source: Literal["quantity", "call_output", "constant"]
    value: Optional[Any] = None
    call_id: Optional[str] = None
    dtype: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"source": self.source}
        if self.source == "quantity":
            d["value"] = self.value
        elif self.source == "call_output":
            d["call_id"] = self.call_id
        elif self.source == "constant":
            d["value"] = self.value
        if self.dtype is not None:
            d["dtype"] = self.dtype
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "BindingRef":
        return cls(
            source=data["source"],
            value=data.get("value"),
            call_id=data.get("call_id"),
            dtype=data.get("dtype"),
        )


@dataclass
class PlanCall:
    """A single subroutine call with explicit bindings."""
    call_id: str
    sub_id: str
    bindings: Dict[str, BindingRef] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "sub_id": self.sub_id,
            "bindings": {k: v.to_dict() for k, v in self.bindings.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlanCall":
        bindings = {}
        for k, v in data.get("bindings", {}).items():
            if isinstance(v, dict):
                bindings[k] = BindingRef.from_dict(v)
            else:
                bindings[k] = BindingRef(source="quantity", value=v)
        return cls(
            call_id=data["call_id"],
            sub_id=data["sub_id"],
            bindings=bindings,
        )


@dataclass
class DataflowPlan:
    """A typed DAG of subroutine calls with explicit bindings."""
    calls: List[PlanCall] = field(default_factory=list)
    final: Optional[BindingRef] = None

    def to_dict(self) -> dict:
        d = {"calls": [c.to_dict() for c in self.calls]}
        if self.final is not None:
            d["final"] = self.final.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "DataflowPlan":
        calls = [PlanCall.from_dict(c) for c in data.get("calls", [])]
        final = None
        if "final" in data and data["final"] is not None:
            final = BindingRef.from_dict(data["final"])
        return cls(calls=calls, final=final)

    def has_true_dataflow(self) -> bool:
        """Check if any call uses a call_output ref (true multi-call flow)."""
        for call in self.calls:
            for ref in call.bindings.values():
                if ref.source == "call_output":
                    return True
        return False

    def flow_edges(self) -> list:
        """Extract (source_call_id, target_call_id) edges from call_output refs."""
        edges = []
        for call in self.calls:
            for ref in call.bindings.values():
                if ref.source == "call_output" and ref.call_id is not None:
                    edges.append((ref.call_id, call.call_id))
        return edges


class DataflowExecutor:
    """Execute a DataflowPlan using explicit bindings only.

    NEVER falls back to implicit env/name/type matching.
    Missing bindings cause immediate failure.
    """

    def __init__(self, library: SubroutineLibrary):
        self.library = library
        self.executor = Executor()

    def execute(self, plan: DataflowPlan) -> tuple:
        """Execute a DataflowPlan.

        Returns: (success, result, stats)
        """
        stats = {
            "calls_made": 0,
            "calls_succeeded": 0,
            "calls_failed": 0,
            "bindings_resolved": 0,
            "bindings_missing": 0,
        }
        call_outputs = {}

        for call in plan.calls:
            sub = self.library.get(call.sub_id)
            if sub is None:
                stats["calls_failed"] += 1
                stats["error"] = f"Unknown subroutine '{call.sub_id}'"
                return False, None, stats

            required_slots = {slot.name for slot in sub.program.slots}
            provided_slots = set(call.bindings.keys())
            missing = required_slots - provided_slots
            if missing:
                stats["calls_failed"] += 1
                stats["bindings_missing"] += len(missing)
                stats["error"] = (
                    f"Call '{call.call_id}' to '{call.sub_id}': "
                    f"missing bindings for {sorted(missing)}"
                )
                return False, None, stats

            call_bindings = {}
            for slot_name, ref in call.bindings.items():
                if slot_name not in required_slots:
                    continue
                resolved = self._resolve_ref(ref, call_outputs)
                if resolved is None and ref.source == "call_output":
                    stats["calls_failed"] += 1
                    stats["error"] = (
                        f"Call '{call.call_id}': cannot resolve "
                        f"call_output ref to '{ref.call_id}'"
                    )
                    return False, None, stats
                call_bindings[slot_name] = resolved
                stats["bindings_resolved"] += 1

            stats["calls_made"] += 1
            success, result, _ = self.executor.execute(sub.program, call_bindings)
            if not success or result is None:
                stats["calls_failed"] += 1
                stats["error"] = f"Execution failed in '{call.sub_id}'"
                return False, None, stats

            call_outputs[call.call_id] = result
            stats["calls_succeeded"] += 1

        if plan.final is not None:
            final_result = self._resolve_ref(plan.final, call_outputs)
        elif call_outputs:
            final_result = list(call_outputs.values())[-1]
        else:
            final_result = None

        return True, final_result, stats

    def _resolve_ref(self, ref: BindingRef, call_outputs: dict) -> Any:
        """Resolve a BindingRef to a concrete value."""
        if ref.source == "quantity":
            return ref.value
        elif ref.source == "constant":
            return ref.value
        elif ref.source == "call_output":
            if ref.call_id is None or ref.call_id not in call_outputs:
                return None
            return call_outputs[ref.call_id]
        return None
