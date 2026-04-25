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

    source="quantity": grounded quantity from problem text
        - qid: stable quantity ID (e.g., "q3") — preferred
        - value: numeric value (must match qid if both present)
        - span: original text span (e.g., "12 boxes")
        - entity: optional entity label
        - role: optional semantic role
    source="call_output": value comes from a previous call's output
        - call_id: source call's call_id
    source="constant": a literal constant (e.g., 0, 1, pi)
        - value: the constant
    """
    source: Literal["quantity", "call_output", "constant"]
    value: Optional[Any] = None
    call_id: Optional[str] = None
    dtype: Optional[str] = None
    qid: Optional[str] = None
    span: Optional[str] = None
    entity: Optional[str] = None
    role: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"source": self.source}
        if self.source == "quantity":
            if self.qid is not None:
                d["qid"] = self.qid
            if self.value is not None:
                d["value"] = self.value
            if self.span is not None:
                d["span"] = self.span
            if self.entity is not None:
                d["entity"] = self.entity
            if self.role is not None:
                d["role"] = self.role
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
            qid=data.get("qid"),
            span=data.get("span"),
            entity=data.get("entity"),
            role=data.get("role"),
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

    def execute_with_quantities(self, plan: DataflowPlan, quantities: dict) -> tuple:
        """Execute using quantity lookup by qid (preferred over raw value).

        quantities: {qid: value} mapping from the problem.
        """
        stats = {
            "calls_made": 0, "calls_succeeded": 0, "calls_failed": 0,
            "bindings_resolved": 0, "bindings_missing": 0,
        }
        call_outputs = {}
        for call in plan.calls:
            sub = self.library.get(call.sub_id)
            if sub is None:
                stats["calls_failed"] += 1
                return False, None, {**stats, "error": f"Unknown sub '{call.sub_id}'"}
            required_slots = {slot.name for slot in sub.program.slots}
            missing = required_slots - set(call.bindings.keys())
            if missing:
                stats["calls_failed"] += 1
                stats["bindings_missing"] += len(missing)
                return False, None, {**stats, "error": f"Missing bindings: {sorted(missing)}"}
            call_bindings = {}
            for slot_name, ref in call.bindings.items():
                if slot_name not in required_slots:
                    continue
                val = self._resolve_ref_with_qid(ref, call_outputs, quantities)
                if val is None:
                    stats["calls_failed"] += 1
                    return False, None, {**stats, "error": f"Cannot resolve {slot_name}"}
                call_bindings[slot_name] = val
                stats["bindings_resolved"] += 1
            stats["calls_made"] += 1
            success, result, _ = self.executor.execute(sub.program, call_bindings)
            if not success or result is None:
                stats["calls_failed"] += 1
                return False, None, {**stats, "error": f"Exec failed in {call.sub_id}"}
            call_outputs[call.call_id] = result
            stats["calls_succeeded"] += 1
        if plan.final is not None:
            final = self._resolve_ref_with_qid(plan.final, call_outputs, quantities)
        elif call_outputs:
            final = list(call_outputs.values())[-1]
        else:
            final = None
        return True, final, stats

    def _resolve_ref(self, ref: BindingRef, call_outputs: dict) -> Any:
        """Resolve a BindingRef to a concrete value (no quantity lookup)."""
        if ref.source == "quantity":
            return ref.value
        elif ref.source == "constant":
            return ref.value
        elif ref.source == "call_output":
            if ref.call_id is None or ref.call_id not in call_outputs:
                return None
            return call_outputs[ref.call_id]
        return None

    def _resolve_ref_with_qid(self, ref: BindingRef, call_outputs: dict, quantities: dict) -> Any:
        """Resolve a BindingRef, preferring qid lookup for quantities."""
        if ref.source == "quantity":
            if ref.qid is not None and ref.qid in quantities:
                return quantities[ref.qid]
            return ref.value
        elif ref.source == "constant":
            return ref.value
        elif ref.source == "call_output":
            if ref.call_id is None or ref.call_id not in call_outputs:
                return None
            return call_outputs[ref.call_id]
        return None


# ============================================================
# V-GIFT: Value-Grounded Interface-Flow Template Composition
# ============================================================

@dataclass
class ValueHint:
    """Model-generated intermediate value prediction for a plan call."""
    value: Any = None
    dtype: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> dict:
        d = {}
        if self.value is not None:
            d["value"] = self.value
        if self.dtype is not None:
            d["dtype"] = self.dtype
        if self.confidence is not None:
            d["confidence"] = self.confidence
        return d

    @classmethod
    def from_dict(cls, data) -> "ValueHint":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            return cls(value=data)
        return cls(
            value=data.get("value"),
            dtype=data.get("dtype"),
            confidence=data.get("confidence"),
        )


@dataclass
class ValueAnnotatedPlanCall(PlanCall):
    """PlanCall with an optional model-generated value hint."""
    value_hint: Optional[ValueHint] = None

    def to_dict(self) -> dict:
        d = super().to_dict()
        if self.value_hint is not None:
            d["value_hint"] = self.value_hint.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ValueAnnotatedPlanCall":
        bindings = {}
        for k, v in data.get("bindings", {}).items():
            if isinstance(v, dict):
                bindings[k] = BindingRef.from_dict(v)
            else:
                bindings[k] = BindingRef(source="quantity", value=v)
        vh = None
        if "value_hint" in data and data["value_hint"] is not None:
            vh = ValueHint.from_dict(data["value_hint"])
        return cls(
            call_id=data["call_id"],
            sub_id=data["sub_id"],
            bindings=bindings,
            value_hint=vh,
        )


@dataclass
class ValueAnnotatedDataflowPlan(DataflowPlan):
    """DataflowPlan where each call may carry a value hint."""

    @classmethod
    def from_dict(cls, data: dict) -> "ValueAnnotatedDataflowPlan":
        calls = []
        for c in data.get("calls", []):
            if "value_hint" in c:
                calls.append(ValueAnnotatedPlanCall.from_dict(c))
            else:
                calls.append(PlanCall.from_dict(c))
        final = None
        if "final" in data and data["final"] is not None:
            final = BindingRef.from_dict(data["final"])
        plan = cls(calls=calls, final=final)
        return plan


class ConsistencyExecutor:
    """Execute a ValueAnnotatedDataflowPlan and check value hint consistency.

    Returns both symbolic execution result and value-hint-based result,
    plus per-call consistency diagnostics.
    """

    def __init__(self, library: SubroutineLibrary):
        self.library = library
        self.executor = Executor()

    def execute(self, plan: ValueAnnotatedDataflowPlan,
                quantities: Optional[dict] = None) -> dict:
        """Execute and return comprehensive diagnostics."""
        quantities = quantities or {}
        result = {
            "symbolic_exec_ok": False,
            "symbolic_result": None,
            "value_hint_result": None,
            "per_call": [],
            "consistency_errors": 0,
            "total_calls": len(plan.calls),
            "value_hints_present": 0,
            "value_hints_consistent": 0,
            "final_agreement": False,
        }

        call_outputs = {}
        for call in plan.calls:
            sub = self.library.get(call.sub_id)
            if sub is None:
                result["per_call"].append({
                    "call_id": call.call_id, "error": f"unknown sub {call.sub_id}",
                })
                return result

            required_slots = {s.name for s in sub.program.slots}
            call_bindings = {}
            for slot_name, ref in call.bindings.items():
                if slot_name not in required_slots:
                    continue
                if ref.source == "quantity":
                    val = quantities.get(ref.qid, ref.value) if ref.qid else ref.value
                elif ref.source == "constant":
                    val = ref.value
                elif ref.source == "call_output":
                    val = call_outputs.get(ref.call_id)
                else:
                    val = None
                if val is None:
                    result["per_call"].append({
                        "call_id": call.call_id,
                        "error": f"cannot resolve {slot_name}",
                    })
                    return result
                call_bindings[slot_name] = val

            ok, exec_result, _ = self.executor.execute(sub.program, call_bindings)
            call_info = {
                "call_id": call.call_id,
                "sub_id": call.sub_id,
                "exec_ok": ok,
                "exec_result": exec_result,
            }

            if ok and exec_result is not None:
                call_outputs[call.call_id] = exec_result

            if isinstance(call, ValueAnnotatedPlanCall) and call.value_hint is not None:
                result["value_hints_present"] += 1
                hint_val = call.value_hint.value
                call_info["value_hint"] = hint_val
                if ok and exec_result is not None and hint_val is not None:
                    try:
                        diff = abs(float(exec_result) - float(hint_val))
                        consistent = diff < max(abs(float(exec_result)) * 0.01, 1e-3)
                    except (ValueError, TypeError):
                        consistent = str(exec_result) == str(hint_val)
                    call_info["consistent"] = consistent
                    if consistent:
                        result["value_hints_consistent"] += 1
                    else:
                        result["consistency_errors"] += 1

            result["per_call"].append(call_info)

        if call_outputs:
            last_id = plan.calls[-1].call_id
            result["symbolic_result"] = call_outputs.get(last_id)
            result["symbolic_exec_ok"] = True

        if plan.final is not None:
            if plan.final.source == "call_output" and plan.final.call_id in call_outputs:
                result["symbolic_result"] = call_outputs[plan.final.call_id]

        last_call = plan.calls[-1] if plan.calls else None
        if isinstance(last_call, ValueAnnotatedPlanCall) and last_call.value_hint is not None:
            result["value_hint_result"] = last_call.value_hint.value

        if result["symbolic_result"] is not None and result["value_hint_result"] is not None:
            try:
                s = float(result["symbolic_result"])
                v = float(result["value_hint_result"])
                result["final_agreement"] = abs(s - v) < max(abs(s) * 0.01, 1e-3)
            except (ValueError, TypeError):
                result["final_agreement"] = str(result["symbolic_result"]) == str(result["value_hint_result"])

        return result
