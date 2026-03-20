"""Template Algebra: formal operations on reasoning templates.
Templates are programs with variable slots extracted from CoT traces."""

import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Variable:
    """A named variable slot in a template."""
    name: str
    var_type: str = "any"  # any, number, string, expression
    constraints: List[str] = field(default_factory=list)

    def matches(self, value: Any) -> bool:
        if self.var_type == "number":
            try:
                float(str(value))
                return True
            except (ValueError, TypeError):
                return False
        return True


@dataclass
class TemplateStep:
    """A single step in a template program."""
    step_id: int
    operation: str  # compute, assign, compare, branch, output
    expression: str  # the step expression with variable placeholders
    inputs: List[str] = field(default_factory=list)
    output_var: Optional[str] = None

    def instantiate(self, bindings: Dict[str, Any]) -> str:
        result = self.expression
        for var_name, value in bindings.items():
            result = result.replace(f"{{{var_name}}}", str(value))
        return result


@dataclass
class ReasoningTemplate:
    """A structured reasoning template: program with variable slots."""
    template_id: str
    name: str
    domain: str  # math, logic, etc.
    variables: List[Variable] = field(default_factory=list)
    steps: List[TemplateStep] = field(default_factory=list)
    source_problems: List[str] = field(default_factory=list)
    reuse_count: int = 0

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def variable_names(self) -> Set[str]:
        return {v.name for v in self.variables}

    def instantiate(self, bindings: Dict[str, Any]) -> List[str]:
        """Instantiate template with concrete values."""
        result = []
        for step in self.steps:
            result.append(step.instantiate(bindings))
        return result

    def to_prompt(self) -> str:
        """Convert template to a readable prompt format."""
        lines = [f"Template: {self.name}", f"Variables: {', '.join(v.name for v in self.variables)}", "Steps:"]
        for step in self.steps:
            lines.append(f"  {step.step_id}. [{step.operation}] {step.expression}")
        return "\n".join(lines)

    def fingerprint(self) -> str:
        """Hash for deduplication."""
        ops = "|".join(s.operation + ":" + s.expression for s in self.steps)
        return hashlib.md5(ops.encode()).hexdigest()[:12]


class TemplateAlgebra:
    """Algebraic operations on reasoning templates."""

    @staticmethod
    def compose(t1: ReasoningTemplate, t2: ReasoningTemplate, name: str = "composed") -> ReasoningTemplate:
        """Compose T1 ; T2: pipe output of T1 into T2.
        T1's output variables become T2's input variables where names match."""
        t1_outputs = {s.output_var for s in t1.steps if s.output_var}
        t2_inputs = {inp for s in t2.steps for inp in s.inputs}
        shared = t1_outputs & t2_inputs

        new_vars = list(t1.variables)
        for v in t2.variables:
            if v.name not in {nv.name for nv in new_vars} and v.name not in shared:
                new_vars.append(copy.deepcopy(v))

        new_steps = []
        for i, s in enumerate(t1.steps):
            new_step = copy.deepcopy(s)
            new_step.step_id = i
            new_steps.append(new_step)

        offset = len(t1.steps)
        for i, s in enumerate(t2.steps):
            new_step = copy.deepcopy(s)
            new_step.step_id = offset + i
            new_steps.append(new_step)

        return ReasoningTemplate(
            template_id=f"compose_{t1.template_id}_{t2.template_id}",
            name=name,
            domain=t1.domain,
            variables=new_vars,
            steps=new_steps,
            source_problems=t1.source_problems + t2.source_problems,
        )

    @staticmethod
    def abstract(template: ReasoningTemplate, threshold: float = 0.8) -> ReasoningTemplate:
        """Abstract a template: replace concrete values with new variables.
        Values that appear in >threshold fraction of source_problems become variables."""
        abstracted = copy.deepcopy(template)
        concrete_pattern = re.compile(r'\b(\d+(?:\.\d+)?)\b')
        var_counter = len(abstracted.variables)

        for step in abstracted.steps:
            matches = concrete_pattern.findall(step.expression)
            for match in matches:
                var_name = f"v{var_counter}"
                step.expression = step.expression.replace(match, f"{{{var_name}}}", 1)
                abstracted.variables.append(Variable(name=var_name, var_type="number"))
                var_counter += 1

        abstracted.template_id = f"abstract_{template.template_id}"
        abstracted.name = f"Abstract({template.name})"
        return abstracted

    @staticmethod
    def specialize(template: ReasoningTemplate, partial_bindings: Dict[str, Any]) -> ReasoningTemplate:
        """Specialize a template: fix some variables to concrete values."""
        specialized = copy.deepcopy(template)

        specialized.variables = [v for v in specialized.variables if v.name not in partial_bindings]

        for step in specialized.steps:
            step.expression = step.instantiate(partial_bindings)

        specialized.template_id = f"spec_{template.template_id}"
        specialized.name = f"Specialize({template.name})"
        return specialized

    @staticmethod
    def merge(templates: List[ReasoningTemplate], name: str = "merged") -> ReasoningTemplate:
        """Merge multiple templates: find common structure, keep union of unique steps."""
        if not templates:
            return ReasoningTemplate(template_id="empty", name=name, domain="")
        if len(templates) == 1:
            return copy.deepcopy(templates[0])

        all_vars = {}
        for t in templates:
            for v in t.variables:
                if v.name not in all_vars:
                    all_vars[v.name] = v

        step_signatures = {}
        for t in templates:
            for s in t.steps:
                sig = f"{s.operation}:{s.expression}"
                if sig not in step_signatures:
                    step_signatures[sig] = copy.deepcopy(s)

        merged_steps = list(step_signatures.values())
        for i, s in enumerate(merged_steps):
            s.step_id = i

        return ReasoningTemplate(
            template_id=f"merge_{'_'.join(t.template_id for t in templates[:3])}",
            name=name,
            domain=templates[0].domain,
            variables=list(all_vars.values()),
            steps=merged_steps,
            source_problems=sum((t.source_problems for t in templates), []),
        )

    @staticmethod
    def decompose(template: ReasoningTemplate) -> List[ReasoningTemplate]:
        """Decompose a template into atomic sub-templates (one step each)."""
        subs = []
        for step in template.steps:
            used_vars = []
            for v in template.variables:
                if f"{{{v.name}}}" in step.expression:
                    used_vars.append(copy.deepcopy(v))
            sub = ReasoningTemplate(
                template_id=f"atom_{template.template_id}_{step.step_id}",
                name=f"Step {step.step_id} of {template.name}",
                domain=template.domain,
                variables=used_vars,
                steps=[copy.deepcopy(step)],
            )
            subs.append(sub)
        return subs


class TemplateBank:
    """Collection of templates with retrieval and matching capabilities."""

    def __init__(self):
        self.templates: Dict[str, ReasoningTemplate] = {}
        self._fingerprint_index: Dict[str, str] = {}

    def add(self, template: ReasoningTemplate) -> bool:
        """Add template if not duplicate. Returns True if added."""
        fp = template.fingerprint()
        if fp in self._fingerprint_index:
            existing = self.templates[self._fingerprint_index[fp]]
            existing.reuse_count += 1
            existing.source_problems.extend(template.source_problems)
            return False
        self.templates[template.template_id] = template
        self._fingerprint_index[fp] = template.template_id
        return True

    def get(self, template_id: str) -> Optional[ReasoningTemplate]:
        return self.templates.get(template_id)

    def search(self, domain: Optional[str] = None, min_reuse: int = 0) -> List[ReasoningTemplate]:
        results = list(self.templates.values())
        if domain:
            results = [t for t in results if t.domain == domain]
        results = [t for t in results if t.reuse_count >= min_reuse]
        return sorted(results, key=lambda t: t.reuse_count, reverse=True)

    def stats(self) -> Dict[str, Any]:
        domains = {}
        for t in self.templates.values():
            domains[t.domain] = domains.get(t.domain, 0) + 1
        return {
            "total_templates": len(self.templates),
            "domains": domains,
            "avg_reuse": sum(t.reuse_count for t in self.templates.values()) / max(len(self.templates), 1),
            "avg_steps": sum(t.num_steps for t in self.templates.values()) / max(len(self.templates), 1),
        }

    def save(self, path: str):
        data = {}
        for tid, t in self.templates.items():
            data[tid] = {
                "template_id": t.template_id,
                "name": t.name,
                "domain": t.domain,
                "variables": [{"name": v.name, "type": v.var_type} for v in t.variables],
                "steps": [
                    {"step_id": s.step_id, "operation": s.operation, "expression": s.expression,
                     "inputs": s.inputs, "output_var": s.output_var}
                    for s in t.steps
                ],
                "reuse_count": t.reuse_count,
                "num_source_problems": len(t.source_problems),
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TemplateBank":
        bank = cls()
        with open(path) as f:
            data = json.load(f)
        for tid, tdata in data.items():
            t = ReasoningTemplate(
                template_id=tdata["template_id"],
                name=tdata["name"],
                domain=tdata["domain"],
                variables=[Variable(name=v["name"], var_type=v["type"]) for v in tdata["variables"]],
                steps=[
                    TemplateStep(
                        step_id=s["step_id"], operation=s["operation"], expression=s["expression"],
                        inputs=s.get("inputs", []), output_var=s.get("output_var"),
                    )
                    for s in tdata["steps"]
                ],
                reuse_count=tdata.get("reuse_count", 0),
            )
            bank.templates[tid] = t
            bank._fingerprint_index[t.fingerprint()] = tid
        return bank
