#!/usr/bin/env python3
"""Extract executable JSON-AST programs from GSM8K/MATH via teacher model.

Pipeline:
1. Load GSM8K/MATH train data
2. Teacher (Qwen3.5-27B) generates K candidate JSON-AST programs per problem
3. Filter: JSON valid + types valid + executor produces correct answer
4. Select shortest valid program
5. Abstract valid programs -> typed subroutines (values -> slots)
6. Cluster by normalized structure -> build SubroutineLibrary
7. Mine library by MDL gain -> select top-L subroutines
8. Remap all training programs to use library calls -> composition plans
9. Save: library, composition plans, flat programs, training data
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a math problem compiler. Convert this problem and solution into an executable JSON program.

Problem: {problem}
Solution: {solution}
Answer: {answer}

Output a JSON object:
{{
  "program_id": "p_{idx}",
  "slots": [{{"name": "var_name", "dtype": "int"|"float"|"string", "description": "what this represents"}}],
  "steps": [
    {{"op": "assign"|"compute"|"compare"|"aggregate"|"condition"|"output",
      "target": "result_var_name",
      "expr": "Python expression using slot names and prior targets",
      "inputs": ["list_of_used_vars"],
      "target_dtype": "int"|"float"}}
  ]
}}

Rules:
- Replace ALL specific numbers with slot variables
- Each step expr must be a valid Python expression
- The last step must have op="output" and produce the final answer
- Use only: +, -, *, /, //, %, **, abs, round, min, max, sum, len, int, float, sqrt, ceil, floor
- Return ONLY valid JSON"""


def load_datasets(config: dict, allow_synthetic: bool = False) -> dict:
    all_data = {}
    for ds_key in ["gsm8k", "math"]:
        ds_cfg = config["datasets"][ds_key]
        logger.info("Loading %s...", ds_key)
        try:
            subsets = ds_cfg.get("subsets")
            subset = ds_cfg.get("subset")
            if subsets:
                parts = [load_dataset(ds_cfg["dataset_id"], s, split=ds_cfg["train_split"]) for s in subsets]
                ds = concatenate_datasets(parts)
            elif subset:
                ds = load_dataset(ds_cfg["dataset_id"], subset, split=ds_cfg["train_split"])
            else:
                ds = load_dataset(ds_cfg["dataset_id"], split=ds_cfg["train_split"])
            max_s = ds_cfg.get("max_train", 5000)
            if len(ds) > max_s:
                ds = ds.shuffle(seed=42).select(range(max_s))

            items = []
            for ex in ds:
                problem = ex.get("question", ex.get("problem", ""))
                solution = ex.get("answer", ex.get("solution", ""))
                answer = ""
                if "####" in str(solution):
                    answer = str(solution).split("####")[-1].strip()
                elif ds_key == "math":
                    boxed = re.findall(r"\\boxed\{([^}]*)\}", str(solution))
                    answer = boxed[-1].strip() if boxed else ""
                elif solution:
                    nums = re.findall(r"[\-\d,]+\.?\d*", str(solution))
                    answer = nums[-1].replace(",", "") if nums else ""
                if problem and solution:
                    items.append({"problem": problem, "solution": solution, "answer": answer, "source": ds_key})
            all_data[ds_key] = items
            logger.info("  Loaded %d problems from %s", len(items), ds_key)
        except Exception as e:
            if not allow_synthetic:
                raise RuntimeError(
                    f"Failed to load {ds_key}: {e}. Pass --allow_synthetic to use synthetic fallback."
                ) from e
            logger.warning("Failed to load %s: %s — generating synthetic fallback", ds_key, e)
            all_data[ds_key] = _synthetic_fallback(ds_key, 500)
    return all_data


def _synthetic_fallback(source: str, n: int) -> list:
    logger.warning(
        "SYNTHETIC DATA: Generating %d synthetic examples for '%s'. "
        "These are NOT real math problems — results will be meaningless for research evaluation. "
        "Pass real data or remove --allow_synthetic for production runs.",
        n, source,
    )
    items = []
    for i in range(n):
        a, b = (i * 7 + 3) % 100 + 1, (i * 11 + 5) % 100 + 1
        items.append({
            "problem": f"A store has {a} apples and buys {b} more. How many total?",
            "solution": f"{a} + {b} = {a + b}",
            "answer": str(a + b),
            "source": source,
            "is_synthetic": True,
        })
    return items


def _answer_matches(exec_result, gold_answer: str, tol: float = 1e-3) -> bool:
    """Check if execution result matches the gold answer within tolerance."""
    if gold_answer is None or gold_answer.strip() == "":
        return False
    try:
        exec_val = float(str(exec_result).replace(",", ""))
        gold_val = float(str(gold_answer).replace(",", ""))
        if gold_val == 0:
            return abs(exec_val) < tol
        return abs(exec_val - gold_val) / max(abs(gold_val), 1e-12) < tol
    except (ValueError, TypeError):
        return str(exec_result).strip() == str(gold_answer).strip()


def _flush_logs():
    """Force flush all log handlers and stdout/stderr."""
    for handler in logging.root.handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()


def generate_programs(data: list, model, tokenizer, config: dict, source: str,
                      output_dir: str = None) -> list:
    """Generate K candidate programs per problem, keep shortest answer-correct one."""
    max_new_tokens = config["teacher"].get("max_new_tokens", 512)
    k_samples = config["teacher"].get("samples_per_example", 1)
    temperature = config["teacher"].get("temperature", 0.7)
    top_p = config["teacher"].get("top_p", 0.95)
    checkpoint_interval = config["teacher"].get("checkpoint_interval", 500)
    executor = Executor()
    results = []
    parse_ok, exec_ok, correct_ok, total = 0, 0, 0, 0

    checkpoint_path = os.path.join(output_dir, f"_checkpoint_{source}.json") if output_dir else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        results = ckpt["results"]
        start_idx = ckpt["next_idx"]
        parse_ok, exec_ok, correct_ok = ckpt["parse_ok"], ckpt["exec_ok"], ckpt["correct_ok"]
        total = start_idx
        logger.info("  Resuming from checkpoint: idx=%d, correct=%d", start_idx, correct_ok)
        _flush_logs()
    else:
        start_idx = 0

    for i, item in enumerate(data):
        if i < start_idx:
            continue
        total += 1
        prompt = EXTRACTION_PROMPT.format(
            problem=item["problem"][:500],
            solution=item["solution"][:500],
            answer=item["answer"][:50],
            idx=i,
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        candidates = []
        for k in range(k_samples):
            with torch.no_grad():
                if k == 0:
                    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                else:
                    output = model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        do_sample=True, temperature=temperature, top_p=top_p,
                    )
            response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            program = _parse_program(response, f"{source}_p{i}_k{k}")
            if program is not None:
                candidates.append(program)
                if k == 0 and program is not None:
                    break

        if not candidates:
            continue
        parse_ok += 1

        correct_candidates = []
        any_exec = False
        for prog in candidates:
            bindings = _extract_bindings(item, prog)
            success, result, env = executor.execute(prog, bindings)
            if success and result is not None:
                any_exec = True
                if _answer_matches(result, item["answer"]):
                    correct_candidates.append((prog, bindings, result))
        if any_exec:
            exec_ok += 1

        if not correct_candidates:
            continue
        correct_ok += 1

        best_prog, best_bindings, best_result = min(
            correct_candidates, key=lambda x: len(x[0].steps)
        )
        best_prog.program_id = f"{source}_p{i}"
        results.append({
            "problem": item["problem"],
            "answer": item["answer"],
            "source": source,
            "program": best_prog.to_dict(),
            "bindings": best_bindings,
            "exec_result": best_result,
        })

        if (i + 1) % 50 == 0:
            logger.info(
                "  [%s] %d/%d: parsed=%d, exec=%d, correct=%d (%.1f%%)",
                source, i + 1, len(data), parse_ok, exec_ok, correct_ok,
                100 * correct_ok / total if total > 0 else 0,
            )
            _flush_logs()

        if checkpoint_path and (i + 1) % checkpoint_interval == 0:
            ckpt_data = {
                "results": results, "next_idx": i + 1,
                "parse_ok": parse_ok, "exec_ok": exec_ok, "correct_ok": correct_ok,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(ckpt_data, f)
            logger.info("  Checkpoint saved: %d results at idx=%d", len(results), i + 1)
            _flush_logs()

    logger.info(
        "[%s] Done: %d/%d parsed, %d/%d executable, %d/%d answer-correct",
        source, parse_ok, total, exec_ok, total, correct_ok, total,
    )
    _flush_logs()
    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    return results


def _parse_program(response: str, prog_id: str) -> Program | None:
    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return None
        data = json.loads(json_match.group())
        data["program_id"] = prog_id
        return Program.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _extract_bindings(item: dict, program: Program) -> dict:
    """Extract slot bindings from problem text."""
    bindings = {}
    numbers = re.findall(r'[\d,]+\.?\d*', item["problem"])
    numbers = [n.replace(",", "") for n in numbers]

    slot_list = list(program.slots)
    for j, slot in enumerate(slot_list):
        if j < len(numbers):
            try:
                cleaned = numbers[j].replace(",", "").strip()
                if not cleaned:
                    bindings[slot.name] = 0
                else:
                    bindings[slot.name] = DType.coerce(cleaned, slot.dtype)
            except (TypeError, ValueError):
                try:
                    cleaned = numbers[j].replace(",", "").strip()
                    bindings[slot.name] = float(cleaned) if cleaned and "." in cleaned else int(cleaned) if cleaned else 0
                except (ValueError, TypeError):
                    bindings[slot.name] = 0
        else:
            bindings[slot.name] = 0
    return bindings


def _step_signature(step) -> str:
    """Build expression-level signature for a step."""
    ops_in_expr = set()
    for op_str in ['+', '-', '*', '/', '**', 'min(', 'max(', 'abs(', 'round(', 'sum(', 'len(', 'sqrt(', 'ceil(', 'floor(']:
        if op_str in step.expr:
            ops_in_expr.add(op_str.rstrip('('))
    return f"{step.op.value}:{'|'.join(sorted(ops_in_expr))}"


def _program_signature(steps) -> tuple:
    """Build expression-level signature for a sequence of steps."""
    return tuple(_step_signature(s) for s in steps)


def build_subroutine_library(programs: list, config: dict) -> SubroutineLibrary:
    """Mine subroutines from programs by structural clustering."""
    lib = SubroutineLibrary()
    lib_cfg = config["library"]
    target_size = lib_cfg.get("main_size", 16)

    min_support_gsm8k = lib_cfg.get("min_support_gsm8k", 3)
    min_support_math = lib_cfg.get("min_support_math", min_support_gsm8k)

    # Group programs by expression-level signature + slot count
    fp_groups = defaultdict(list)
    source_groups = defaultdict(lambda: defaultdict(int))
    for item in programs:
        prog = Program.from_dict(item["program"])
        steps_sig = _program_signature(prog.steps)
        n_slots = len(prog.slots)
        key = (steps_sig, n_slots)
        fp_groups[key].append(prog)
        source_groups[key][item.get("source", "gsm8k")] += 1

    sorted_groups = sorted(fp_groups.items(), key=lambda x: len(x[1]), reverse=True)

    sub_counter = 0
    for key, progs in sorted_groups:
        if lib.size >= target_size:
            break
        src_counts = source_groups[key]
        primary_source = max(src_counts, key=src_counts.get) if src_counts else "gsm8k"
        min_support = min_support_math if primary_source == "math" else min_support_gsm8k
        if len(progs) < min_support:
            continue

        representative = progs[0]
        sub_id = f"L{sub_counter:02d}"
        sub = Subroutine(
            sub_id=sub_id,
            program=representative,
            support=len(progs),
            mdl_gain=len(progs) * len(representative.steps),
        )
        if lib.add(sub):
            sub_counter += 1

    # Also mine common subsequences for multi-call plans
    if lib.size < target_size:
        existing_sigs = {_program_signature(s.program.steps) for s in lib.subroutines.values()}
        for subseq_len in [2, 3]:
            subseq_groups = defaultdict(list)
            for item in programs:
                prog = Program.from_dict(item["program"])
                if len(prog.steps) <= subseq_len:
                    continue
                for start in range(len(prog.steps) - subseq_len + 1):
                    subseq = prog.steps[start:start + subseq_len]
                    if subseq[-1].op == Op.OUTPUT:
                        continue
                    sig = _program_signature(subseq)
                    if sig in existing_sigs:
                        continue
                    used_inputs = set()
                    targets_so_far = set()
                    for s in subseq:
                        used_inputs.update(v for v in s.inputs if v not in targets_so_far)
                        targets_so_far.add(s.target)
                    n_inputs = len(used_inputs)
                    subseq_groups[(sig, n_inputs)].append((prog, start))

            for (sig, n_inputs), instances in sorted(
                subseq_groups.items(), key=lambda x: len(x[1]), reverse=True
            ):
                if lib.size >= target_size:
                    break
                if len(instances) < min_support_gsm8k:
                    continue
                prog, offset = instances[0]
                subseq = prog.steps[offset:offset + subseq_len]
                used_inputs = set()
                targets_so_far = set()
                for s in subseq:
                    used_inputs.update(v for v in s.inputs if v not in targets_so_far)
                    targets_so_far.add(s.target)
                sub_slots = [slot for slot in prog.slots if slot.name in used_inputs]
                sub_prog = Program(f"subseq_{sub_counter}", sub_slots, list(subseq))
                sub_id = f"L{sub_counter:02d}"
                sub = Subroutine(
                    sub_id=sub_id,
                    program=sub_prog,
                    support=len(instances),
                    mdl_gain=len(instances) * subseq_len,
                )
                if lib.add(sub):
                    sub_counter += 1
                    existing_sigs.add(sig)

    logger.info("Built subroutine library: %d subroutines (full-program + subsequence mining)", lib.size)
    return lib


def _find_matching_bindings(subroutine, values, expected_answer, executor):
    """Try permutations of values to find the binding that produces the correct answer."""
    from itertools import permutations as _perms
    from math import factorial

    sub_slots = subroutine.program.slots
    n_slots = len(sub_slots)
    if not values or n_slots == 0:
        return None
    if len(values) < n_slots:
        return None
    n_perms = factorial(len(values)) // factorial(len(values) - n_slots)
    if n_perms > 2000:
        bindings = {slot.name: values[j] for j, slot in enumerate(sub_slots) if j < len(values)}
        success, result, _ = executor.execute(subroutine.program, bindings)
        if success and result is not None and _answer_matches(str(result), str(expected_answer)):
            return bindings
        return None
    for perm in _perms(values, n_slots):
        bindings = {slot.name: val for slot, val in zip(sub_slots, perm)}
        success, result, _ = executor.execute(subroutine.program, bindings)
        if success and result is not None and _answer_matches(str(result), str(expected_answer)):
            return bindings
    return None


def build_composition_plans(programs: list, library: SubroutineLibrary) -> list:
    """Decompose each program into multi-call composition plans via greedy covering."""
    executor = Executor()
    plans = []
    call_counts = []
    binding_found, binding_missed = 0, 0
    for item in programs:
        prog = Program.from_dict(item["program"])
        steps = prog.steps
        calls = []
        i = 0
        while i < len(steps):
            best_sub, best_len = None, 0
            for sub in library.subroutines.values():
                sub_steps = sub.program.steps
                sub_len = len(sub_steps)
                if i + sub_len <= len(steps):
                    prog_ops = tuple(s.op.value for s in steps[i:i + sub_len])
                    sub_ops = tuple(s.op.value for s in sub_steps)
                    if prog_ops == sub_ops and sub_len > best_len:
                        best_sub = sub
                        best_len = sub_len
            if best_sub is not None:
                prog_bindings = item.get("bindings", {})
                values = list(prog_bindings.values())
                call_bindings = _find_matching_bindings(
                    best_sub, values, item.get("answer", item.get("exec_result", "")), executor
                )
                if call_bindings is not None:
                    binding_found += 1
                else:
                    binding_missed += 1
                    call_bindings = {
                        slot.name: values[j] if j < len(values) else 0
                        for j, slot in enumerate(best_sub.program.slots)
                    }
                calls.append({"sub_id": best_sub.sub_id, "bindings": call_bindings})
                i += best_len
            else:
                i += 1

        if not calls:
            first_sub = list(library.subroutines.values())[0]
            prog_bindings = item.get("bindings", {})
            values = list(prog_bindings.values())
            call_bindings = _find_matching_bindings(
                first_sub, values, item.get("answer", item.get("exec_result", "")), executor
            )
            if call_bindings is None:
                call_bindings = {
                    slot.name: values[j] if j < len(values) else 0
                    for j, slot in enumerate(first_sub.program.slots)
                }
            calls = [{"sub_id": first_sub.sub_id, "bindings": call_bindings}]

        plan = CompositionPlan(calls=calls)
        call_counts.append(len(calls))
        plans.append({**item, "plan_data": plan.to_dict(), "flat_program": prog.to_dict()})

    if call_counts:
        logger.info(
            "Composition plan stats: mean=%.2f, median=%.1f, max=%d calls",
            sum(call_counts) / len(call_counts),
            sorted(call_counts)[len(call_counts) // 2],
            max(call_counts),
        )
    logger.info(
        "Built %d composition plans (bindings: %d found, %d missed)",
        len(plans), binding_found, binding_missed,
    )
    return plans


def build_training_data(plans: list, library: SubroutineLibrary, output_dir: str):
    """Create planner training data (problem -> composition plan JSON)."""
    lib_sigs = "\n".join(library.signatures())

    compose_data = []
    flat_data = []
    inline_failures = 0

    for item in plans:
        problem = item["problem"]
        plan_json = json.dumps(item["plan_data"])

        compose_data.append({
            "instruction": f"Available subroutines:\n{lib_sigs}\n\nProblem: {problem}\n\nGenerate a composition plan (JSON):",
            "output": plan_json,
            "source": item.get("source", ""),
        })

        # Derive flat program by inlining the composition plan for fair comparison
        plan_obj = CompositionPlan.from_dict(item["plan_data"])
        inlined = inline_program(plan_obj, library)
        if inlined is not None:
            flat_program_dict = inlined.to_dict()
        else:
            inline_failures += 1
            flat_program_dict = item["flat_program"]
        flat_program_json = json.dumps(flat_program_dict)
        flat_data.append({
            "instruction": f"Problem: {problem}\n\nGenerate an executable program (JSON):",
            "output": flat_program_json,
            "source": item.get("source", ""),
        })

    if inline_failures > 0:
        logger.warning("Inline failures (fell back to raw program): %d/%d", inline_failures, len(plans))

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "compose_train.json"), "w") as f:
        json.dump(compose_data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "flat_train.json"), "w") as f:
        json.dump(flat_data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "plans_with_programs.json"), "w") as f:
        json.dump(plans, f, indent=2, ensure_ascii=False)

    logger.info("Training data saved: compose=%d, flat=%d", len(compose_data), len(flat_data))


def generate_synthetic_programs(data: list, source: str) -> list:
    """Generate synthetic executable programs (no LLM needed)."""
    executor = Executor()
    results = []
    for i, item in enumerate(data):
        raw_numbers = re.findall(r'\d[\d,]*\.?\d*', item["problem"])
        numbers = []
        for n in raw_numbers:
            cleaned = n.replace(",", "").strip()
            if cleaned and cleaned != ".":
                try:
                    float(cleaned)
                    numbers.append(cleaned)
                except ValueError:
                    continue
        if len(numbers) < 2:
            numbers = [str((i * 7 + 3) % 100 + 1), str((i * 11 + 5) % 50 + 1)]

        a_val = float(numbers[0])
        b_val = float(numbers[1])

        slots = [Slot("a", DType.FLOAT, "first number"), Slot("b", DType.FLOAT, "second number")]
        steps = [
            Step(Op.COMPUTE, "result", "a * b", ["a", "b"], DType.FLOAT),
            Step(Op.OUTPUT, "__output__", "result", ["result"], DType.FLOAT),
        ]
        prog = Program(f"{source}_p{i}", slots, steps, source=source)
        bindings = {"a": a_val, "b": b_val}

        success, result, env = executor.execute(prog, bindings)
        if success:
            results.append({
                "problem": item["problem"],
                "answer": item["answer"],
                "source": source,
                "program": prog.to_dict(),
                "bindings": bindings,
                "exec_result": result,
            })

    logger.info("[%s] Synthetic: %d programs generated", source, len(results))
    return results


def run_post_split(args, config):
    """Post-split mode: rebuild library and training data using ONLY train partition."""
    programs_path = args.programs_path or os.path.join(args.output_dir, "all_programs.json")
    logger.info("=" * 60)
    logger.info("  Post-split mode: rebuild from train partition only")
    logger.info("=" * 60)

    logger.info("Loading programs from %s", programs_path)
    with open(programs_path) as f:
        all_programs = json.load(f)
    logger.info("Loaded %d total programs", len(all_programs))

    logger.info("Loading MCD split from %s", args.split_path)
    with open(args.split_path) as f:
        mcd_split = json.load(f)

    # Get train partition indices/IDs
    train_ids = set()
    if "train" in mcd_split:
        train_partition = mcd_split["train"]
        if isinstance(train_partition, list):
            if train_partition and isinstance(train_partition[0], int):
                train_ids = set(train_partition)
            elif train_partition and isinstance(train_partition[0], str):
                train_ids = set(train_partition)
            elif train_partition and isinstance(train_partition[0], dict):
                # List of dicts with an index or id field
                for entry in train_partition:
                    if "index" in entry:
                        train_ids.add(entry["index"])
                    elif "id" in entry:
                        train_ids.add(entry["id"])

    # Filter to train-only programs
    if train_ids:
        train_programs = []
        for idx, prog in enumerate(all_programs):
            prog_id = prog.get("program", {}).get("program_id", "")
            if idx in train_ids or prog_id in train_ids:
                train_programs.append(prog)
        # If ID-based matching found nothing, try index-based
        if not train_programs and all(isinstance(x, int) for x in train_ids):
            train_programs = [all_programs[i] for i in sorted(train_ids) if i < len(all_programs)]
    else:
        logger.warning("Could not parse train partition from MCD split; using all programs")
        train_programs = all_programs

    logger.info("Train partition: %d programs (out of %d total)", len(train_programs), len(all_programs))

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Rebuild Subroutine Library (train-only)")
    logger.info("=" * 60)
    library = build_subroutine_library(train_programs, config)
    library.save(os.path.join(args.output_dir, "subroutine_library.json"))
    logger.info("Library stats: %s", json.dumps(library.stats()))

    logger.info("=" * 60)
    logger.info("  Rebuild Composition Plans (train-only)")
    logger.info("=" * 60)
    plans = build_composition_plans(train_programs, library)

    # --- Diagnostic: measure plan faithfulness (does NOT filter) ---
    logger.info("Measuring plan faithfulness (diagnostic, not filtering)...")
    comp_exec = CompositionExecutor(library)
    n_faithful = 0
    for item in plans:
        plan_obj = CompositionPlan.from_dict(item["plan_data"])
        bindings = item.get("bindings", {})
        success, result, _stats = comp_exec.execute(plan_obj, bindings)
        gold = item.get("exec_result", item.get("answer"))
        if success and result is not None and _answer_matches(result, str(gold)):
            n_faithful += 1
    pct = (n_faithful / len(plans) * 100) if plans else 0.0
    logger.info("Plan faithfulness (diagnostic): %d/%d (%.1f%%)", n_faithful, len(plans), pct)
    # NOTE: We keep ALL structurally-matched plans as training data.
    # The planner learns problem->plan mapping; faithfulness measures
    # how well the subroutine's fixed expressions generalize.

    logger.info("=" * 60)
    logger.info("  Rebuild Training Data (train-only)")
    logger.info("=" * 60)
    build_training_data(plans, library, args.output_dir)

    logger.info("=" * 60)
    logger.info("  Post-split rebuild complete")
    logger.info("  Train programs: %d", len(train_programs))
    logger.info("  Library: %d subroutines", library.size)
    logger.info("  Plans: %d", len(plans))
    logger.info("=" * 60)

    meta = {
        "total_programs": len(all_programs),
        "train_programs": len(train_programs),
        "library_size": library.size,
        "plans": len(plans),
        "post_split": True,
        "split_path": args.split_path,
    }
    with open(os.path.join(args.output_dir, "extraction_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Extraction metadata saved to %s/extraction_meta.json", args.output_dir)


def main():
    parser = argparse.ArgumentParser(description="Extract executable programs from CoT traces")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "template_config.yaml"))
    parser.add_argument("--output_dir", type=str, default="results/templates")
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument("--use_student", action="store_true", help="Use student model (9B) instead of teacher (32B)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic programs (for smoke testing)")
    parser.add_argument("--allow_synthetic", action="store_true", help="Allow synthetic fallback when real data unavailable")
    parser.add_argument("--post_split", action="store_true", help="Post-split mode: rebuild library/plans/training data from train partition only")
    parser.add_argument("--split_path", type=str, default=None, help="Path to MCD split JSON (required for --post_split)")
    parser.add_argument("--programs_path", type=str, default=None, help="Path to all_programs.json (defaults to output_dir/all_programs.json)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Post-split mode: rebuild library and training data from train partition only
    if args.post_split:
        if not args.split_path:
            parser.error("--split_path is required when using --post_split")
        run_post_split(args, config)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Stage 1: Extract Executable Programs")
    logger.info("=" * 60)

    source_data = load_datasets(config, allow_synthetic=args.allow_synthetic)
    if args.max_per_source:
        for k in source_data:
            source_data[k] = source_data[k][:args.max_per_source]

    all_programs = []

    if args.synthetic:
        logger.info("Using synthetic program generation (no LLM)")
        for source_name, items in source_data.items():
            programs = generate_synthetic_programs(items, source_name)
            all_programs.extend(programs)
    else:
        model_name = config["planner"]["model"] if args.use_student else config["teacher"]["model"]
        logger.info("Loading model: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        model.eval()

        for source_name, items in source_data.items():
            logger.info("Processing %s (%d problems)...", source_name, len(items))
            programs = generate_programs(items, model, tokenizer, config, source_name,
                                          output_dir=args.output_dir)
            all_programs.extend(programs)

        del model
        torch.cuda.empty_cache()

    logger.info("Total executable programs: %d", len(all_programs))
    with open(os.path.join(args.output_dir, "all_programs.json"), "w") as f:
        json.dump(all_programs, f, indent=2, ensure_ascii=False)

    # NOTE: Library, composition plans, and training data are NO LONGER built here.
    # They must be built AFTER the MCD split to prevent test data leakage.
    # Use --post_split mode after running build_mcd_split.py.
    logger.info("=" * 60)
    logger.info("  Program extraction complete")
    logger.info("  Programs: %d", len(all_programs))
    logger.info("  Next: run build_mcd_split.py, then re-run with --post_split")
    logger.info("=" * 60)

    meta = {
        "total_programs": len(all_programs),
        "synthetic_used": args.synthetic or args.allow_synthetic,
        "synthetic_flag": "--synthetic" in sys.argv or "--allow_synthetic" in sys.argv,
        "post_split_required": True,
    }
    if args.synthetic:
        meta["warning"] = "ALL programs are synthetic — not valid for research evaluation"
    with open(os.path.join(args.output_dir, "extraction_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Extraction metadata saved to %s/extraction_meta.json", args.output_dir)


if __name__ == "__main__":
    main()
