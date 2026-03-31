#!/usr/bin/env python3
"""Extract executable JSON-AST programs from GSM8K/MATH via teacher model.

Pipeline:
1. Load GSM8K/MATH train data
2. Teacher (Qwen3.5-32B) generates K candidate JSON-AST programs per problem
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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.template_dsl import (
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
            subset = ds_cfg.get("subset")
            if subset:
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


def generate_programs(data: list, model, tokenizer, config: dict, source: str) -> list:
    """Generate executable JSON-AST programs from problems."""
    max_new_tokens = config["teacher"]["max_new_tokens"]
    executor = Executor()
    results = []
    parse_ok, exec_ok, total = 0, 0, 0

    for i, item in enumerate(data):
        total += 1
        prompt = EXTRACTION_PROMPT.format(
            problem=item["problem"][:500],
            solution=item["solution"][:500],
            answer=item["answer"][:50],
            idx=i,
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        program = _parse_program(response, f"{source}_p{i}")
        if program is None:
            continue
        parse_ok += 1

        bindings = _extract_bindings(item, program)
        success, result, env = executor.execute(program, bindings)
        if success and result is not None:
            exec_ok += 1
            results.append({
                "problem": item["problem"],
                "answer": item["answer"],
                "source": source,
                "program": program.to_dict(),
                "bindings": bindings,
                "exec_result": result,
            })

        if (i + 1) % 100 == 0:
            logger.info("  [%s] %d/%d: parsed=%d, executable=%d", source, i + 1, len(data), parse_ok, exec_ok)

    logger.info("[%s] Done: %d/%d parsed, %d/%d executable", source, parse_ok, total, exec_ok, total)
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
                bindings[slot.name] = DType.coerce(numbers[j], slot.dtype)
            except TypeError:
                bindings[slot.name] = float(numbers[j]) if "." in numbers[j] else int(numbers[j])
        else:
            bindings[slot.name] = 0
    return bindings


def build_subroutine_library(programs: list, config: dict) -> SubroutineLibrary:
    """Mine subroutines from programs by structural clustering."""
    lib = SubroutineLibrary()
    lib_cfg = config["library"]
    target_size = lib_cfg["main_size"]

    fp_groups = defaultdict(list)
    for item in programs:
        prog = Program.from_dict(item["program"])
        steps_sig = tuple(s.op.value for s in prog.steps)
        fp_groups[steps_sig].append(prog)

    sorted_groups = sorted(fp_groups.items(), key=lambda x: len(x[1]), reverse=True)

    sub_counter = 0
    for sig, progs in sorted_groups:
        if lib.size >= target_size:
            break
        if len(progs) < lib_cfg.get("min_support_gsm8k", 5):
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

    logger.info("Built subroutine library: %d subroutines", lib.size)
    return lib


def build_composition_plans(programs: list, library: SubroutineLibrary) -> list:
    """Map each program to a composition plan using library subroutines."""
    plans = []
    for item in programs:
        prog = Program.from_dict(item["program"])
        steps_sig = tuple(s.op.value for s in prog.steps)

        best_sub = None
        for sub in library.subroutines.values():
            sub_sig = tuple(s.op.value for s in sub.program.steps)
            if sub_sig == steps_sig:
                best_sub = sub
                break

        if best_sub is None:
            for sub in library.subroutines.values():
                sub_sig = tuple(s.op.value for s in sub.program.steps)
                if len(sub_sig) <= len(steps_sig):
                    best_sub = sub
                    break

        if best_sub is None:
            best_sub = list(library.subroutines.values())[0]

        plan = CompositionPlan(calls=[{
            "sub_id": best_sub.sub_id,
            "bindings": item.get("bindings", {}),
        }])

        plans.append({
            **item,
            "plan_data": plan.to_dict(),
            "flat_program": prog.to_dict(),
        })

    logger.info("Built %d composition plans", len(plans))
    return plans


def build_training_data(plans: list, library: SubroutineLibrary, output_dir: str):
    """Create planner training data (problem -> composition plan JSON)."""
    lib_sigs = "\n".join(library.signatures())

    compose_data = []
    flat_data = []

    for item in plans:
        problem = item["problem"]
        plan_json = json.dumps(item["plan_data"])

        compose_data.append({
            "instruction": f"Available subroutines:\n{lib_sigs}\n\nProblem: {problem}\n\nGenerate a composition plan (JSON):",
            "output": plan_json,
            "source": item.get("source", ""),
        })

        flat_program_json = json.dumps(item["flat_program"])
        flat_data.append({
            "instruction": f"Problem: {problem}\n\nGenerate an executable program (JSON):",
            "output": flat_program_json,
            "source": item.get("source", ""),
        })

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


def main():
    parser = argparse.ArgumentParser(description="Extract executable programs from CoT traces")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "template_config.yaml"))
    parser.add_argument("--output_dir", type=str, default="results/templates")
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument("--use_student", action="store_true", help="Use student model (9B) instead of teacher (32B)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic programs (for smoke testing)")
    parser.add_argument("--allow_synthetic", action="store_true", help="Allow synthetic fallback when real data unavailable")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
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
            programs = generate_programs(items, model, tokenizer, config, source_name)
            all_programs.extend(programs)

        del model
        torch.cuda.empty_cache()

    logger.info("Total executable programs: %d", len(all_programs))
    with open(os.path.join(args.output_dir, "all_programs.json"), "w") as f:
        json.dump(all_programs, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("  Stage 2: Build Subroutine Library")
    logger.info("=" * 60)
    library = build_subroutine_library(all_programs, config)
    library.save(os.path.join(args.output_dir, "subroutine_library.json"))
    logger.info("Library stats: %s", json.dumps(library.stats()))

    logger.info("=" * 60)
    logger.info("  Stage 3: Build Composition Plans")
    logger.info("=" * 60)
    plans = build_composition_plans(all_programs, library)

    logger.info("=" * 60)
    logger.info("  Stage 4: Build Training Data")
    logger.info("=" * 60)
    build_training_data(plans, library, args.output_dir)

    logger.info("=" * 60)
    logger.info("  Template extraction complete")
    logger.info("  Programs: %d", len(all_programs))
    logger.info("  Library: %d subroutines", library.size)
    logger.info("  Plans: %d", len(plans))
    logger.info("=" * 60)

    synthetic_count = sum(1 for p in all_programs if p.get("source", "").startswith("synth") or any(
        item.get("is_synthetic") for item in source_data.get(p.get("source", ""), []) if isinstance(item, dict)
    ))
    meta = {
        "total_programs": len(all_programs),
        "library_size": library.size,
        "plans": len(plans),
        "synthetic_used": args.synthetic or args.allow_synthetic,
        "synthetic_flag": "--synthetic" in sys.argv or "--allow_synthetic" in sys.argv,
    }
    if args.synthetic:
        meta["warning"] = "ALL programs are synthetic — not valid for research evaluation"
    with open(os.path.join(args.output_dir, "extraction_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Extraction metadata saved to %s/extraction_meta.json", args.output_dir)


if __name__ == "__main__":
    main()
