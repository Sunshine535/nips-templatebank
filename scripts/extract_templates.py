#!/usr/bin/env python3
"""Extract structured reasoning templates from GSM8K/MATH CoT traces.

Pipeline:
1. Generate CoT traces from Qwen/Qwen3.5-9B on GSM8K train (7473) + MATH train (7500)
2. Parse each trace into TemplateSteps: identify operations, variables, dependencies
3. Abstract into ReasoningTemplates (replace concrete numbers with variables)
4. Cluster similar templates → create TemplateBank
5. Target: 50-100 GSM8K templates, 200-300 MATH templates
6. Save bank to results/template_bank.json
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.template_algebra import (
    ReasoningTemplate,
    TemplateAlgebra,
    TemplateBank,
    TemplateStep,
    Variable,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)



EXTRACTION_PROMPT = """Analyze the following math problem and its solution. Extract a reusable reasoning TEMPLATE.

Problem: {problem}
Solution: {solution}

Output a JSON object with:
- "name": short template name (e.g., "proportion_comparison", "multi_step_arithmetic")
- "domain": one of ["arithmetic", "algebra", "geometry", "word_problem", "combinatorics", "number_theory", "probability"]
- "variables": list of {{"name": "var_name", "type": "number"|"string"|"expression"}}
- "steps": list of {{"operation": "compute"|"assign"|"compare"|"branch"|"output", "expression": "step with {{var}} placeholders", "inputs": ["list_of_input_vars"], "output_var": "result_var"}}

Make the template GENERIC: replace ALL specific numbers/names with {{variable}} placeholders.
Return ONLY valid JSON, no explanation."""


COT_GENERATION_PROMPT = """Solve the following problem step by step. Show your reasoning clearly.

Problem: {problem}

Let's think step by step:"""


def load_source_datasets(config: dict) -> dict:
    """Load GSM8K and MATH training data, keyed by source name."""
    all_data = {}
    for ds_cfg in config["extraction"]["source_datasets"]:
        name = ds_cfg["name"]
        logger.info("Loading dataset: %s", name)
        try:
            subset = ds_cfg.get("subset")
            if subset:
                ds = load_dataset(ds_cfg["dataset_id"], subset, split=ds_cfg["split"])
            else:
                ds = load_dataset(ds_cfg["dataset_id"], split=ds_cfg["split"])
            max_s = ds_cfg.get("max_samples", 5000)
            if len(ds) > max_s:
                ds = ds.shuffle(seed=42).select(range(max_s))

            items = []
            for ex in ds:
                problem = ex.get("question", ex.get("problem", ""))
                solution = ex.get("answer", ex.get("solution", ""))
                if problem and solution:
                    items.append({"problem": problem, "solution": solution, "source": name})
            all_data[name] = items
            logger.info("  Loaded %d problems from %s", len(items), name)
        except Exception as e:
            logger.warning("Failed to load %s: %s — generating synthetic fallback", name, e)
            all_data[name] = _generate_synthetic(name, min(ds_cfg.get("max_samples", 1000), 1000))

    return all_data


def _generate_synthetic(source: str, n: int) -> list:
    items = []
    for i in range(n):
        a, b = (i * 7 + 3) % 100 + 1, (i * 11 + 5) % 100 + 1
        if source == "gsm8k":
            items.append({
                "problem": f"A store has {a} apples and receives {b} more. How many apples total?",
                "solution": f"Start with {a}. Add {b}. Total = {a} + {b} = {a + b}. #### {a + b}",
                "source": source,
            })
        else:
            items.append({
                "problem": f"Find the value of {a}x + {b} when x = 3.",
                "solution": f"{a} * 3 + {b} = {a * 3} + {b} = {a * 3 + b}",
                "source": source,
            })
    return items


def generate_cot_traces(data: list, model, tokenizer, max_new_tokens: int = 512) -> list:
    """Generate CoT traces for problems that don't have full solutions."""
    enhanced = []
    for i, item in enumerate(data):
        if len(item["solution"]) > 50:
            enhanced.append(item)
            continue

        prompt = COT_GENERATION_PROMPT.format(problem=item["problem"][:500])
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        enhanced.append({**item, "solution": response, "cot_generated": True})

        if (i + 1) % 100 == 0:
            logger.info("  CoT generation: %d/%d", i + 1, len(data))

    return enhanced


def extract_templates_with_llm(data: list, model, tokenizer, config: dict, source_name: str) -> TemplateBank:
    """Use LLM to extract templates from CoT traces."""
    max_templates = config["extraction"]["max_templates_per_dataset"]
    max_new_tokens = config["extraction"]["max_new_tokens"]

    bank = TemplateBank()
    extracted = 0
    parse_failures = 0

    for i, item in enumerate(data):
        if extracted >= max_templates:
            break

        prompt = EXTRACTION_PROMPT.format(
            problem=item["problem"][:500],
            solution=item["solution"][:500],
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        template = parse_template_response(response, item, f"{source_name}_t{i}")
        if template:
            added = bank.add(template)
            if added:
                extracted += 1
        else:
            parse_failures += 1

        if (i + 1) % 50 == 0:
            logger.info("  [%s] Processed %d/%d, extracted %d templates (%d parse failures)",
                        source_name, i + 1, len(data), extracted, parse_failures)

    logger.info("Extraction from %s complete: %d templates from %d problems (%d parse failures)",
                source_name, extracted, len(data), parse_failures)
    return bank


def parse_template_response(response: str, source_item: dict, template_id: str):
    """Parse LLM response into a ReasoningTemplate."""
    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return None
        data = json.loads(json_match.group())

        variables = [
            Variable(name=v["name"], var_type=v.get("type", "any"))
            for v in data.get("variables", [])
        ]
        steps = [
            TemplateStep(
                step_id=idx,
                operation=s.get("operation", "compute"),
                expression=s.get("expression", ""),
                inputs=s.get("inputs", []),
                output_var=s.get("output_var"),
            )
            for idx, s in enumerate(data.get("steps", []))
        ]

        if not steps:
            return None

        return ReasoningTemplate(
            template_id=template_id,
            name=data.get("name", f"template_{template_id}"),
            domain=data.get("domain", "math"),
            variables=variables,
            steps=steps,
            source_problems=[source_item["problem"][:200]],
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.debug("Failed to parse template: %s", e)
        return None


def cluster_and_deduplicate(bank: TemplateBank, algebra: TemplateAlgebra) -> TemplateBank:
    """Cluster similar templates and merge them to reduce redundancy."""
    templates = list(bank.templates.values())
    if len(templates) < 5:
        return bank

    by_domain = defaultdict(list)
    for t in templates:
        by_domain[t.domain].append(t)

    clean_bank = TemplateBank()
    for domain, domain_templates in by_domain.items():
        op_groups = defaultdict(list)
        for t in domain_templates:
            op_sig = tuple(s.operation for s in t.steps)
            op_groups[op_sig].append(t)

        for sig, group in op_groups.items():
            if len(group) == 1:
                clean_bank.add(group[0])
            else:
                merged = algebra.merge(group[:5], name=f"merged_{domain}_{len(sig)}step")
                merged.reuse_count = sum(t.reuse_count for t in group)
                clean_bank.add(merged)

    logger.info("Clustering: %d → %d templates", len(templates), len(clean_bank.templates))
    return clean_bank


def create_compiler_training_data(bank: TemplateBank, source_data: dict, output_dir: str) -> list:
    """Create two-stage SFT data for template compiler."""
    all_templates = bank.search(min_reuse=0)
    if not all_templates:
        logger.warning("No templates in bank, cannot create training data")
        return []

    by_domain = defaultdict(list)
    for t in all_templates:
        by_domain[t.domain].append(t)

    selection_data = []
    filling_data = []

    for source_name, items in source_data.items():
        for item in items:
            best_template = None
            for t in all_templates:
                if any(kw in item["problem"].lower() for kw in [t.domain, t.name.split("_")[0]]):
                    best_template = t
                    break
            if not best_template:
                best_template = all_templates[hash(item["problem"]) % len(all_templates)]

            # Stage 1: Template selection
            selection_data.append({
                "instruction": f"Select the best reasoning template for this problem:\n\n{item['problem']}",
                "output": f"Template: {best_template.name}\nDomain: {best_template.domain}\n"
                         f"Steps: {best_template.num_steps}\n{best_template.to_prompt()}",
                "template_id": best_template.template_id,
                "stage": "selection",
            })

            # Stage 2: Variable filling
            filling_data.append({
                "instruction": (
                    f"Given the following template, fill in the variables and solve:\n\n"
                    f"Template: {best_template.to_prompt()}\n\nProblem: {item['problem']}"
                ),
                "output": item["solution"],
                "template_id": best_template.template_id,
                "stage": "filling",
            })

    selection_path = os.path.join(output_dir, "compiler_stage1_selection.json")
    with open(selection_path, "w") as f:
        json.dump(selection_data, f, indent=2, ensure_ascii=False)

    filling_path = os.path.join(output_dir, "compiler_stage2_filling.json")
    with open(filling_path, "w") as f:
        json.dump(filling_data, f, indent=2, ensure_ascii=False)

    combined = selection_data + filling_data
    combined_path = os.path.join(output_dir, "compiler_training_data.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    logger.info("Created compiler training data:")
    logger.info("  Stage 1 (selection): %d examples → %s", len(selection_data), selection_path)
    logger.info("  Stage 2 (filling):   %d examples → %s", len(filling_data), filling_path)
    logger.info("  Combined:            %d examples → %s", len(combined), combined_path)

    return combined


def main():
    parser = argparse.ArgumentParser(description="Extract reasoning templates from CoT traces")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "template_config.yaml"))
    parser.add_argument("--output_dir", type=str, default="results/templates")
    parser.add_argument("--skip_cot_generation", action="store_true", help="Use existing solutions as-is")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load source datasets
    logger.info("=" * 50)
    logger.info("  STEP 1: Load Source Datasets")
    logger.info("=" * 50)
    source_data = load_source_datasets(config)
    total_problems = sum(len(v) for v in source_data.values())
    logger.info("Total problems: %d", total_problems)

    # Step 2: Load model
    model_name = config["extraction"]["teacher_model"]
    logger.info("=" * 50)
    logger.info("  STEP 2: Load Model — %s", model_name)
    logger.info("=" * 50)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )
    model.eval()

    # Step 3: Generate CoT traces where needed
    if not args.skip_cot_generation:
        logger.info("=" * 50)
        logger.info("  STEP 3: Generate CoT Traces")
        logger.info("=" * 50)
        for name in source_data:
            logger.info("  Generating CoT for %s (%d problems)...", name, len(source_data[name]))
            source_data[name] = generate_cot_traces(source_data[name], model, tokenizer)

    # Step 4: Extract templates
    logger.info("=" * 50)
    logger.info("  STEP 4: Extract Templates")
    logger.info("=" * 50)
    combined_bank = TemplateBank()
    for source_name, items in source_data.items():
        logger.info("Extracting from %s (%d problems)...", source_name, len(items))
        bank = extract_templates_with_llm(items, model, tokenizer, config, source_name)
        for tid, t in bank.templates.items():
            combined_bank.add(t)
        logger.info("  %s: %d templates extracted", source_name, len(bank.templates))

    # Step 5: Cluster and deduplicate
    logger.info("=" * 50)
    logger.info("  STEP 5: Cluster and Deduplicate")
    logger.info("=" * 50)
    algebra = TemplateAlgebra()
    combined_bank = cluster_and_deduplicate(combined_bank, algebra)

    bank_path = os.path.join(args.output_dir, "template_bank.json")
    combined_bank.save(bank_path)
    logger.info("Saved template bank: %s", bank_path)

    stats = combined_bank.stats()
    logger.info("Template bank stats: %s", json.dumps(stats, indent=2))
    with open(os.path.join(args.output_dir, "bank_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Step 6: Create compiler training data
    logger.info("=" * 50)
    logger.info("  STEP 6: Create Compiler Training Data")
    logger.info("=" * 50)
    create_compiler_training_data(combined_bank, source_data, args.output_dir)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("  Template extraction complete")
    logger.info("  Bank: %s (%d templates)", bank_path, len(combined_bank.templates))
    logger.info("  Stats: %s", json.dumps(stats))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
