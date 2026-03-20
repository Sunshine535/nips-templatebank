#!/usr/bin/env python3
"""Extract structured reasoning templates from GSM8K/MATH CoT traces using Qwen3.5-27B."""

import argparse
import json
import logging
import os
import re
import sys

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.template_algebra import ReasoningTemplate, TemplateBank, TemplateStep, Variable


EXTRACTION_PROMPT = """Analyze the following math problem and its solution. Extract a reusable reasoning TEMPLATE.

Problem: {problem}
Solution: {solution}

Output a JSON object with:
- "name": short template name
- "domain": "math" or "algebra" or "geometry" or "word_problem"
- "variables": list of {{"name": "var_name", "type": "number"|"string"|"expression"}}
- "steps": list of {{"operation": "compute"|"assign"|"compare"|"output", "expression": "step with {{var}} placeholders", "output_var": "result_var"}}

Make the template GENERIC: replace specific numbers/names with {{variable}} placeholders.
Return ONLY valid JSON, no explanation."""


def load_source_datasets(config: dict):
    """Load GSM8K and MATH training data."""
    all_data = []
    for ds_cfg in config["extraction"]["source_datasets"]:
        name = ds_cfg["name"]
        logger.info("Loading dataset: %s", name)
        try:
            subset = ds_cfg.get("subset")
            if subset:
                ds = load_dataset(ds_cfg["dataset_id"], subset, split=ds_cfg["split"], trust_remote_code=True)
            else:
                ds = load_dataset(ds_cfg["dataset_id"], split=ds_cfg["split"], trust_remote_code=True)
            max_s = ds_cfg.get("max_samples", 5000)
            if len(ds) > max_s:
                ds = ds.shuffle(seed=42).select(range(max_s))
            for ex in ds:
                problem = ex.get("question", ex.get("problem", ""))
                solution = ex.get("answer", ex.get("solution", ""))
                if problem and solution:
                    all_data.append({"problem": problem, "solution": solution, "source": name})
            logger.info("  Loaded %d examples from %s", len(ds), name)
        except Exception as e:
            logger.warning("Failed to load %s: %s", name, e)

    if not all_data:
        logger.info("Generating synthetic math problems...")
        for i in range(1000):
            a, b = (i * 7 + 3) % 100, (i * 11 + 5) % 100
            all_data.append({
                "problem": f"If a store has {a} apples and receives {b} more, how many apples total?",
                "solution": f"The store starts with {a} apples. They receive {b} more. Total = {a} + {b} = {a + b}. The answer is {a + b}.",
                "source": "synthetic",
            })
    return all_data


def extract_templates_with_llm(data: list, config: dict) -> TemplateBank:
    """Use teacher LLM to extract templates from CoT traces."""
    model_name = config["extraction"]["teacher_model"]
    batch_size = config["extraction"]["batch_size"]
    max_templates = config["extraction"].get("max_templates_per_dataset", 500)

    logger.info("Loading teacher model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bank = TemplateBank()
    extracted = 0

    for i, item in enumerate(data):
        if extracted >= max_templates:
            break

        prompt = EXTRACTION_PROMPT.format(problem=item["problem"][:500], solution=item["solution"][:500])
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=config["extraction"]["max_new_tokens"],
                do_sample=False,
                temperature=1.0,
            )
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        template = parse_template_response(response, item, f"t_{i}")
        if template:
            added = bank.add(template)
            if added:
                extracted += 1

        if (i + 1) % 50 == 0:
            logger.info("Processed %d/%d, extracted %d templates", i + 1, len(data), extracted)

    del model
    torch.cuda.empty_cache()
    logger.info("Extraction complete: %d templates from %d problems", extracted, len(data))
    return bank


def parse_template_response(response: str, source_item: dict, template_id: str) -> ReasoningTemplate:
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


def create_compiler_training_data(bank: TemplateBank, source_data: list, output_path: str):
    """Create SFT data for template compiler: problem → template → instantiation."""
    training_data = []
    templates = bank.search(min_reuse=0)

    for item in source_data:
        best_template = None
        for t in templates:
            if t.domain in item.get("source", ""):
                best_template = t
                break
        if not best_template and templates:
            best_template = templates[0]
        if not best_template:
            continue

        instruction = (
            f"Given the following problem, identify the appropriate reasoning template "
            f"and solve step by step.\n\nProblem: {item['problem']}"
        )
        response = f"Template: {best_template.name}\n\n"
        response += best_template.to_prompt() + "\n\n"
        response += f"Solution:\n{item['solution']}"

        training_data.append({
            "instruction": instruction,
            "output": response,
            "template_id": best_template.template_id,
        })

    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    logger.info("Created %d compiler training examples → %s", len(training_data), output_path)
    return training_data


def main():
    parser = argparse.ArgumentParser(description="Extract reasoning templates from CoT traces")
    parser.add_argument("--config", type=str, default="configs/template_config.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/templates")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=== Step 1: Load source datasets ===")
    data = load_source_datasets(config)
    logger.info("Total problems: %d", len(data))

    logger.info("=== Step 2: Extract templates with LLM ===")
    bank = extract_templates_with_llm(data, config)

    bank_path = os.path.join(args.output_dir, "template_bank.json")
    bank.save(bank_path)
    logger.info("Saved template bank: %s", bank_path)

    stats = bank.stats()
    logger.info("Template bank stats: %s", json.dumps(stats, indent=2))
    with open(os.path.join(args.output_dir, "bank_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=== Step 3: Create compiler training data ===")
    compiler_data_path = os.path.join(args.output_dir, "compiler_training_data.json")
    create_compiler_training_data(bank, data, compiler_data_path)

    logger.info("=== Template extraction complete ===")


if __name__ == "__main__":
    main()
