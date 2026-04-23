#!/usr/bin/env python3
"""Fast extraction using vLLM offline batch inference. ~20x faster than HF generate().

Usage:
    python3 scripts/extract_vllm.py \
        --model /root/assets/models/Qwen3.5-27B \
        --output_dir results/templates_pod \
        --tp 4
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.template_dsl import DType, Executor, Op, Program, Slot, Step

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


def load_datasets(config):
    from datasets import load_dataset
    all_data = {}
    for ds_key in ["gsm8k", "math"]:
        ds_cfg = config["datasets"][ds_key]
        logger.info("Loading %s...", ds_key)
        if ds_key == "gsm8k":
            ds = load_dataset(ds_cfg["name"], ds_cfg.get("subset", "main"), split="train")
            items = []
            for row in ds:
                answer = row.get("answer", "")
                if "####" in answer:
                    answer = answer.split("####")[-1].strip()
                items.append({
                    "problem": row["question"],
                    "solution": row.get("answer", ""),
                    "answer": answer,
                })
        else:
            ds = load_dataset(ds_cfg["name"], split="train", trust_remote_code=True)
            items = []
            for row in ds:
                items.append({
                    "problem": row.get("problem", ""),
                    "solution": row.get("solution", ""),
                    "answer": str(row.get("answer", "")),
                })
        logger.info("  Loaded %d problems from %s", len(items), ds_key)
        all_data[ds_key] = items
    return all_data


def build_prompts(data, source, tokenizer=None):
    prompts = []
    for i, item in enumerate(data):
        prompt = EXTRACTION_PROMPT.format(
            problem=item["problem"][:500],
            solution=item["solution"][:500],
            answer=item["answer"][:50],
            idx=i,
        )
        if tokenizer is not None:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            text = prompt
        prompts.append({"text": text, "idx": i, "item": item, "source": source})
    return prompts


def parse_program(response, prog_id):
    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return None
        data = json.loads(json_match.group())
        data["program_id"] = prog_id
        return Program.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def extract_bindings(item, program):
    bindings = {}
    numbers = re.findall(r'[\d,]+\.?\d*', item["problem"])
    numbers = [n.replace(",", "") for n in numbers]
    for j, slot in enumerate(program.slots):
        if j < len(numbers):
            try:
                cleaned = numbers[j].replace(",", "").strip()
                if not cleaned:
                    bindings[slot.name] = 0
                else:
                    bindings[slot.name] = DType.coerce(cleaned, slot.dtype)
            except (TypeError, ValueError):
                try:
                    bindings[slot.name] = float(cleaned) if "." in cleaned else int(float(cleaned))
                except (ValueError, TypeError):
                    bindings[slot.name] = 0
        else:
            bindings[slot.name] = 0
    return bindings


def answer_matches(result, answer):
    try:
        r = float(str(result).replace(",", "").strip())
        a = float(str(answer).replace(",", "").strip())
        if a == 0:
            return abs(r) < 1e-3
        return abs(r - a) / max(abs(a), 1e-8) < 0.01
    except (ValueError, TypeError):
        return str(result).strip().lower() == str(answer).strip().lower()


def run_vllm_batch(prompts, model_path, tp, max_new_tokens, batch_tag=""):
    from vllm import LLM, SamplingParams

    logger.info("Initializing vLLM (tp=%d, model=%s)...", tp, model_path)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )
    params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
    )

    texts = [p["text"] for p in prompts]
    logger.info("Running vLLM batch inference on %d prompts %s...", len(texts), batch_tag)
    t0 = time.time()
    outputs = llm.generate(texts, params)
    elapsed = time.time() - t0
    tokens_total = sum(len(o.outputs[0].token_ids) for o in outputs)
    logger.info(
        "vLLM done in %.1fs (%.0f tok/s, %.1f problems/s)",
        elapsed, tokens_total / elapsed, len(texts) / elapsed,
    )

    responses = []
    for o in outputs:
        responses.append(o.outputs[0].text)

    del llm
    import torch
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return responses


def process_responses(prompts, responses, output_dir):
    executor = Executor()
    results = []
    parse_ok, exec_ok, correct_ok = 0, 0, 0

    for prompt_info, response in zip(prompts, responses):
        item = prompt_info["item"]
        source = prompt_info["source"]
        idx = prompt_info["idx"]

        program = parse_program(response, f"{source}_p{idx}")
        if program is None:
            continue
        parse_ok += 1

        bindings = extract_bindings(item, program)
        success, result, _ = executor.execute(program, bindings)
        if not (success and result is not None):
            continue
        exec_ok += 1

        if not answer_matches(result, item["answer"]):
            continue
        correct_ok += 1

        results.append({
            "problem": item["problem"],
            "answer": item["answer"],
            "source": source,
            "program": program.to_dict(),
            "bindings": bindings,
            "exec_result": result,
        })

    total = len(prompts)
    logger.info(
        "Results: %d/%d parsed (%.1f%%), %d exec, %d correct (%.1f%%)",
        parse_ok, total, 100 * parse_ok / max(total, 1),
        exec_ok, correct_ok, 100 * correct_ok / max(total, 1),
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/template_config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output_dir", default="results/templates_pod")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Process in batches (0 = all at once)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_path = args.model or config["teacher"]["model"]
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  vLLM Fast Extraction")
    logger.info("  Model: %s, TP: %d", model_path, args.tp)
    logger.info("=" * 60)

    # Load datasets
    all_data = load_datasets(config)

    # Build prompts (use vLLM's built-in chat template)
    all_prompts = []
    for source, items in all_data.items():
        prompts = build_prompts(items, source)
        all_prompts.extend(prompts)
    logger.info("Total prompts: %d", len(all_prompts))

    # Run inference
    all_results = []
    batch_size = args.batch_size or len(all_prompts)

    for batch_start in range(0, len(all_prompts), batch_size):
        batch = all_prompts[batch_start:batch_start + batch_size]
        tag = f"[{batch_start}:{batch_start + len(batch)}]"
        responses = run_vllm_batch(batch, model_path, args.tp, args.max_new_tokens, tag)
        results = process_responses(batch, responses, args.output_dir)
        all_results.extend(results)

        # Save intermediate
        with open(os.path.join(args.output_dir, "all_programs.json"), "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d programs so far", len(all_results))

    # Save final stats
    by_source = {}
    for r in all_results:
        s = r["source"]
        by_source[s] = by_source.get(s, 0) + 1

    stats = {
        "total": len(all_results),
        "has_answer": len(all_results),
        "exec_ok": len(all_results),
        "answer_correct": len(all_results),
        "by_source": by_source,
    }
    with open(os.path.join(args.output_dir, "all_programs_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    meta = {
        "model": model_path,
        "tp": args.tp,
        "max_new_tokens": args.max_new_tokens,
        "total_prompts": len(all_prompts),
        "total_programs": len(all_results),
        "extraction_rate": len(all_results) / max(len(all_prompts), 1),
    }
    with open(os.path.join(args.output_dir, "extraction_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("=" * 60)
    logger.info("  Extraction Complete")
    logger.info("  Total programs: %d / %d (%.1f%%)",
                len(all_results), len(all_prompts),
                100 * len(all_results) / max(len(all_prompts), 1))
    logger.info("  By source: %s", by_source)
    logger.info("  Saved to: %s", args.output_dir)
    logger.info("=" * 60)

    # Mark done
    with open(os.path.join(args.output_dir, ".extraction_done"), "w") as f:
        f.write(f"done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
