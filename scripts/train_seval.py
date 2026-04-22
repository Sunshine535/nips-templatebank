#!/usr/bin/env python3
"""SEVAL Training Script: SFT + GRPO with verified execution reward.

Supports two modes:
  1. Flat program mode (default): model generates JSON-AST programs directly
  2. Compose mode: model generates composition plans using subroutine library

Usage:
    # SFT on verified programs (flat mode)
    torchrun --nproc_per_node=4 scripts/train_seval.py \
        --config configs/template_config.yaml \
        --mode sft --train_data results/templates_pod/all_programs.json

    # GRPO with execution reward (flat mode, from SFT checkpoint)
    torchrun --nproc_per_node=4 scripts/train_seval.py \
        --config configs/template_config.yaml \
        --mode grpo --resume results/seval/sft/model_final

    # Full pipeline: SFT then GRPO
    torchrun --nproc_per_node=4 scripts/train_seval.py \
        --config configs/template_config.yaml \
        --mode full --train_data results/templates_pod/all_programs.json
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.template_dsl import (
    CompositionExecutor,
    CompositionPlan,
    DType,
    Executor,
    Program,
    Slot,
    SubroutineLibrary,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="SEVAL: SFT + GRPO Training")
    p.add_argument("--config", type=str, default="configs/template_config.yaml")
    p.add_argument("--mode", type=str, choices=["sft", "grpo", "full"], default="full",
                   help="sft: supervised only, grpo: RL only, full: sft then grpo")
    p.add_argument("--train_data", type=str, default=None)
    p.add_argument("--eval_data", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="results/seval")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--sft_epochs", type=int, default=3)
    p.add_argument("--sft_lr", type=float, default=2e-4)
    p.add_argument("--sft_batch_size", type=int, default=4)
    p.add_argument("--grpo_steps", type=int, default=2000)
    p.add_argument("--grpo_lr", type=float, default=5e-6)
    p.add_argument("--grpo_batch_size", type=int, default=2)
    p.add_argument("--grpo_num_generations", type=int, default=8)
    p.add_argument("--grpo_temperature", type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--eval_samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def extract_numbers(text):
    """Extract numbers from problem text for binding."""
    numbers = re.findall(r'[\d,]+\.?\d*', text)
    result = []
    for n in numbers:
        cleaned = n.replace(",", "").strip()
        if cleaned and cleaned != ".":
            try:
                val = float(cleaned)
                result.append(val if "." in cleaned else int(float(cleaned)))
            except ValueError:
                continue
    return result


def normalize_training_data(raw_data):
    """Normalize various data formats to a unified schema."""
    normalized = []
    for item in raw_data:
        entry = {}
        entry["problem"] = (
            item.get("problem")
            or item.get("question")
            or item.get("instruction", "").split("Problem: ")[-1].split("\n")[0]
        )
        entry["answer"] = str(
            item.get("answer")
            or item.get("gold_answer")
            or item.get("exec_result", "")
        )
        if "program" in item:
            entry["program"] = item["program"]
        elif "output" in item:
            try:
                entry["program"] = json.loads(item["output"])
            except (json.JSONDecodeError, TypeError):
                entry["program"] = None
        if "bindings" in item:
            if isinstance(item["bindings"], str):
                try:
                    entry["bindings"] = json.loads(item["bindings"])
                except json.JSONDecodeError:
                    entry["bindings"] = {}
            else:
                entry["bindings"] = item["bindings"]
        else:
            entry["bindings"] = {}
        entry["source"] = item.get("source", "")
        if entry["program"] is not None and entry["problem"] and entry["answer"]:
            normalized.append(entry)
    return normalized


def setup_model(model_name, lora_r, lora_alpha, resume=None):
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map={"": local_rank}, trust_remote_code=True,
    )

    if resume:
        logger.info(f"Loading LoRA adapter from {resume}")
        model = PeftModel.from_pretrained(model, resume)
    else:
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model, tokenizer


def build_sft_dataset(data, tokenizer, max_length=1536):
    """Build SFT dataset: problem → JSON program."""
    from datasets import Dataset

    records = []
    for item in data:
        prog_json = json.dumps(item["program"], ensure_ascii=False)
        prompt = f"Problem: {item['problem']}\n\nGenerate an executable JSON program:"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": prog_json},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        records.append({"text": text})
    return Dataset.from_list(records)


def build_grpo_dataset(data, tokenizer):
    """Build GRPO dataset: prompts for program generation."""
    from datasets import Dataset

    records = []
    for i, item in enumerate(data):
        prompt = f"Problem: {item['problem']}\n\nGenerate an executable JSON program:"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        records.append({
            "prompt": text,
            "problem_idx": i,
        })
    return Dataset.from_list(records)


def make_reward_fn(data):
    """Create a reward function that verifies program execution against gold answers."""
    executor = Executor()
    problem_map = {}
    for i, item in enumerate(data):
        problem_map[i] = {
            "problem": item["problem"],
            "answer": item["answer"],
            "bindings": item.get("bindings", {}),
        }

    def reward_fn(completions, prompts=None, problem_idx=None, **kwargs):
        rewards = []
        for j, completion in enumerate(completions):
            idx = problem_idx[j] if problem_idx is not None else 0
            info = problem_map.get(idx, {})
            gold = info.get("answer", "")
            problem_text = info.get("problem", "")

            try:
                json_match = re.search(r'\{[\s\S]*\}', completion)
                if json_match is None:
                    rewards.append(0.0)
                    continue
                prog_data = json.loads(json_match.group())
                program = Program.from_dict(prog_data)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                rewards.append(0.0)
                continue

            numbers = extract_numbers(problem_text)
            bindings = {}
            for k, slot in enumerate(program.slots):
                if k < len(numbers):
                    bindings[slot.name] = numbers[k]
                else:
                    bindings[slot.name] = 0

            success, result, _ = executor.execute(program, bindings)
            if not success or result is None:
                rewards.append(0.1)
                continue

            try:
                r_val = float(result)
                g_val = float(gold)
                if g_val == 0:
                    correct = abs(r_val) < 1e-3
                else:
                    correct = abs(r_val - g_val) / max(abs(g_val), 1e-8) < 0.01
            except (ValueError, TypeError):
                correct = str(result).strip() == str(gold).strip()

            rewards.append(1.0 if correct else 0.2)
        return rewards

    return reward_fn


def run_sft(model, tokenizer, dataset, output_dir, args):
    """Run supervised fine-tuning."""
    from trl import SFTConfig, SFTTrainer

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=args.sft_batch_size,
        learning_rate=args.sft_lr,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to="none",
        max_seq_length=1536,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training...")
    trainer.train()
    model.save_pretrained(os.path.join(output_dir, "model_final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "model_final"))
    logger.info(f"SFT complete. Model saved to {output_dir}/model_final")
    return model


def run_grpo(model, tokenizer, dataset, reward_fn, output_dir, args):
    """Run GRPO training with execution-verification reward."""
    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=args.grpo_steps,
        per_device_train_batch_size=args.grpo_batch_size,
        learning_rate=args.grpo_lr,
        num_generations=args.grpo_num_generations,
        temperature=args.grpo_temperature,
        max_completion_length=args.max_new_tokens,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to="wandb",
        run_name=f"seval_grpo_seed{args.seed}",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO training...")
    trainer.train()
    model.save_pretrained(os.path.join(output_dir, "model_final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "model_final"))
    logger.info(f"GRPO complete. Model saved to {output_dir}/model_final")
    return model


def evaluate(model, tokenizer, eval_data, output_dir, max_samples=100):
    """Evaluate: generate programs for test problems, execute, check answers."""
    executor = Executor()
    model.eval()

    results = {"correct": 0, "executable": 0, "parsed": 0, "total": 0, "details": []}
    eval_subset = eval_data[:max_samples]

    for item in eval_subset:
        results["total"] += 1
        prompt = f"Problem: {item['problem']}\n\nGenerate an executable JSON program:"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match is None:
                continue
            prog = Program.from_dict(json.loads(json_match.group()))
            results["parsed"] += 1
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue

        numbers = extract_numbers(item["problem"])
        bindings = {slot.name: numbers[k] if k < len(numbers) else 0
                    for k, slot in enumerate(prog.slots)}
        success, result, _ = executor.execute(prog, bindings)

        if success and result is not None:
            results["executable"] += 1
            try:
                gold = float(item["answer"])
                pred = float(result)
                correct = abs(pred - gold) / max(abs(gold), 1e-8) < 0.01 if gold != 0 else abs(pred) < 1e-3
            except (ValueError, TypeError):
                correct = str(result).strip() == str(item["answer"]).strip()
            if correct:
                results["correct"] += 1
                results["details"].append({
                    "problem": item["problem"][:100],
                    "answer": item["answer"],
                    "predicted": str(result),
                })

    total = results["total"]
    results["accuracy"] = results["correct"] / total if total > 0 else 0
    results["parse_rate"] = results["parsed"] / total if total > 0 else 0
    results["exec_rate"] = results["executable"] / total if total > 0 else 0

    logger.info(f"Evaluation: {results['correct']}/{total} correct "
                f"({results['accuracy']:.1%}), "
                f"parsed={results['parsed']}, executable={results['executable']}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    model_name = args.model or cfg.get("planner", {}).get("model", "Qwen/Qwen3.5-9B")

    train_path = args.train_data or os.path.join("results", "templates_pod", "all_programs.json")
    logger.info(f"Loading training data from {train_path}")
    raw_data = json.load(open(train_path))
    data = normalize_training_data(raw_data)
    logger.info(f"Normalized: {len(data)} training examples (from {len(raw_data)} raw)")

    from sklearn.model_selection import train_test_split
    try:
        train_data, eval_data = train_test_split(data, test_size=0.1, random_state=args.seed)
    except Exception:
        split = int(len(data) * 0.9)
        train_data, eval_data = data[:split], data[split:]

    logger.info(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    sft_dir = os.path.join(args.output_dir, f"sft_seed{args.seed}")
    grpo_dir = os.path.join(args.output_dir, f"grpo_seed{args.seed}")

    if args.mode in ("sft", "full"):
        logger.info("=" * 60)
        logger.info("  Phase 1: Supervised Fine-Tuning on Verified Programs")
        logger.info("=" * 60)

        model, tokenizer = setup_model(model_name, args.lora_r, args.lora_alpha,
                                       resume=args.resume if args.mode == "sft" else None)
        sft_dataset = build_sft_dataset(train_data, tokenizer)
        model = run_sft(model, tokenizer, sft_dataset, sft_dir, args)

        logger.info("Evaluating SFT model...")
        sft_results = evaluate(model, tokenizer, eval_data, sft_dir, args.eval_samples)

        if args.mode == "sft":
            return

        resume_path = os.path.join(sft_dir, "model_final")
    else:
        resume_path = args.resume

    if args.mode in ("grpo", "full"):
        logger.info("=" * 60)
        logger.info("  Phase 2: GRPO with Verified Execution Reward")
        logger.info("=" * 60)

        if args.mode == "grpo":
            model, tokenizer = setup_model(model_name, args.lora_r, args.lora_alpha,
                                           resume=resume_path)
        grpo_dataset = build_grpo_dataset(train_data, tokenizer)
        reward_fn = make_reward_fn(train_data)
        model = run_grpo(model, tokenizer, grpo_dataset, reward_fn, grpo_dir, args)

        logger.info("Evaluating GRPO model...")
        grpo_results = evaluate(model, tokenizer, eval_data, grpo_dir, args.eval_samples)

    logger.info("=" * 60)
    logger.info("  Training Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
