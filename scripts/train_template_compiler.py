#!/usr/bin/env python3
"""Train template compiler: input problem → select template → instantiate and solve.
SFT on Qwen/Qwen3.5-9B with LoRA."""

import argparse
import json
import logging
import os
import sys

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_training_data(data_path: str, tokenizer) -> Dataset:
    """Load compiler training data and format for SFT."""
    logger.info("Loading training data from %s", data_path)

    if os.path.exists(data_path):
        with open(data_path) as f:
            raw_data = json.load(f)
    else:
        logger.warning("Training data not found at %s, generating synthetic", data_path)
        raw_data = generate_synthetic_compiler_data()

    formatted = []
    for item in raw_data:
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        if not instruction or not output:
            continue
        text = (
            f"<|im_start|>system\nYou are a reasoning template compiler. "
            f"Given a problem, select the best template and solve step by step.<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
        formatted.append({"text": text})

    ds = Dataset.from_list(formatted)
    logger.info("Training dataset: %d examples", len(ds))
    return ds


def generate_synthetic_compiler_data():
    """Generate synthetic template compiler training data."""
    data = []
    templates = [
        {
            "name": "Addition Template",
            "steps": "1. Identify quantities\n2. Add quantities\n3. State result",
        },
        {
            "name": "Multiplication Template",
            "steps": "1. Identify rate and quantity\n2. Multiply\n3. State result",
        },
        {
            "name": "Percentage Template",
            "steps": "1. Identify base and percentage\n2. Compute percentage\n3. Apply to base",
        },
    ]
    for i in range(5000):
        t = templates[i % len(templates)]
        a, b = (i * 7 + 3) % 100 + 1, (i * 11 + 5) % 100 + 1
        if i % 3 == 0:
            problem = f"A store has {a} items. They receive {b} more. How many total?"
            solution = f"Template: {t['name']}\n{t['steps']}\nTotal = {a} + {b} = {a + b}"
        elif i % 3 == 1:
            problem = f"If each box has {a} items and there are {b} boxes, how many items total?"
            solution = f"Template: {t['name']}\n{t['steps']}\nTotal = {a} × {b} = {a * b}"
        else:
            problem = f"What is {a}% of {b * 10}?"
            solution = f"Template: {t['name']}\n{t['steps']}\nResult = {a}/100 × {b * 10} = {a * b * 10 / 100}"
        data.append({"instruction": f"Solve using template:\n\nProblem: {problem}", "output": solution})
    return data


def main():
    parser = argparse.ArgumentParser(description="Train template compiler")
    parser.add_argument("--config", type=str, default="configs/template_config.yaml")
    parser.add_argument("--training_data", type=str, default="outputs/templates/compiler_training_data.json")
    parser.add_argument("--output_dir", type=str, default="outputs/compiler")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    base_model = config["extraction"]["student_model"]
    lora_cfg = config["compiler"]["lora"]
    train_cfg = config["training"]

    logger.info("=== Training Template Compiler ===")
    logger.info("Model: %s", base_model)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    dataset = load_training_data(args.training_data, tokenizer)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=3,
        max_seq_length=train_cfg["max_seq_length"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=False,
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("=== Template compiler training complete ===")


if __name__ == "__main__":
    main()
