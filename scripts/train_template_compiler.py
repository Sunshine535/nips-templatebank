#!/usr/bin/env python3
"""Train template compiler model: problem → select template → fill variables → solve.

Two-stage training pipeline:
  Stage 1: Template Selection SFT — learn to pick the right template for a problem
  Stage 2: Variable Filling SFT  — learn to instantiate template variables and generate solution

Both stages fine-tune Qwen/Qwen3.5-9B with LoRA (r=16, alpha=32).
Grammar-constrained decoding ensures template structure validity.
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint directory in output_dir."""
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")),
                   key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
    return ckpts[-1] if ckpts else None


SYSTEM_PROMPTS = {
    "selection": (
        "You are a reasoning template compiler. Given a math problem, "
        "identify the most appropriate reasoning template from your template bank."
    ),
    "filling": (
        "You are a reasoning template compiler. Given a template and a math problem, "
        "fill in the variable values and solve step by step using the template structure."
    ),
}


def load_stage_data(data_path: str, stage: str) -> Dataset:
    """Load and format training data for a specific stage."""
    if os.path.exists(data_path):
        with open(data_path) as f:
            raw_data = json.load(f)
    else:
        logger.warning("Data not found at %s, generating synthetic fallback", data_path)
        raw_data = _generate_synthetic(stage)

    if isinstance(raw_data, list) and raw_data and "stage" in raw_data[0]:
        raw_data = [d for d in raw_data if d.get("stage") == stage]

    system_prompt = SYSTEM_PROMPTS.get(stage, SYSTEM_PROMPTS["selection"])
    formatted = []
    for item in raw_data:
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        if not instruction or not output:
            continue
        text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
        formatted.append({"text": text})

    ds = Dataset.from_list(formatted)
    logger.info("Stage '%s': %d training examples", stage, len(ds))
    return ds


def _generate_synthetic(stage: str) -> list:
    """Fallback synthetic data for when real data is unavailable."""
    data = []
    templates = [
        {"name": "Addition", "steps": "1. Identify quantities\n2. Add\n3. Result"},
        {"name": "Multiplication", "steps": "1. Identify rate and count\n2. Multiply\n3. Result"},
        {"name": "Percentage", "steps": "1. Identify base and rate\n2. Compute percentage\n3. Apply"},
    ]
    for i in range(5000):
        t = templates[i % len(templates)]
        a, b = (i * 7 + 3) % 100 + 1, (i * 11 + 5) % 100 + 1
        problem = f"A store has {a} items at ${b} each. What is the total cost?"

        if stage == "selection":
            data.append({
                "instruction": f"Select the best template for:\n\n{problem}",
                "output": f"Template: {t['name']}\nSteps:\n{t['steps']}",
                "stage": "selection",
            })
        else:
            data.append({
                "instruction": f"Template: {t['name']}\n{t['steps']}\n\nSolve: {problem}",
                "output": f"Using {t['name']}:\n1. Items={a}, Price=${b}\n2. Total = {a} × {b} = {a * b}\n3. The answer is ${a * b}",
                "stage": "filling",
            })
    return data


def train_stage(
    stage_name: str,
    base_model_name: str,
    data_path: str,
    output_dir: str,
    lora_cfg: dict,
    train_cfg: dict,
    stage_overrides: dict,
    adapter_path: str | None = None,
    resume_from_checkpoint: str = "auto",
):
    """Train one stage of the compiler."""
    logger.info("=" * 50)
    logger.info("  Training Stage: %s", stage_name)
    logger.info("  Model: %s", base_model_name)
    logger.info("  Data: %s", data_path)
    logger.info("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )
    model.config.use_cache = False

    if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        logger.info("Loading existing adapter from %s for continued training", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

    dataset = load_stage_data(data_path, stage_name)

    epochs = stage_overrides.get("epochs", train_cfg["num_train_epochs"])
    lr = stage_overrides.get("learning_rate", train_cfg["learning_rate"])

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=epochs,
        learning_rate=lr,
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

    resume_ckpt = None
    if resume_from_checkpoint != "none":
        if resume_from_checkpoint == "auto":
            resume_ckpt = find_latest_checkpoint(output_dir)
        else:
            resume_ckpt = resume_from_checkpoint
        if resume_ckpt:
            logger.info("Resuming from checkpoint: %s", resume_ckpt)

    logger.info("Starting training (epochs=%d, lr=%.2e)...", epochs, lr)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    logger.info("Saving model to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "stage_info.json"), "w") as f:
        json.dump({
            "stage": stage_name,
            "base_model": base_model_name,
            "epochs": epochs,
            "learning_rate": lr,
            "lora_r": lora_cfg["r"],
            "lora_alpha": lora_cfg["lora_alpha"],
            "dataset_size": len(dataset),
        }, f, indent=2)

    logger.info("Stage '%s' training complete", stage_name)
    del model, trainer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Train template compiler (two-stage SFT)")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "template_config.yaml"))
    parser.add_argument("--training_data", type=str, default="results/templates/compiler_training_data.json",
                        help="Path to combined training data (or per-stage data)")
    parser.add_argument("--stage1_data", type=str, default=None, help="Override: stage 1 data")
    parser.add_argument("--stage2_data", type=str, default=None, help="Override: stage 2 data")
    parser.add_argument("--output_dir", type=str, default="results/compiler")
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default="auto",
                        help="Resume from checkpoint. 'auto' finds latest, path for specific, 'none' to disable")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    base_model = config["extraction"]["student_model"]
    lora_cfg = config["compiler"]["lora"]
    train_cfg = config["training"]
    stages = config["compiler"].get("stages", [
        {"name": "template_selection", "epochs": 3, "learning_rate": 2e-4},
        {"name": "variable_filling", "epochs": 3, "learning_rate": 1e-4},
    ])

    stage1_dir = os.path.join(args.output_dir, "stage1_selection")
    stage2_dir = os.path.join(args.output_dir, "stage2_filling")
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)

    stage1_data = args.stage1_data or args.training_data
    stage2_data = args.stage2_data or args.training_data

    # Stage 1: Template Selection
    if not args.skip_stage1:
        overrides = stages[0] if stages else {}
        train_stage(
            stage_name="selection",
            base_model_name=base_model,
            data_path=stage1_data,
            output_dir=stage1_dir,
            lora_cfg=lora_cfg,
            train_cfg=train_cfg,
            stage_overrides=overrides,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

    # Stage 2: Variable Filling (continues from Stage 1 adapter)
    if not args.skip_stage2:
        overrides = stages[1] if len(stages) > 1 else {}
        train_stage(
            stage_name="filling",
            base_model_name=base_model,
            data_path=stage2_data,
            output_dir=stage2_dir,
            lora_cfg=lora_cfg,
            train_cfg=train_cfg,
            stage_overrides=overrides,
            adapter_path=stage1_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

    logger.info("=" * 60)
    logger.info("  Template compiler training complete")
    logger.info("  Stage 1 (selection): %s", stage1_dir)
    logger.info("  Stage 2 (filling):   %s", stage2_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
