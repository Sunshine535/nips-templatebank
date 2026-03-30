#!/usr/bin/env python3
"""Train subroutine planner: problem -> JSON composition plan.

Two training modes:
  compose: problem + library signatures -> composition plan JSON
  flat:    problem -> flat executable program JSON (baseline)

Both use Qwen3.5-9B + LoRA with grammar-constrained output.
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
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir: str) -> str | None:
    ckpts = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0,
    )
    return ckpts[-1] if ckpts else None


SYSTEM_PROMPTS = {
    "compose": (
        "You are a subroutine planner. Given a math problem and available subroutines, "
        "output a JSON composition plan that solves the problem by calling subroutines."
    ),
    "flat": (
        "You are a program compiler. Given a math problem, "
        "output an executable JSON program that solves the problem step by step."
    ),
}


def load_training_data(data_path: str, mode: str) -> Dataset:
    if not os.path.exists(data_path):
        logger.warning("Data not found at %s, generating synthetic", data_path)
        return _synthetic_data(mode)

    with open(data_path) as f:
        raw = json.load(f)

    system_prompt = SYSTEM_PROMPTS[mode]
    formatted = []
    for item in raw:
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
    logger.info("Loaded %d training examples for mode='%s'", len(ds), mode)
    return ds


def _synthetic_data(mode: str) -> Dataset:
    data = []
    for i in range(2000):
        a, b = (i * 7 + 3) % 100 + 1, (i * 11 + 5) % 50 + 1
        problem = f"A store has {a} items at ${b} each. What is the total cost?"
        if mode == "compose":
            output = json.dumps({"plan": [{"sub_id": "L00", "bindings": {"quantity": a, "price": b}}]})
        else:
            output = json.dumps({
                "program_id": f"flat_{i}",
                "slots": [{"name": "quantity", "dtype": "int"}, {"name": "price", "dtype": "float"}],
                "steps": [
                    {"op": "compute", "target": "total", "expr": "quantity * price", "inputs": ["quantity", "price"], "target_dtype": "float"},
                    {"op": "output", "target": "__output__", "expr": "total", "inputs": ["total"], "target_dtype": "float"},
                ],
            })
        text = (
            f"<|im_start|>system\n{SYSTEM_PROMPTS[mode]}<|im_end|>\n"
            f"<|im_start|>user\nProblem: {problem}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
        data.append({"text": text})
    return Dataset.from_list(data)


def train(
    mode: str,
    model_name: str,
    data_path: str,
    output_dir: str,
    lora_cfg: dict,
    train_cfg: dict,
    resume: str = "auto",
):
    logger.info("=" * 60)
    logger.info("  Training mode=%s | model=%s", mode, model_name)
    logger.info("  Data: %s | Output: %s", data_path, output_dir)
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
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

    dataset = load_training_data(data_path, mode)
    max_seq = train_cfg.get("max_seq_length", 2048)

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg.get("save_total_limit", 3),
        max_seq_length=max_seq,
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=False,
        report_to=train_cfg.get("report_to", "wandb"),
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
    if resume != "none":
        resume_ckpt = find_latest_checkpoint(output_dir) if resume == "auto" else resume
        if resume_ckpt:
            logger.info("Resuming from %s", resume_ckpt)

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "train_info.json"), "w") as f:
        json.dump({
            "mode": mode,
            "model": model_name,
            "dataset_size": len(dataset),
            "epochs": train_cfg["num_train_epochs"],
            "lr": train_cfg["learning_rate"],
            "lora_r": lora_cfg["r"],
        }, f, indent=2)

    logger.info("Training complete: %s", output_dir)
    del model, trainer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Train subroutine planner / flat-program baseline")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "template_config.yaml"))
    parser.add_argument("--mode", type=str, default="compose", choices=["compose", "flat"])
    parser.add_argument("--training_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default="auto", help="auto / none / path")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_seq_length", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["planner"]["model"]
    lora_cfg = config["planner"]["lora"]
    train_cfg = config["training"]

    if args.max_seq_length:
        train_cfg["max_seq_length"] = args.max_seq_length

    if args.mode == "compose":
        data_path = args.training_data or "results/templates/compose_train.json"
        output_dir = args.output_dir or "results/planner/compose"
    else:
        data_path = args.training_data or "results/templates/flat_train.json"
        output_dir = args.output_dir or "results/planner/flat"

    train(
        mode=args.mode,
        model_name=model_name,
        data_path=data_path,
        output_dir=output_dir,
        lora_cfg=lora_cfg,
        train_cfg=train_cfg,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
