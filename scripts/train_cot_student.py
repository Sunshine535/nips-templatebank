#!/usr/bin/env python3
"""Train CoT distillation student via LoRA SFT.

Takes CoT distillation data (from generate_cot_distill_data.py) and trains
a student model (Qwen3.5-9B) to reproduce teacher CoT reasoning directly.

This is the primary comparison baseline: distill CoT reasoning from the
teacher without any subroutine structure.

Training format:
  system: "Solve this math problem step by step."
  user:   <problem>
  assistant: <cot>\n\n#### <answer>
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
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "Solve this math problem step by step."


def find_latest_checkpoint(output_dir: str) -> str | None:
    ckpts = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0,
    )
    return ckpts[-1] if ckpts else None


def load_distill_data(train_file: str) -> Dataset:
    with open(train_file) as f:
        raw = json.load(f)

    formatted = []
    for item in raw:
        problem = item.get("problem", "")
        cot = item.get("cot", "")
        answer = item.get("answer", "")
        if not problem or not cot:
            continue
        # Build assistant response: CoT followed by answer marker
        assistant = cot
        if answer and f"#### {answer}" not in cot:
            assistant = f"{cot}\n\n#### {answer}"
        text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>"
        )
        formatted.append({"text": text})

    ds = Dataset.from_list(formatted)
    logger.info("Loaded %d CoT distillation examples from %s", len(ds), train_file)
    return ds


def main():
    parser = argparse.ArgumentParser(description="Train CoT distillation student (LoRA SFT)")
    parser.add_argument("--model", type=str, default=None, help="Student model name/path (default: from config)")
    parser.add_argument("--train_file", type=str, required=True, help="Path to distillation data JSON")
    parser.add_argument("--output_dir", type=str, default="results/planner/cot_distill")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "template_config.yaml"))
    parser.add_argument("--resume", type=str, default="auto", help="auto / none / path")
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = args.model or config["planner"]["model"]
    lora_cfg = config["planner"]["lora"]
    train_cfg = config["training"]

    # Override with CLI args
    train_cfg["num_train_epochs"] = args.epochs
    train_cfg["learning_rate"] = args.lr
    if args.max_seq_length:
        train_cfg["max_seq_length"] = args.max_seq_length

    logger.info("=" * 60)
    logger.info("  CoT Distillation Student Training")
    logger.info("  Model: %s", model_name)
    logger.info("  Data:  %s", args.train_file)
    logger.info("  Output: %s", args.output_dir)
    logger.info("  Epochs: %d | LR: %s", args.epochs, args.lr)
    logger.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_qlora = train_cfg.get("qlora", False)
    if use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Using QLoRA (4-bit quantization)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="sdpa",
            device_map="auto",
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
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

    dataset = load_distill_data(args.train_file)
    max_seq = train_cfg.get("max_seq_length", 2048)

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
        save_total_limit=train_cfg.get("save_total_limit", 3),
        max_length=max_seq,
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=False,
        report_to=train_cfg.get("report_to", "wandb"),
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    resume_ckpt = None
    if args.resume != "none":
        resume_ckpt = find_latest_checkpoint(args.output_dir) if args.resume == "auto" else args.resume
        if resume_ckpt:
            logger.info("Resuming from %s", resume_ckpt)

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "train_info.json"), "w") as f:
        json.dump({
            "mode": "cot_distill",
            "model": model_name,
            "dataset_size": len(dataset),
            "epochs": args.epochs,
            "lr": args.lr,
            "lora_r": lora_cfg["r"],
            "lora_alpha": lora_cfg["lora_alpha"],
            "target_modules": lora_cfg["target_modules"],
        }, f, indent=2)

    logger.info("Training complete: %s", args.output_dir)
    del model, trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
