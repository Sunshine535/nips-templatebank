#!/usr/bin/env python3
"""Train one ablation variant (GPT-5.5 Task 10 A/B/C).

Variants:
  old_fragment_only    — Flat SFT on 697 verified programs (Variant A)
  gift_no_call_output  — GIFT data but call_output refs stripped
  gift_no_active_gate  — GIFT data unfiltered by active-binding audit
  full_gift_step       — Full GIFT on step-primitive data (Variant C)

Usage (torchrun):
  torchrun --nproc_per_node=4 scripts/train_ablation.py \
      --variant full_gift_step \
      --seed 42 \
      --output_dir results/gift_ablation/full_gift_step/seed42
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _seed_everything(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hash(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as fp:
        return hashlib.sha256(fp.read(1024 * 1024)).hexdigest()[:16]


def build_records(variant: str, tokenizer, step_data_path: str, flat_data_path: str):
    """Build SFT records for each variant."""
    records = []

    if variant == "old_fragment_only":
        data = json.load(open(flat_data_path))
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

    elif variant == "flat_matched_565":
        gift_data = json.load(open(step_data_path))
        gift_problems = {item["problem"] for item in gift_data}
        flat_data = json.load(open(flat_data_path))
        matched = [item for item in flat_data if item["problem"] in gift_problems]
        for item in matched:
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

    elif variant == "gift_no_explicit_refs_oracle_values":
        data = json.load(open(step_data_path))
        from src.dataflow_plan import BindingRef, DataflowExecutor, DataflowPlan
        from src.template_dsl import SubroutineLibrary
        lib_path = step_data_path.replace("compose_train_gift.json", "library_gift.json")
        library = SubroutineLibrary.load(lib_path)
        executor = DataflowExecutor(library)
        for item in data:
            plan_dict = json.loads(json.dumps(item["plan"]))
            quantities = item.get("quantities", {})
            qty_map = {qid: q["value"] for qid, q in quantities.items()}
            plan_obj = DataflowPlan.from_dict(plan_dict)
            ok, _, _ = executor.execute_with_quantities(plan_obj, qty_map)
            if not ok:
                continue
            call_outputs = {}
            for call in plan_obj.calls:
                sub = library.get(call.sub_id)
                if sub is None:
                    break
                args_d = {}
                for slot_name, ref in call.bindings.items():
                    if ref.source == "quantity":
                        args_d[slot_name] = qty_map.get(ref.qid, ref.value)
                    elif ref.source == "call_output":
                        args_d[slot_name] = call_outputs.get(ref.call_id)
                    else:
                        args_d[slot_name] = ref.value
                ok2, result, _ = executor.executor.execute(sub.program, args_d)
                if not ok2:
                    break
                call_outputs[call.call_id] = result
            for call in plan_dict["calls"]:
                for slot, ref in call["bindings"].items():
                    if isinstance(ref, dict) and ref.get("source") == "call_output":
                        cid = ref["call_id"]
                        if cid in call_outputs:
                            ref["source"] = "constant"
                            ref["value"] = call_outputs[cid]
                            ref.pop("call_id", None)
            plan_json = json.dumps(plan_dict, ensure_ascii=False)
            prompt = f"Problem: {item['problem']}\n\nGenerate a dataflow composition plan (JSON):"
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": plan_json},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            records.append({"text": text})

    elif variant == "full_gift_step":
        data = json.load(open(step_data_path))
        for item in data:
            plan_json = json.dumps(item["plan"], ensure_ascii=False)
            prompt = f"Problem: {item['problem']}\n\nGenerate a dataflow composition plan (JSON):"
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": plan_json},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            records.append({"text": text})

    elif variant == "gift_no_call_output":
        data = json.load(open(step_data_path))
        qty_values = {}
        for item in data:
            plan = json.loads(json.dumps(item["plan"]))
            quantities = item.get("quantities", {})
            for call in plan.get("calls", []):
                for slot, ref in call.get("bindings", {}).items():
                    if isinstance(ref, dict) and ref.get("source") == "call_output":
                        ref["source"] = "quantity"
                        if quantities:
                            first_qid = next(iter(quantities.keys()))
                            ref["qid"] = first_qid
                            ref["value"] = quantities[first_qid]["value"]
                        else:
                            ref["value"] = 0
                        ref.pop("call_id", None)
            plan_json = json.dumps(plan, ensure_ascii=False)
            prompt = f"Problem: {item['problem']}\n\nGenerate a dataflow composition plan (JSON):"
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": plan_json},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            records.append({"text": text})

    elif variant == "gift_no_active_gate":
        data = json.load(open(step_data_path))
        for item in data:
            plan_json = json.dumps(item["plan"], ensure_ascii=False)
            prompt = f"Problem: {item['problem']}\n\nGenerate a dataflow composition plan (JSON):"
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": plan_json},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            records.append({"text": text})

    elif variant == "vgift_value_only":
        # Same as gift_no_explicit_refs_oracle_values (E control)
        return build_records("gift_no_explicit_refs_oracle_values", tokenizer,
                             step_data_path, flat_data_path)

    elif variant == "vgift_no_value_hints":
        # Same as full_gift_step (C control — refs only, no value hints)
        return build_records("full_gift_step", tokenizer,
                             step_data_path, flat_data_path)

    elif variant == "vgift_no_consistency":
        # Refs + value hints but no consistency gate in labels
        # (same data as vgift_full, consistency is an eval-time concept)
        return build_records("vgift_full", tokenizer,
                             step_data_path, flat_data_path)

    elif variant == "vgift_full":
        # Full V-GIFT: refs + value_hint annotations per call
        data = json.load(open(step_data_path))
        from src.dataflow_plan import DataflowExecutor, DataflowPlan
        from src.template_dsl import SubroutineLibrary
        lib_path = step_data_path.replace("compose_train_gift.json", "library_gift.json")
        library = SubroutineLibrary.load(lib_path)
        executor = DataflowExecutor(library)
        for item in data:
            plan_dict = json.loads(json.dumps(item["plan"]))
            quantities = item.get("quantities", {})
            qty_map = {qid: q["value"] for qid, q in quantities.items()}
            plan_obj = DataflowPlan.from_dict(plan_dict)
            call_outputs = {}
            for call in plan_obj.calls:
                sub = library.get(call.sub_id)
                if sub is None:
                    break
                args_d = {}
                for slot_name, ref in call.bindings.items():
                    if ref.source == "quantity":
                        args_d[slot_name] = qty_map.get(ref.qid, ref.value)
                    elif ref.source == "call_output":
                        args_d[slot_name] = call_outputs.get(ref.call_id)
                    else:
                        args_d[slot_name] = ref.value
                ok, result, _ = executor.executor.execute(sub.program, args_d)
                if ok and result is not None:
                    call_outputs[call.call_id] = result
            # Add value_hint to each call in the plan
            for call in plan_dict["calls"]:
                cid = call["call_id"]
                if cid in call_outputs:
                    call["value_hint"] = {"value": call_outputs[cid]}
            plan_json = json.dumps(plan_dict, ensure_ascii=False)
            prompt = f"Problem: {item['problem']}\n\nGenerate a value-annotated dataflow plan (JSON):"
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": plan_json},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            records.append({"text": text})

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["old_fragment_only", "flat_matched_565",
                                 "gift_no_call_output", "gift_no_active_gate",
                                 "gift_no_explicit_refs_oracle_values",
                                 "full_gift_step",
                                 "vgift_full", "vgift_no_value_hints",
                                 "vgift_no_consistency", "vgift_value_only"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="/root/assets/models/Qwen3.5-9B")
    parser.add_argument("--step_data",
                        default="results/gift_step/compose_train_gift.json")
    parser.add_argument("--flat_data",
                        default="results/templates_verified/all_programs.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    args = parser.parse_args()

    _seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Variant: %s, Seed: %d", args.variant, args.seed)
    logger.info("Model: %s", args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = build_records(
        args.variant, tokenizer, args.step_data, args.flat_data,
    )
    logger.info("Training records: %d", len(records))
    dataset = Dataset.from_list(records)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map={"": local_rank}, trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    manifest = {
        "variant": args.variant,
        "seed": args.seed,
        "n_training_records": len(records),
        "step_data_sha256_prefix": _hash(args.step_data),
        "flat_data_sha256_prefix": _hash(args.flat_data),
        "model": args.model,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "command": " ".join(sys.argv),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
    }
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        manifest["git_commit"] = commit[:10]
    except Exception:
        pass
    with open(os.path.join(args.output_dir, "train_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to="none",
        max_length=args.max_length,
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model, args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    logger.info("Starting training for variant %s seed %d ...", args.variant, args.seed)
    trainer.train()

    final_dir = os.path.join(args.output_dir, "model_final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Saved to %s", final_dir)


if __name__ == "__main__":
    main()
