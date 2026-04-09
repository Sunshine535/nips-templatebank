#!/usr/bin/env python3
"""SEVAL Training Script: GRPO + Library Evolution.

Usage:
    # Single GPU
    python scripts/train_seval.py --config configs/template_config.yaml

    # Multi-GPU
    torchrun --nproc_per_node=4 scripts/train_seval.py --config configs/template_config.yaml

    # With custom evolution params
    python scripts/train_seval.py \\
        --config configs/template_config.yaml \\
        --evolution_interval 200 \\
        --evolution_rounds 5 \\
        --grpo_num_steps 2000

    # Resume from checkpoint
    python scripts/train_seval.py --resume results/seval/checkpoint-1000
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.template_dsl import SubroutineLibrary, CompositionPlan, CompositionExecutor
from src.rlvr_evolution import (
    CompositionReward,
    LibraryEvolver,
    SEVALConfig,
    SEVALTrainer,
    cot_pass_at_k,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="SEVAL: GRPO + Library Evolution")
    p.add_argument("--config", type=str, default="configs/template_config.yaml")
    p.add_argument("--library", type=str, default=None,
                   help="Path to initial library JSON (L₀)")
    p.add_argument("--train_data", type=str, default=None,
                   help="Path to training data JSON (composition plans)")
    p.add_argument("--eval_data", type=str, default=None,
                   help="Path to evaluation data JSON")
    p.add_argument("--output_dir", type=str, default="results/seval")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint directory")

    # GRPO params
    p.add_argument("--grpo_num_steps", type=int, default=2000)
    p.add_argument("--grpo_num_generations", type=int, default=8)
    p.add_argument("--grpo_learning_rate", type=float, default=5e-6)
    p.add_argument("--grpo_batch_size", type=int, default=4)
    p.add_argument("--grpo_temperature", type=float, default=0.8)

    # Evolution params
    p.add_argument("--evolution_interval", type=int, default=200)
    p.add_argument("--evolution_rounds", type=int, default=5)
    p.add_argument("--min_pattern_count", type=int, default=5)
    p.add_argument("--min_success_rate", type=float, default=0.7)
    p.add_argument("--min_mdl_gain", type=float, default=0.5)
    p.add_argument("--max_library_size", type=int, default=64)

    # Eval params
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_max_samples", type=int, default=64)

    # Model
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)

    # Multi-GPU
    p.add_argument("--local_rank", type=int, default=-1)

    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def setup_model(model_name: str, lora_r: int, lora_alpha: int, resume: str = None):
    """Load model with LoRA for GRPO training."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        trust_remote_code=True,
    )

    if resume:
        logger.info(f"Loading LoRA adapter from {resume}")
        model = PeftModel.from_pretrained(model, os.path.join(resume, "model"))
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def build_grpo_dataset(train_data: list, library: SubroutineLibrary, tokenizer):
    """Build a dataset for GRPO training.

    Each example has a prompt (problem + library) and the model generates
    composition plans. Rewards come from execution verification.
    """
    from datasets import Dataset

    lib_sigs = "\n".join(library.signatures())
    records = []

    for i, example in enumerate(train_data):
        question = example.get("question", "")
        prompt = (
            f"Available subroutines:\n{lib_sigs}\n\n"
            f"Problem: {question}\n\n"
            f"Output a composition plan as JSON. Use the format: "
            f'{{"plan": [{{"sub_id": "L00", "bindings": {{"x": 5}}}}]}}\n'
            f"Plan:"
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        records.append({
            "prompt": text,
            "problem_id": str(i),
            "question": question,
            "bindings": json.dumps(example.get("bindings", {})),
            "gold_answer": str(example.get("gold_answer", "")),
        })

    return Dataset.from_list(records)


def train_grpo(
    model,
    tokenizer,
    dataset,
    reward_fn,
    config: SEVALConfig,
    output_dir: str,
):
    """Run GRPO training using TRL."""
    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=config.grpo_num_steps,
        per_device_train_batch_size=config.grpo_batch_size,
        learning_rate=config.grpo_learning_rate,
        num_generations=config.grpo_num_generations,
        temperature=config.grpo_temperature,
        max_completion_length=config.grpo_max_new_tokens,
        logging_steps=10,
        save_steps=config.save_interval,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to="wandb",
        run_name="seval_grpo",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    return trainer


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    cfg = load_config(args.config)
    model_name = args.model or cfg.get("planner", {}).get("model", "Qwen/Qwen3.5-9B")

    # Build SEVAL config
    seval_config = SEVALConfig(
        grpo_num_generations=args.grpo_num_generations,
        grpo_temperature=args.grpo_temperature,
        grpo_max_new_tokens=cfg.get("datasets", {}).get("math", {}).get("max_new_tokens_plan", 512),
        grpo_learning_rate=args.grpo_learning_rate,
        grpo_num_steps=args.grpo_num_steps,
        grpo_batch_size=args.grpo_batch_size,
        evolution_interval=args.evolution_interval,
        evolution_rounds=args.evolution_rounds,
        min_pattern_count=args.min_pattern_count,
        min_success_rate=args.min_success_rate,
        min_mdl_gain=args.min_mdl_gain,
        max_library_size=args.max_library_size,
        eval_interval=args.eval_interval,
        eval_max_samples=args.eval_max_samples,
        output_dir=args.output_dir,
        save_interval=500,
    )

    # Load library
    library_path = args.library or os.path.join(
        cfg.get("output_dir", "results"), "templates_verified", "subroutine_library.json"
    )
    logger.info(f"Loading initial library from {library_path}")
    library = SubroutineLibrary.load(library_path)
    logger.info(f"Library L₀: {library.size} subroutines")
    logger.info(f"Library snapshot: {library.snapshot()}")

    # Load data
    train_path = args.train_data or os.path.join(
        cfg.get("output_dir", "results"), "templates_verified", "compose_train.json"
    )
    eval_path = args.eval_data or os.path.join(
        cfg.get("output_dir", "results"), "templates_verified", "flat_train.json"
    )
    logger.info(f"Loading train data from {train_path}")
    train_data = load_data(train_path)
    logger.info(f"Loading eval data from {eval_path}")
    eval_data = load_data(eval_path) if os.path.exists(eval_path) else train_data[:100]

    holdout_data = train_data[-50:]  # last 50 for evolution verification
    train_data = train_data[:-50]

    # Setup model
    model, tokenizer = setup_model(
        model_name, args.lora_r, args.lora_alpha,
        resume=args.resume,
    )

    # Build reward function
    reward_fn = CompositionReward(library)

    # Build evolver
    evolver = LibraryEvolver(
        library=library,
        min_pattern_count=seval_config.min_pattern_count,
        min_success_rate=seval_config.min_success_rate,
        min_mdl_gain=seval_config.min_mdl_gain,
        max_library_size=seval_config.max_library_size,
    )

    # Build dataset
    dataset = build_grpo_dataset(train_data, library, tokenizer)
    logger.info(f"Dataset: {len(dataset)} examples")

    # Build problem map for reward function
    problem_map = {}
    for i, example in enumerate(train_data):
        problem_map[str(i)] = {
            "bindings": example.get("bindings", {}),
            "gold_answer": example.get("gold_answer"),
        }

    # Create TRL-compatible reward function
    def trl_reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        for completion in completions:
            # Simple binary reward based on execution
            result = reward_fn(completion, {}, None)
            if result.plan_parsed and result.execution_success:
                rewards.append(1.0)
            elif result.plan_parsed:
                rewards.append(0.1)
            else:
                rewards.append(0.0)

            # Record for evolution
            if result.plan is not None:
                evolver.record(
                    result.plan, {},
                    result.answer_correct or result.execution_success,
                )
        return rewards

    # GRPO Training with evolution loop
    os.makedirs(args.output_dir, exist_ok=True)

    # Save initial state
    library.save(os.path.join(args.output_dir, "library_L0.json"))

    logger.info("=" * 60)
    logger.info("SEVAL Training Starting")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Library L₀: {library.size} subroutines")
    logger.info(f"  GRPO steps: {seval_config.grpo_num_steps}")
    logger.info(f"  Evolution interval: {seval_config.evolution_interval}")
    logger.info(f"  Max evolution rounds: {seval_config.evolution_rounds}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 60)

    # Build GRPO trainer
    trainer = train_grpo(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_fn=trl_reward_fn,
        config=seval_config,
        output_dir=args.output_dir,
    )

    # Register evolution callback
    from transformers import TrainerCallback

    class EvolutionCallback(TrainerCallback):
        def __init__(self, evolver, library, reward_fn, config, holdout, output_dir):
            self.evolver = evolver
            self.library = library
            self.reward_fn = reward_fn
            self.config = config
            self.holdout = holdout
            self.output_dir = output_dir
            self.evolution_round = 0

        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step
            if step > 0 and step % self.config.evolution_interval == 0:
                if self.evolution_round < self.config.evolution_rounds:
                    logger.info(f"\n{'='*40}")
                    logger.info(f"Evolution Round {self.evolution_round + 1} at step {step}")
                    logger.info(f"{'='*40}")

                    result = self.evolver.evolve(self.holdout)

                    if result["evolved"]:
                        self.evolution_round += 1
                        self.reward_fn.update_library(self.library)
                        lib_path = os.path.join(
                            self.output_dir,
                            f"library_L{self.evolution_round}.json",
                        )
                        self.library.save(lib_path)
                        logger.info(
                            f"Library evolved: {result['library_size_before']} → "
                            f"{result['library_size_after']}"
                        )
                        for sub_info in result.get("new_subroutines", []):
                            logger.info(f"  New: {sub_info['sub_id']} from {sub_info['source_pattern']}")
                    else:
                        logger.info(f"No evolution: {result.get('reason', 'unknown')}")

                    self.evolver.clear_buffers()

    trainer.add_callback(EvolutionCallback(
        evolver=evolver,
        library=library,
        reward_fn=reward_fn,
        config=seval_config,
        holdout=holdout_data,
        output_dir=args.output_dir,
    ))

    # Train
    logger.info("Starting GRPO training...")
    trainer.train()

    # Save final state
    library.save(os.path.join(args.output_dir, "library_final.json"))
    model.save_pretrained(os.path.join(args.output_dir, "model_final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer_final"))

    # Final CoT-Pass@K evaluation
    logger.info("Running final CoT-Pass@K evaluation...")
    final_metrics = cot_pass_at_k(
        model=model,
        tokenizer=tokenizer,
        problems=eval_data[:100],
        library=library,
        k_values=[1, 4, 16, 64],
        max_samples=64,
    )
    logger.info(f"Final CoT-Pass@K: {final_metrics}")

    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump({
            "cot_passk": final_metrics,
            "library_final": library.snapshot(),
            "evolution_history": evolver.evolution_history,
        }, f, indent=2)

    logger.info("SEVAL training complete!")
    logger.info(f"Final library: {library.size} subroutines")
    logger.info(f"Evolution rounds: {evolver.evolution_history}")


if __name__ == "__main__":
    main()
