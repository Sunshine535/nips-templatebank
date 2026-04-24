import json, os, re, sys, torch, logging
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, "/root/nips-templatebank")
from src.template_dsl import DType, Executor, Program
from src.dataflow_plan import BindingRef, DataflowExecutor, DataflowPlan, PlanCall, SubroutineLibrary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def extract_numbers(text):
    numbers = re.findall(r"[\d,]+\.?\d*", text)
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

ds = load_dataset("openai/gsm8k", "main", split="test")
logger.info("GSM8K test: %d", len(ds))

model_name = "/root/assets/models/Qwen3.5-9B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

max_samples = 200
test_data = list(ds)[:max_samples]

# Load GIFT model
logger.info("Loading GIFT SFT model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, "/root/nips-templatebank/results/gift/sft_gift_seed42/model_final")
model.eval()

# Load library for DataflowExecutor
library = SubroutineLibrary.load("/root/nips-templatebank/results/templates_verified/subroutine_library.json")
df_executor = DataflowExecutor(library)
flat_executor = Executor()

correct_gift = 0
correct_flat_fallback = 0
parsed_gift = 0
parsed_flat = 0
exec_gift = 0
total = 0

for i, item in enumerate(test_data):
    total += 1
    problem = item["question"]
    answer = item["answer"].split("####")[-1].strip()

    prompt = "Problem: " + problem + "\n\nGenerate a dataflow composition plan (JSON):"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Try parsing as DataflowPlan first
    try:
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match is None:
            continue
        plan_data = json.loads(json_match.group())

        if "calls" in plan_data:
            plan = DataflowPlan.from_dict(plan_data)
            parsed_gift += 1
            ok, result, stats = df_executor.execute(plan)
            if ok and result is not None:
                exec_gift += 1
                try:
                    gold = float(answer.replace(",", ""))
                    pred = float(result)
                    is_correct = abs(pred - gold) / max(abs(gold), 1e-8) < 0.01 if gold != 0 else abs(pred) < 1e-3
                except (ValueError, TypeError):
                    is_correct = str(result).strip() == answer.strip()
                if is_correct:
                    correct_gift += 1
        else:
            prog = Program.from_dict(plan_data)
            parsed_flat += 1
            numbers = extract_numbers(problem)
            bindings = {slot.name: numbers[k] if k < len(numbers) else 0 for k, slot in enumerate(prog.slots)}
            ok, result, _ = flat_executor.execute(prog, bindings)
            if ok and result is not None:
                try:
                    gold = float(answer.replace(",", ""))
                    pred = float(result)
                    is_correct = abs(pred - gold) / max(abs(gold), 1e-8) < 0.01 if gold != 0 else abs(pred) < 1e-3
                except (ValueError, TypeError):
                    is_correct = str(result).strip() == answer.strip()
                if is_correct:
                    correct_flat_fallback += 1
    except Exception:
        continue

    if (i + 1) % 50 == 0:
        logger.info("[gift] %d/%d: gift_correct=%d, flat_fallback=%d, parsed_gift=%d, parsed_flat=%d",
                   i + 1, max_samples, correct_gift, correct_flat_fallback, parsed_gift, parsed_flat)
        sys.stdout.flush()
        sys.stderr.flush()

total_correct = correct_gift + correct_flat_fallback
logger.info("=" * 60)
logger.info("  GIFT Evaluation on GSM8K Test (%d samples)", max_samples)
logger.info("=" * 60)
logger.info("  GIFT plans correct: %d/%d (%.1f%%)", correct_gift, total, 100 * correct_gift / total)
logger.info("  Flat fallback correct: %d", correct_flat_fallback)
logger.info("  Total correct: %d/%d (%.1f%%)", total_correct, total, 100 * total_correct / total)
logger.info("  Parsed as GIFT plan: %d", parsed_gift)
logger.info("  Parsed as flat program: %d", parsed_flat)
logger.info("  GIFT exec OK: %d", exec_gift)

results = {
    "model": "gift_sft_seed42",
    "dataset": "gsm8k_test",
    "samples": max_samples,
    "gift_correct": correct_gift,
    "flat_fallback_correct": correct_flat_fallback,
    "total_correct": total_correct,
    "accuracy": total_correct / total,
    "method_accuracy": correct_gift / total,
    "parsed_gift": parsed_gift,
    "parsed_flat": parsed_flat,
    "exec_gift": exec_gift,
}
os.makedirs("/root/nips-templatebank/results/gift/gsm8k_test", exist_ok=True)
with open("/root/nips-templatebank/results/gift/gsm8k_test/results.json", "w") as f:
    json.dump(results, f, indent=2)
logger.info("Saved to results/gift/gsm8k_test/results.json")
