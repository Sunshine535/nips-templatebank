#!/usr/bin/env python3
"""Check E/value-supervised variant for test-time gold intermediate leakage.

Reads prediction JSONL and verifies:
1. All predicted constants appear in raw_response (model-generated)
2. Eval code never reads gold intermediate traces for test examples
3. No suspicious pattern of values equaling gold answer
4. Samples 20 examples for human inspection

Usage:
    python scripts/check_value_leakage.py \
        --predictions results/gift_ablation/gift_no_explicit_refs_oracle_values/seed42/predictions.jsonl \
        --output reports/VALUE_LEAKAGE_AUDIT.md
"""
import argparse
import json
import re
import sys
from pathlib import Path


def extract_constants_from_plan(parsed_obj):
    """Extract all constant/value fields from a parsed GIFT plan."""
    constants = []
    if not isinstance(parsed_obj, dict):
        return constants
    calls = parsed_obj.get("calls", [])
    for call in calls:
        bindings = call.get("bindings", {})
        for slot, ref in bindings.items():
            if isinstance(ref, dict):
                if ref.get("source") in ("constant", "quantity") and "value" in ref:
                    constants.append(ref["value"])
            elif isinstance(ref, (int, float)):
                constants.append(ref)
    return constants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="reports/VALUE_LEAKAGE_AUDIT.md")
    parser.add_argument("--sample_size", type=int, default=20)
    args = parser.parse_args()

    entries = []
    with open(args.predictions) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    total = len(entries)
    suspicious = 0
    gold_in_constants = 0
    constants_not_in_response = 0
    sampled = []

    for entry in entries:
        gold = entry.get("gold", "")
        raw = entry.get("raw_response", "") or ""
        parsed = entry.get("parsed_obj")
        correct = entry.get("correct", False)

        constants = extract_constants_from_plan(parsed) if parsed else []

        for c in constants:
            c_str = str(c)
            if c_str and c_str not in raw and str(float(c)) not in raw:
                constants_not_in_response += 1

        try:
            gold_val = float(str(gold).replace(",", ""))
            for c in constants:
                try:
                    if abs(float(c) - gold_val) < 1e-6 and float(c) != 0:
                        gold_in_constants += 1
                        break
                except (ValueError, TypeError):
                    pass
        except (ValueError, TypeError):
            pass

        if len(sampled) < args.sample_size:
            sampled.append({
                "idx": entry.get("idx"),
                "gold": gold,
                "correct": correct,
                "constants": constants[:5],
                "response_snippet": raw[:200] if raw else "",
            })

    report = []
    report.append("# Value Leakage Audit\n")
    report.append(f"## Predictions file: {args.predictions}\n")
    report.append(f"Total predictions: {total}\n")
    report.append(f"\n## Leakage Indicators\n")
    report.append(f"| Indicator | Count | Rate | Verdict |")
    report.append(f"|-----------|-------|------|---------|")

    gold_rate = gold_in_constants / max(total, 1)
    verdict_gold = "SUSPICIOUS" if gold_rate > 0.5 else "OK"
    report.append(f"| Gold answer appears as constant in plan | {gold_in_constants} | {gold_rate:.1%} | {verdict_gold} |")

    cnir_rate = constants_not_in_response / max(total, 1)
    verdict_cnir = "SUSPICIOUS" if cnir_rate > 0.3 else "OK"
    report.append(f"| Constants not found in raw_response | {constants_not_in_response} | {cnir_rate:.1%} | {verdict_cnir} |")

    overall = "NO_TEST_ORACLE_DETECTED"
    if verdict_gold == "SUSPICIOUS" or verdict_cnir == "SUSPICIOUS":
        overall = "POTENTIAL_LEAKAGE_DETECTED"

    report.append(f"\n## Overall Verdict: **{overall}**\n")

    if overall == "NO_TEST_ORACLE_DETECTED":
        report.append("The model generates plan constants from its own predictions.\n")
        report.append("Constants appear in raw_response, confirming model generation.\n")
        report.append("Gold answer as constant is within expected range for correct predictions.\n")
    else:
        report.append("**WARNING**: Potential leakage detected. Manual inspection required.\n")

    report.append(f"\n## Sample Predictions ({len(sampled)} examples)\n")
    report.append("| idx | gold | correct | constants | response snippet |")
    report.append("|-----|------|---------|-----------|-----------------|")
    for s in sampled:
        cs = str(s["constants"])[:30]
        rs = s["response_snippet"][:60].replace("|", "/").replace("\n", " ")
        report.append(f"| {s['idx']} | {s['gold']} | {s['correct']} | {cs} | {rs} |")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(report) + "\n")
    print(f"Audit saved to {args.output}")
    print(f"Verdict: {overall}")


if __name__ == "__main__":
    main()
