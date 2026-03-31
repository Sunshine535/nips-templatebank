#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone


WORD_RE = re.compile(r"[a-zA-Z]+")
DEFAULT_INPUT = "results/per_sample_input.csv"
ACTIONS = [64, 128, 256]
ANCHORS = [
    "each",
    "total",
    "left",
    "more",
    "less",
    "times",
    "percent",
    "hour",
    "minute",
    "dollar",
    "cost",
    "sum",
    "difference",
    "average",
]


def to_int(v):
    try:
        return int(float(v))
    except Exception:
        return 0


def to_float(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def utility(row, budget, lambda_cost):
    c = to_int(row.get(f"fixed_{budget}_correct", 0))
    t = to_float(row.get(f"fixed_{budget}_tokens", 0.0))
    return c - lambda_cost * (t / 256.0)


def best_action(row, lambda_cost):
    best_b = ACTIONS[0]
    best_u = utility(row, best_b, lambda_cost)
    for b in ACTIONS[1:]:
        u = utility(row, b, lambda_cost)
        if u > best_u:
            best_u = u
            best_b = b
    return best_b


def extract_template(question):
    q = (question or "").lower()
    hits = [k for k in ANCHORS if k in q]
    if not hits:
        toks = WORD_RE.findall(q)
        hits = toks[:2] if toks else ["misc"]
    return "|".join(sorted(set(hits))[:4])


def make_bank(rows, lambda_cost):
    stats = {}
    for r in rows:
        key = extract_template(r.get("question", ""))
        if key not in stats:
            stats[key] = {64: [0.0, 0], 128: [0.0, 0], 256: [0.0, 0]}
        for b in ACTIONS:
            stats[key][b][0] += utility(r, b, lambda_cost)
            stats[key][b][1] += 1
    bank = {}
    for key, st in stats.items():
        best_b, best_u = 256, -1e18
        for b in ACTIONS:
            cnt = st[b][1]
            avg_u = st[b][0] / cnt if cnt else -1e18
            if avg_u > best_u:
                best_u = avg_u
                best_b = b
        bank[key] = best_b
    return bank, stats


def eval_static(rows, bank, default_action, lambda_cost):
    n = max(1, len(rows))
    reuse = 0
    total_correct = 0
    total_tokens = 0.0
    total_utility = 0.0
    for r in rows:
        key = extract_template(r.get("question", ""))
        if key in bank:
            reuse += 1
        b = bank.get(key, default_action)
        total_correct += to_int(r.get(f"fixed_{b}_correct", 0))
        t = to_float(r.get(f"fixed_{b}_tokens", 0.0))
        total_tokens += t
        total_utility += to_int(r.get(f"fixed_{b}_correct", 0)) - lambda_cost * (t / 256.0)
    return {
        "accuracy": total_correct / n,
        "avg_tokens": total_tokens / n,
        "avg_utility": total_utility / n,
        "reuse_rate": reuse / n,
    }


def eval_dynamic(rows, base_bank, base_stats, default_action, lambda_cost):
    bank = dict(base_bank)
    stats = {k: {b: [v[0], v[1]] for b, v in st.items()} for k, st in base_stats.items()}
    n = max(1, len(rows))
    reuse = 0
    total_correct = 0
    total_tokens = 0.0
    total_utility = 0.0
    for r in sorted(rows, key=lambda x: to_int(x.get("idx", 0))):
        key = extract_template(r.get("question", ""))
        if key in bank:
            reuse += 1
        b = bank.get(key, default_action)
        total_correct += to_int(r.get(f"fixed_{b}_correct", 0))
        t = to_float(r.get(f"fixed_{b}_tokens", 0.0))
        total_tokens += t
        total_utility += to_int(r.get(f"fixed_{b}_correct", 0)) - lambda_cost * (t / 256.0)

        if key not in stats:
            stats[key] = {64: [0.0, 0], 128: [0.0, 0], 256: [0.0, 0]}
        for ab in ACTIONS:
            stats[key][ab][0] += utility(r, ab, lambda_cost)
            stats[key][ab][1] += 1

        best_b, best_u = 256, -1e18
        for ab in ACTIONS:
            s, c = stats[key][ab]
            avg_u = s / c if c else -1e18
            if avg_u > best_u:
                best_u = avg_u
                best_b = ab
        bank[key] = best_b

    return {
        "accuracy": total_correct / n,
        "avg_tokens": total_tokens / n,
        "avg_utility": total_utility / n,
        "reuse_rate": reuse / n,
    }


def fixed_metrics(rows, budget, lambda_cost):
    n = max(1, len(rows))
    c = sum(to_int(r.get(f"fixed_{budget}_correct", 0)) for r in rows)
    t = sum(to_float(r.get(f"fixed_{budget}_tokens", 0.0)) for r in rows)
    return {
        "accuracy": c / n,
        "avg_tokens": t / n,
        "avg_utility": (c / n) - lambda_cost * ((t / n) / 256.0),
    }


def main():
    ap = argparse.ArgumentParser(description="TemplateBank++ pilot with static/dynamic template memory")
    ap.add_argument("--input_csv", type=str, default=DEFAULT_INPUT)
    ap.add_argument("--output_dir", type=str, default="results/pilot")
    ap.add_argument("--lambda_cost", type=float, default=0.15)
    args = ap.parse_args()

    with open(args.input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_csv}")

    train = [r for r in rows if (to_int(r.get("idx", 0)) % 5) != 0]
    test = [r for r in rows if (to_int(r.get("idx", 0)) % 5) == 0]
    if not test:
        test = rows[-max(1, len(rows) // 5) :]
        train = rows[: len(rows) - len(test)]

    default_action = max(
        ACTIONS,
        key=lambda b: sum(utility(r, b, args.lambda_cost) for r in train) / max(1, len(train)),
    )
    bank, stats = make_bank(train, args.lambda_cost)
    static_res = eval_static(test, bank, default_action, args.lambda_cost)
    dynamic_res = eval_dynamic(test, bank, stats, default_action, args.lambda_cost)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, f"templatebank_pilot_{ts}.json")
    result = {
        "meta": {
            "timestamp_utc": ts,
            "input_csv": args.input_csv,
            "train_size": len(train),
            "test_size": len(test),
            "lambda_cost": args.lambda_cost,
            "default_action": default_action,
            "template_count_train": len(bank),
        },
        "static_memory_test": static_res,
        "dynamic_memory_test": dynamic_res,
        "fixed_baselines_test": {
            "fixed64": fixed_metrics(test, 64, args.lambda_cost),
            "fixed128": fixed_metrics(test, 128, args.lambda_cost),
            "fixed256": fixed_metrics(test, 256, args.lambda_cost),
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
