#!/usr/bin/env bash
# Run after extraction completes to check go/no-go gate
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "$0")/../.venv/bin/activate"

TEMPLATE_DIR="results/templates_full"
VERIFIED_DIR="results/templates_verified"
CONFIG="configs/template_config.yaml"

mkdir -p "$VERIFIED_DIR" logs

echo "================================================================"
echo "  GO/NO-GO GATE DIAGNOSTICS"
echo "  $(date)"
echo "================================================================"

# Step 1: Verify programs
echo "========== Step 1: Answer Verification =========="
python scripts/verify_programs.py \
    --input "$TEMPLATE_DIR/all_programs.json" \
    --output "$VERIFIED_DIR/all_programs.json" \
    --re_execute \
    2>&1 | tee logs/gate_verify.log

VERIFIED_COUNT=$(python3 -c "import json; print(len(json.load(open('$VERIFIED_DIR/all_programs.json'))))")
echo "Verified programs: $VERIFIED_COUNT"

# Step 2: Build MCD split
echo "========== Step 2: MCD Split =========="
python scripts/build_mcd_split.py \
    --config "$CONFIG" \
    --programs "$VERIFIED_DIR/all_programs.json" \
    --output results/mcd_split_v2.json \
    --num_trials 500 \
    2>&1 | tee logs/gate_mcd.log

# Step 3: Post-split rebuild (library + plans from train)
echo "========== Step 3: Post-split Rebuild =========="
python scripts/extract_templates.py \
    --config "$CONFIG" \
    --output_dir "$VERIFIED_DIR" \
    --post_split \
    --split_path results/mcd_split_v2.json \
    --programs_path "$VERIFIED_DIR/all_programs.json" \
    2>&1 | tee logs/gate_rebuild.log

# Step 4: Gate diagnostics
echo "========== GATE DIAGNOSTICS =========="
python3 -c "
import json

# Load results
with open('$VERIFIED_DIR/all_programs.json') as f:
    progs = json.load(f)
with open('$VERIFIED_DIR/subroutine_library.json') as f:
    lib = json.load(f)
with open('$VERIFIED_DIR/extraction_meta.json') as f:
    meta = json.load(f)
with open('results/mcd_split_v2.json') as f:
    split = json.load(f)

# Compute metrics
lib_size = len(lib)
total_plans = meta.get('plans', 0)
faithful = meta.get('faithful_plans', total_plans)
call_dist = meta.get('call_distribution', {})
multi_call = sum(v for k, v in call_dist.items() if int(k) >= 2)
multi_call_pct = 100 * multi_call / max(total_plans, 1)
faithful_pct = 100 * faithful / max(total_plans, 1) if 'faithful_plans' in meta else -1

# Support histogram
supports = [lib[sid]['support'] for sid in lib]
supports.sort(reverse=True)

stats = split.get('stats', {})

print('=' * 60)
print('  GO/NO-GO GATE RESULTS')
print('=' * 60)
print(f'  Verified programs:     {len(progs)}')
print(f'  Library size:          {lib_size} (GATE: >=16)')
print(f'  Support histogram:     {supports[:10]}...')
print(f'  Total plans:           {total_plans}')
print(f'  Faithful plans:        {faithful} ({faithful_pct:.1f}%) (GATE: >=15%)')
print(f'  Multi-call plans:      {multi_call} ({multi_call_pct:.1f}%)')
print(f'  Call distribution:     {call_dist}')
print(f'  MCD compound div:      {stats.get(\"compound_divergence\", \"?\")}')
print(f'  MCD atom TVD:          {stats.get(\"atom_tvd\", \"?\")}')
print(f'  Train/Dev/Test:        {stats.get(\"n_train\",\"?\")}/{stats.get(\"n_dev\",\"?\")}/{stats.get(\"n_test\",\"?\")}')
print()

# Gate decision
lib_ok = lib_size >= 16
faith_ok = faithful_pct >= 15 if faithful_pct >= 0 else True
multi_ok = multi_call_pct >= 10

if lib_ok and faith_ok and multi_ok:
    print('  >>> GATE: GO — proceed with training <<<')
elif lib_ok and multi_ok:
    print('  >>> GATE: MARGINAL — proceed cautiously <<<')
else:
    print('  >>> GATE: NO-GO — need more data or different strategy <<<')
    if not lib_ok:
        print(f'    Library too small: {lib_size} < 16')
    if not faith_ok:
        print(f'    Faithfulness too low: {faithful_pct:.1f}% < 15%')
    if not multi_ok:
        print(f'    Too few multi-call plans: {multi_call_pct:.1f}% < 10%')
print('=' * 60)
" 2>&1 | tee logs/gate_results.log

echo "Gate diagnostics complete. Check logs/gate_results.log"
