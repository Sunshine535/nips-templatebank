# Value Leakage Audit

## Predictions file: results/gift_ablation/gift_no_explicit_refs_oracle_values/seed42/predictions.jsonl

Total predictions: 200


## Leakage Indicators

| Indicator | Count | Rate | Verdict |
|-----------|-------|------|---------|
| Gold answer appears as constant in plan | 58 | 29.0% | OK |
| Constants not found in raw_response | 0 | 0.0% | OK |

## Overall Verdict: **NO_TEST_ORACLE_DETECTED**

The model generates plan constants from its own predictions.

Constants appear in raw_response, confirming model generation.

Gold answer as constant is within expected range for correct predictions.


## Sample Predictions (20 examples)

| idx | gold | correct | constants | response snippet |
|-----|------|---------|-----------|-----------------|
| 0 | 18 | True | [16, 3, 13.0, 4, 9.0] | {"calls": [{"call_id": "c0", "sub_id": "P002", "bindings": { |
| 1 | 3 | True | [2, 2, 1.0, 3.0] | {"calls": [{"call_id": "c0", "sub_id": "P006", "bindings": { |
| 2 | 70000 | False | [80000, 50000, 50000, 150.0, 7 | {"calls": [{"call_id": "c0", "sub_id": "P007", "bindings": { |
| 3 | 540 | True | [3, 3, 9.0, 60, 540.0] | {"calls": [{"call_id": "c0", "sub_id": "P000", "bindings": { |
| 4 | 20 | True | [15, 25, 3, 20, 60.0] | {"calls": [{"call_id": "c0", "sub_id": "P001", "bindings": { |
| 5 | 64 | False | [5, 60, 16, 5, 3.0] | {"calls": [{"call_id": "c0", "sub_id": "P007", "bindings": { |
| 6 | 260 | False | [4, 20, 4, 80.0, 320.0] | {"calls": [{"call_id": "c0", "sub_id": "P000", "bindings": { |
| 7 | 160 | False | [] | {"calls": [{"call_id": "c0", "sub_id": "P007", "bindings": { |
| 8 | 45 | False | [3, 60, 2, 30, 80] | {"calls": [{"call_id": "c0", "sub_id": "P000", "bindings": { |
| 9 | 460 | False | [10.0, 40, 10.0, 1.2, 12.0] | {"calls": [{"call_id": "c0", "sub_id": "P000", "bindings": { |
| 10 | 366 | True | [60, 3, 180.0, 30, 60] | {"calls": [{"call_id": "c0", "sub_id": "P000", "bindings": { |
| 11 | 694 | True | [3, 68, 2, 80, 6] | {"calls": [{"call_id": "c0", "sub_id": "P000", "bindings": { |
| 12 | 13 | False | [7, 1.5, 10.5, 3, 90] | {"calls": [{"call_id": "c0", "sub_id": "P000", "bindings": { |
| 13 | 18 | False | [5, 2, 7.0, 2, 9.0] | {"calls": [{"call_id": "c0", "sub_id": "P002", "bindings": { |
| 14 | 60 | False | [20, 20, 20, 4.0, 16.0] | {"calls": [{"call_id": "c0", "sub_id": "P007", "bindings": { |
| 15 | 125 | False | [5000, 2.5, 8000.0, 1.2, 125.0 | {"calls": [{"call_id": "c0", "sub_id": "P007", "bindings": { |
| 16 | 230 | True | [80, 150, 230.0] | {"calls": [{"call_id": "c0", "sub_id": "P001", "bindings": { |
| 17 | 57500 | True | [20, 50, 35, 30, 50] | {"calls": [{"call_id": "c0", "sub_id": "P000", "bindings": { |
| 18 | 7 | True | [3, 4, 12.0, 7, 84.0] | {"calls": [{"call_id": "c0", "sub_id": "P000", "bindings": { |
| 19 | 6 | False | [] | {"calls": [{"call_id": "c0", "sub_id": "P002", "bindings": { |
