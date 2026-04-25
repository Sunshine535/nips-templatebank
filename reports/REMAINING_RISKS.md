# Remaining Risks (Updated R4)

| Risk | Severity | Status | Mitigation |
|------|----------|--------|------------|
| E may have test-time value leakage | HIGH | **Leakage audit pending** | Run check_value_leakage.py; if found, E invalid |
| Only seed 42 | HIGH | Pending | Run seeds 123, 456 after gate |
| Only first 200 GSM8K test samples | MEDIUM | Pending | Full test after multi-seed gate |
| No official baselines (PAL/BoT/FCoT) | HIGH | Not started | Add before paper claims |
| C (symbolic refs) < A (flat) | HIGH | **Confirmed negative** | C is ablation, not main method |
| E "oracle" naming misleading | MEDIUM | Renaming to value_supervised_plan | Config added |
| Stale PAPERS.md placeholders | MEDIUM | Warning added | Remove before paper |
| V-GIFT not yet implemented | HIGH | **Task 3 pending** | Implement ValueAnnotatedDataflowPlan |
| No MATH/BBH evaluation | HIGH | Pending | After GSM8K gates pass |
| Active-binding gate shows no effect | MEDIUM | B2≈C on seed 42 | May help at larger scale |
| Prediction logs not yet committed for all variants | MEDIUM | Pending rerun with --predictions_out | Task 4 |
| Train-split-only library not enforced | HIGH | Current library uses all 697 | Must fix before MCD claims |
