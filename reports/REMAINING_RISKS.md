# Remaining Risks

| Risk | Severity | Impact | Mitigation |
|------|----------|--------|------------|
| GIFT coverage 20.2% (below 30%) | HIGH | Insufficient training data for GIFT to compete | Need finer subroutine mining or step-level primitives |
| Library mining too coarse | HIGH | Only 141/697 programs match any subroutine faithfully | Recompute library with semantic interfaces, not just op signature |
| Zero two-call dataflow plans | HIGH | No true composition in GIFT data | Need subroutines that compose (shorter, complementary) |
| Only GSM8K evaluated | HIGH | No MATH results, paper needs multi-dataset | Extract MATH programs, evaluate on MATH test |
| Only seed 42 | HIGH | Results may be seed-dependent | Run seeds 42, 123, 456 with CI |
| No official baselines | HIGH | Cannot claim superiority without PAL, BoT, Faithful CoT | Implement official baselines at matched compute |
| Flat SFT overfits (84% → 29.5%) | MEDIUM | Need more training data or regularization | Extract more programs, data augmentation |
| GRPO no improvement | MEDIUM | Execution reward saturated on small data | Need larger dataset or Type-Local credit assignment |
| Eval fallback masking | MEDIUM | method_accuracy not separated from fallback | Task 7 (eval reliability fix) not yet applied |
| Training reproducibility | MEDIUM | Missing seed control, checkpoint manifests | Task 8 (training fix) not yet applied |
| MCD compounds from adjacency | MEDIUM | Split doesn't test real composition | Task 6 (MCD rewrite) not yet applied |
| Placeholder citations in PAPERS.md | LOW | Academic integrity issue | Mark as unverified or remove |
| Template algebra dead code | LOW | Confusion risk | Archived in manifest |
