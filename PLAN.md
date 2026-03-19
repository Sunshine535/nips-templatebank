# Plan: TemplateBank++ (Stage-Gate v2)

## Gate 0: Template Corpus Integrity (Partial)
- [x] Added runnable static/dynamic template-memory pilot (`run_templatebank_pilot.py`).
- [ ] Build full trace corpus and extraction pipeline with leakage checks.
- Go criterion: zero known leakage violations.

## Gate 1: Retrieval Quality (Pilot proxy done, full pending)
- [x] Coarse lexical template retrieval pilot and reuse-rate reporting.
- [ ] Implement semantic+structural retrieval on abstract templates.
- Go criterion: retrieval quality significantly better than semantic-only baseline.

## Gate 2: Instantiation and Repair (Pending)
- [ ] Add constrained instantiation and verifier-guided patching.
- Go criterion: matched-cost accuracy gain `>= +2` absolute over best baseline.

## Gate 3: Dynamic Memory
- Implement promote/prune with staleness decay.
- Go criterion: dynamic memory beats static memory on long-horizon evaluation.

## Gate 4: Paper Package
- Main quality-cost tables, transfer matrix, failure taxonomy, artifacts.

## Kill Criteria
- If template method cannot beat CoT under any cost regime, pivot to negative-result study of reasoning artifact reuse.
