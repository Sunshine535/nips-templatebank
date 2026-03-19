# Proposal: TemplateBank++ (Revised v2)

## Thesis
Reasoning tasks contain reusable structure. Explicit template memory plus constrained instantiation can improve quality-cost efficiency vs free-form CoT.

## Falsifiable Questions
1. Does template retrieval + constrained instantiation improve accuracy-cost tradeoff vs CoT/static templates?
2. Is dynamic memory update better than static template banks?
3. Does structure-aware retrieval improve OOD transfer?

## Quantitative Success Criteria
- Primary: at matched cost, `>= +2` absolute accuracy over strongest non-template baseline on at least two datasets.
- Secondary: OOD transfer delta `>= +2` absolute.

## Method
- Extract abstract templates from successful traces.
- Retrieve by semantic and structural signatures.
- Instantiate with variable binding and constraints.
- Optional verifier repair.
- Memory promote/prune with staleness decay.

## What Was Unreasonable Before and Is Corrected
- Leakage risk not controlled -> strict split hygiene added.
- Static-memory assumption -> dynamic memory update protocol added.
- Cost ignored -> matched-cost evaluation made mandatory.

## Current Gap
- Pilot static/dynamic memory experiment exists (`run_templatebank_pilot.py`).
- Full template extraction/instantiation pipeline with leakage-safe splits is still pending.
