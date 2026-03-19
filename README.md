# TemplateBank++: Dynamic Structural Memory for Cost-Efficient Reasoning

## Overview

This project proposes a **structured reasoning template memory** system. Instead of generating chain-of-thought from scratch for every problem, TemplateBank++ retrieves and instantiates reusable high-level reasoning templates from a dynamic memory bank—reducing token cost while maintaining or improving accuracy.

**Target venue:** NeurIPS 2026

**Status:** ~30% complete (pilot stage)

## Research Questions

1. Does template retrieval + constrained instantiation improve accuracy-cost tradeoff vs CoT/static templates?
2. Is dynamic memory update better than static template banks?
3. Does structure-aware retrieval improve OOD transfer?

## Core Idea

```
Input Problem
      │
┌─────┴──────────────┐
│ Semantic + Structural │
│ Similarity Search     │
└─────┬──────────────┘
      │
      ▼
┌─────────────────┐     ┌───────────────────┐
│ Template Bank   │ ──→ │ Top-K Templates   │
│ (dynamic)       │     │ with step graphs  │
└─────────────────┘     └─────┬─────────────┘
                              │
                        ┌─────┴─────────┐
                        │ Constrained   │
                        │ Instantiation │
                        │ (variable     │
                        │  binding)     │
                        └─────┬─────────┘
                              │
                        ┌─────┴─────────┐
                        │ Optional      │
                        │ Verifier      │
                        │ Repair        │
                        └─────┬─────────┘
                              │
                        Final Answer
                              │
                   ┌──────────┴──────────┐
                   │ Memory Manager      │
                   │ promote/prune       │
                   │ staleness decay     │
                   └─────────────────────┘
```

## Method

1. **Template Extraction:** Abstract successful reasoning traces into step-graph templates
2. **Retrieval:** Semantic + structural similarity matching
3. **Constrained Instantiation:** Variable binding with type/domain constraints
4. **Verifier Repair:** Optional verification and correction pass
5. **Dynamic Memory:** Promote successful templates, prune stale ones, decay unused entries

## Current Results (Pilot)

Static/dynamic memory pilot on GSM8K-derived traces. Both below fixed256 quality but better utility than fixed64/fixed128.

## Repository Structure

```
nips-templatebank/
├── README.md              # This file
├── PROPOSAL.md            # Falsifiable thesis and success criteria
├── PLAN.md                # Stage-gate execution plan
├── EXPERIMENTS.md          # Evaluation protocol and results
├── PAPERS.md              # Core references with URLs
├── README_RUN.md          # Runbook
├── environment.yml        # Conda environment spec
├── scripts/
│   └── run_templatebank_pilot.py   # Pilot experiment script
└── results/
    └── templatebank_pilot_20260227_150036.json
```

## Quick Start

```bash
conda env create -f environment.yml
conda activate nips_templatebank
python scripts/run_templatebank_pilot.py
```

## Quantitative Success Criteria

- **Primary:** At matched cost, >= +2 absolute accuracy over strongest non-template baseline on >= 2 datasets
- **Secondary:** OOD transfer delta >= +2 absolute

## Key References

- Buffer of Thoughts (NeurIPS 2024)
- DeAR (NeurIPS 2024)
- Chain of Preference Optimization (NeurIPS 2024)
- Self-Refine (NeurIPS 2023)
- Reasoning Boundary Framework (NeurIPS 2024)

See [PAPERS.md](PAPERS.md) for full list with direct URLs.

## Remaining Work

1. Build template extraction pipeline from real reasoning traces
2. Implement structural similarity retrieval
3. Build constrained instantiation engine
4. Implement dynamic memory manager with decay/promotion
5. Evaluate on GSM8K, MATH, BBH, StrategyQA with leakage-safe splits

## License

Research code for academic use.
