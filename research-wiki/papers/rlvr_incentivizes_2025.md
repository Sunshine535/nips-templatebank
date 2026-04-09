---
type: paper
node_id: paper:rlvr_incentivizes_2025
title: "Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs"
authors: ["et al."]
year: 2025
venue: arXiv (Oct 2025)
external_ids:
  arxiv: "2506.14245"
tags: [rlvr, reasoning-generalization, grpo, capability-expansion]
relevance: core
origin_skill: research-pipeline
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# RLVR extends reasoning boundary for math/code and generalizes correct reasoning, emerging early in training

## Problem / Gap
Does RLVR actually expand reasoning capability or just optimize existing?

## Method
- CoT-Pass@K metric to measure reasoning generalization
- GRPO training on math/code benchmarks
- Analysis of when reasoning capability emerges

## Key Results
- RLVR CAN extend reasoning boundary (both math and code)
- Enhanced capability emerges EARLY in training
- Generalizes smoothly to higher K values
- But debate continues: base models may surpass RL at large pass@k

## Limitations / Failure Modes
- Only studied flat CoT reasoning
- NOT studied for compositional/structured reasoning
- Open question remains for library-based settings

## Reusable Ingredients
- CoT-Pass@K evaluation metric
- Evidence that RLVR works for reasoning expansion

## Connections
*(auto-generated from graph/edges.jsonl)*

## Relevance to This Project
**Gap G5 directly.** If we show RLVR expands reasoning in COMPOSITIONAL settings (with library), this would be a major finding. The flat-CoT evidence exists; the compositional evidence is missing.
