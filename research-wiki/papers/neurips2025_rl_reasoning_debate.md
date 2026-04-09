---
type: paper
node_id: paper:neurips2025_rl_reasoning_debate
title: "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"
authors: ["et al."]
year: 2025
venue: NeurIPS 2025 Runner-Up
external_ids:
  arxiv: null
tags: [rlvr, reasoning-boundary, capability-expansion, base-model]
relevance: core
origin_skill: research-pipeline
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# Systematic exploration shows RL fine-tuning enhances sampling efficiency without clearly expanding reasoning capacity beyond base model

## Problem / Gap
Does RL (GRPO etc.) create NEW reasoning ability or just optimize existing?

## Method
- Systematic evaluation across model families
- Math and programming benchmarks
- Compare base vs RL-trained at various pass@k

## Key Results
- RL enhances sampling efficiency (better pass@1)
- At large pass@k, base models may match or surpass RL
- Suggests RL OPTIMIZES rather than EXPANDS

## Limitations / Failure Modes
- Only studied FLAT CoT reasoning
- NOT studied for compositional/structured/library-based reasoning
- The question remains OPEN for structured settings

## Reusable Ingredients
- Evaluation methodology for capability expansion
- pass@k comparison framework

## Connections
*(auto-generated from graph/edges.jsonl)*

## Relevance to This Project
**THE question our paper can answer.** If RLVR + library composition shows capability EXPANSION (not just optimization) on MCD-hard, this would be a headline result. Gap G5 directly.
