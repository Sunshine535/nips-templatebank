---
type: paper
node_id: paper:deepseek2025_r1
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
authors: ["DeepSeek-AI"]
year: 2025
venue: Nature 2025
external_ids:
  arxiv: "2501.12948"
tags: [rlvr, grpo, reasoning, reinforcement-learning]
relevance: core
origin_skill: research-pipeline
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# GRPO with verifiable rewards enables reasoning comparable to o1 with dynamic compute allocation

## Problem / Gap
Standard RLHF doesn't specifically optimize for reasoning. Need verifiable reward signals.

## Method
1. Group Relative Policy Optimization (GRPO)
2. Verifiable rewards from deterministic verifiers (no reward model needed)
3. Dynamic computational resource allocation based on problem complexity

## Key Results
- Comparable to o1-1217 across math/code/reasoning
- Published in Nature — highest-impact reasoning paper
- Distilled models work down to 1.5B parameters

## Limitations / Failure Modes
- Flat CoT reasoning — no compositional structure
- No library/abstraction concept
- RLVR debate: may only optimize, not expand capability

## Reusable Ingredients
- GRPO algorithm
- Verifiable reward framework
- Distillation to small models

## Connections
*(auto-generated from graph/edges.jsonl)*

## Relevance to This Project
GRPO is the RL algorithm we should use for library evolution. Verifiable rewards = composition execution success. This is the bridge from frozen library to self-evolving library.
