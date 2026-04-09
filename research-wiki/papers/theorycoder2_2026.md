---
type: paper
node_id: paper:theorycoder2_2026
title: "Learning Abstractions for Hierarchical Planning in Program-Synthesis Agents"
authors: ["et al."]
year: 2026
venue: arXiv (Feb 2026)
external_ids:
  arxiv: "2602.00929"
tags: [abstraction-learning, hierarchical-planning, tbrl, program-synthesis]
relevance: related
origin_skill: research-pipeline
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# Auto-learning reusable abstractions via few-shot LLM synthesis enables more sample-efficient hierarchical planning than prior agents

## Problem / Gap
TBRL agents rely on hand-specified abstractions that don't scale.

## Method
1. LLM in-context learning synthesizes abstractions from experience
2. Abstractions integrated into hierarchical planning
3. Library grows automatically — no manual intervention

## Key Results
- More sample-efficient than WorldCoder and classical planning baselines
- Solves complex tasks baselines fail
- Minimal human prompts required

## Limitations / Failure Modes
- Only tested on BabyAI, Minihack, Sokoban (toy RL domains)
- Not applied to mathematical reasoning
- No formal verification of learned abstractions

## Reusable Ingredients
- Auto-abstraction learning from experience
- Hierarchical planning with growing library

## Connections
*(auto-generated from graph/edges.jsonl)*

## Relevance to This Project
Closest spirit to SEVAL but in RL/planning domain. Validates the idea of auto-growing abstraction libraries. Our contribution: bring this to MATH + add verification + add RLVR.
