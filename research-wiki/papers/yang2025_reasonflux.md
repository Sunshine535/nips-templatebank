---
type: paper
node_id: paper:yang2025_reasonflux
title: "ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates"
authors: ["Ling Yang", "et al."]
year: 2025
venue: NeurIPS 2025 Spotlight
external_ids:
  arxiv: "2502.06772"
tags: [template-reasoning, hierarchical-rl, inference-scaling, math-reasoning]
relevance: core
origin_skill: research-pipeline
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# ~500 hand-designed thought templates + hierarchical RL enable 91.2% MATH and 56.7% AIME, surpassing o1-preview

## Problem / Gap
LLMs lack structured reasoning strategies. CoT is flat and doesn't scale to complex problems.

## Method
1. Library of ~500 hand-designed high-level thought templates
2. Hierarchical RL optimizes template trajectory planning (not long CoTs)
3. Inference-time scaling: adaptively scale thought templates at inference

## Key Results
- 91.2% MATH (surpasses o1-preview by 6.7%)
- 56.7% AIME (surpasses o1-preview by 27%, DeepSeek-V3 by 45%)
- 63.3% OlympiadBench
- 40% fewer computational steps than MCTS and Best-of-N

## Assumptions
- Templates are hand-designed and fixed
- Template quality is the bottleneck

## Limitations / Failure Modes
- Templates are NOT automatically mined — requires expert design
- No cross-model transfer (templates tied to one model)
- No compositional generalization evaluation (no MCD splits)
- No execution verification of template outputs
- Library is FROZEN — no self-evolution

## Reusable Ingredients
- Hierarchical RL for template trajectory optimization
- Inference-time template scaling mechanism

## Open Questions
- Can templates be automatically mined instead of hand-designed?
- Does the approach generalize compositionally?

## Claims
*(none linked yet)*

## Connections
*(auto-generated from graph/edges.jsonl)*

## Relevance to This Project
**Most direct competitor.** Our approach differs on: (1) automatic mining vs hand-design, (2) verified execution, (3) cross-model transfer, (4) MCD evaluation. But ReasonFlux's NeurIPS Spotlight status means we MUST differentiate strongly.
