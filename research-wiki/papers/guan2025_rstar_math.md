---
type: paper
node_id: paper:guan2025_rstar_math
title: "rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking"
authors: ["et al."]
year: 2025
venue: ICML 2025
external_ids:
  arxiv: "2501.04519"
tags: [mcts, self-play, process-reward, math-reasoning, small-model]
relevance: core
origin_skill: research-pipeline
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# MCTS self-play with process preference model enables small LLMs to achieve 90% MATH and 8/15 AIME

## Problem / Gap
Small LLMs can't match large model reasoning without distillation.

## Method
1. Code-augmented CoT data synthesis via MCTS rollouts
2. Novel process preference model (PPM) training (avoids naive step-level annotation)
3. Self-evolution recipe: policy SLM + PPM iteratively improve from scratch

## Key Results
- 90% MATH (matches o1 level)
- 8/15 AIME (top 20% of high school students)
- No distillation from superior models needed

## Limitations / Failure Modes
- No reusable library/abstraction concept
- No compositional generalization evaluation
- Self-evolution is slow (multiple rounds)

## Reusable Ingredients
- Process preference model design
- Self-evolution recipe without superior teacher

## Connections
*(auto-generated from graph/edges.jsonl)*

## Relevance to This Project
MCTS approach competitor. Our MCTS repair is a small piece; rStar-Math uses MCTS as the core method. Different paradigm but shows small models CAN master math.
