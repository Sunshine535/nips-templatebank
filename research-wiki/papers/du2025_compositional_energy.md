---
type: paper
node_id: paper:du2025_compositional_energy
title: "Generalizable Reasoning through Compositional Energy Minimization"
authors: ["Yilun Du", "et al."]
year: 2025
venue: NeurIPS 2025 Spotlight
external_ids:
  arxiv: "2510.20607"
tags: [energy-based, compositional-reasoning, generalization, parallel-energy-minimization]
relevance: core
origin_skill: research-pipeline
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# Composing energy landscapes of subproblems enables generalization to larger/harder problems than training distribution

## Problem / Gap
Models fail to generalize to problems more complex than training distribution.

## Method
1. Learn energy landscapes over solution spaces of small subproblems
2. At test time: compose energy functions of multiple subproblems → global landscape
3. Parallel Energy Minimization (PEM): particle-based optimization over composed energies
4. Additional constraints can be incorporated at inference

## Key Results
- Outperforms domain-specific SOTA on N-Queens, 3-SAT, Graph Coloring
- Generalizes to larger/more complex problems than seen during training
- NeurIPS 2025 Spotlight

## Limitations / Failure Modes
- Only tested on combinatorial problems (N-Queens, SAT, coloring)
- NOT tested on mathematical reasoning
- Energy landscape learning may not scale to high-dimensional math spaces

## Reusable Ingredients
- Compositional energy function design
- PEM optimization strategy
- Subproblem decomposition framework

## Connections
*(auto-generated from graph/edges.jsonl)*

## Relevance to This Project
**Gap G2 opportunity.** Applying this to math reasoning would be novel. Our subroutines could define subproblem energy landscapes; composition = energy product. Could be the theoretical backbone of SEVAL.
