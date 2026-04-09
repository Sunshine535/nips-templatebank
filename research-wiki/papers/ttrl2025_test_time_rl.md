---
type: paper
node_id: paper:ttrl2025_test_time_rl
title: "TTRL: Test-Time Reinforcement Learning"
authors: ["et al."]
year: 2025
venue: arXiv (Apr 2025)
external_ids:
  arxiv: "2504.16084"
tags: [test-time-rl, self-evolution, pseudo-labels, reasoning]
relevance: related
origin_skill: research-pipeline
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# Test-time RL uses majority-vote pseudo-labels to continue training during inference, improving reasoning without ground truth

## Problem / Gap
Models can't improve at test time without labeled data.

## Method
- Generate multiple rollouts at test time
- Use majority vote as pseudo-label
- Continue RL training with these pseudo-labels
- No ground-truth needed

## Key Results
- Improves reasoning on unseen problems
- Works without any labeled test data

## Limitations / Failure Modes
- Majority vote can collapse to wrong consensus (addressed by T3RL)
- No structured abstraction concept

## Reusable Ingredients
- Test-time RL framework
- Pseudo-label generation mechanism

## Connections
*(auto-generated from graph/edges.jsonl)*

## Relevance to This Project
Foundation for test-time library evolution. T3RL improves on TTRL with verification. Our SEVAL would add library structure on top.
