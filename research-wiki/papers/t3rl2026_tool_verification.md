---
type: paper
node_id: paper:t3rl2026_tool_verification
title: "Tool Verification for Test-Time Reinforcement Learning"
authors: ["et al."]
year: 2026
venue: arXiv (Mar 2026)
external_ids:
  arxiv: "2603.02203"
tags: [test-time-rl, tool-verification, code-execution, math-reasoning, rlvr]
relevance: core
origin_skill: research-pipeline
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# Tool verification at test time via code execution achieves +31.6% relative improvement on AIME2024

## Problem / Gap
Test-time RL (TTRL) uses majority vote pseudo-labels which can collapse to wrong modes.

## Method
1. Verifier LLM: extracts answer, transforms rollout to Python code, judges validity
2. Verification Tool: code interpreter executes Python to validate reasoning trace
3. Verification Weight: scalar factor upweights verified rollouts in majority vote
4. Verification-aware voting produces reliable pseudo-labels for TTRL

## Key Results
- +31.6% relative improvement on AIME2024 (hardest benchmark)
- Bigger gains on harder benchmarks
- Prevents incorrect mode collapse

## Limitations / Failure Modes
- Only verifies existing rollouts, doesn't BUILD new tools
- Verification limited to code-translatable reasoning
- No library/abstraction concept

## Reusable Ingredients
- Tool-verification-as-reward paradigm
- Verification-weighted voting mechanism
- Code-based reasoning trace validation

## Connections
*(auto-generated from graph/edges.jsonl)*

## Relevance to This Project
**Critical ingredient.** T3RL's verification mechanism can be directly applied to verify library compositions. Combining T3RL verification + our library composition = Gap G3 addressed. Published March 2026 — extremely timely.
