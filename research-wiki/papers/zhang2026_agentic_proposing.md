---
type: paper
node_id: paper:zhang2026_agentic_proposing
title: "Agentic Proposing: Enhancing Large Language Model Reasoning via Compositional Skill Synthesis"
authors: ["et al."]
year: 2026
venue: arXiv (Feb 2026)
external_ids:
  arxiv: "2602.03279"
tags: [compositional-skills, MGPO, agentic, skill-library, math-reasoning, rlvr]
relevance: core
origin_skill: research-pipeline
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# Autonomous compositional skill synthesis via MGPO achieves 91.6% AIME25 with only 11K trajectories from a 30B solver

## Problem / Gap
Training data synthesis for reasoning is ad-hoc and doesn't target the model's frontier.

## Method
1. Skill Acquisition: extract + filter atomic skills from diverse corpora → autonomous skill library
2. Agentic SFT: mimic expert trajectories with reflection, tool execution, dynamic skill pruning
3. Agentic RL (MGPO): Multi-Granularity Policy Optimization with multi-level rewards
4. Agentic-Proposer-4B generates training trajectories across math/code/science

## Key Results
- 91.6% AIME25 (30B solver, 11K trajectories)
- Rivals GPT-5 level performance
- State-of-the-art across math, coding, and science

## Assumptions
- Skill library is built offline from corpora
- MGPO is effective for multi-granularity optimization

## Limitations / Failure Modes
- Skills are NOT execution-verified (no step-level verification)
- No MCD compositional evaluation
- Skill library is not self-evolving at test time
- Heavy reliance on MGPO training infrastructure

## Reusable Ingredients
- MGPO algorithm for multi-granularity reward optimization
- Autonomous skill extraction pipeline
- Skill composition mechanism

## Open Questions
- Can verified skills outperform unverified ones?
- Does MGPO work with library-composition rewards?

## Claims
*(none linked yet)*

## Connections
*(auto-generated from graph/edges.jsonl)*

## Relevance to This Project
**Second most dangerous competitor.** Achieves SOTA with compositional skills. Key differentiator for us: execution verification + MCD evaluation + self-evolution. Their skills are unverified (Gap G8).
