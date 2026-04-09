---
type: paper
node_id: paper:lilo2024_library_learning
title: "LILO: Learning Interpretable Libraries by Compressing and Documenting Code"
authors: ["Gabriel Grand", "et al."]
year: 2024
venue: ICLR 2024
external_ids:
  arxiv: null
tags: [library-learning, program-compression, stitch, neurosymbolic]
relevance: core
origin_skill: research-pipeline
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# LLM-guided synthesis + Stitch compression + auto-documentation learns interpretable reusable function libraries

## Problem / Gap
Program synthesis lacks reusable abstractions; DreamCoder too slow.

## Method
1. LLM-guided program synthesis
2. Stitch symbolic compression for library extraction
3. Auto-documentation: infer names/docstrings from usage context

## Key Results
- Interpretable libraries on toy domains (string editing, scene reasoning, logo)
- More efficient than DreamCoder

## Limitations / Failure Modes
- TOY DOMAINS ONLY — not tested on real math reasoning
- No verification of library functions
- No transfer to different models
- Library is learned offline and frozen

## Reusable Ingredients
- Stitch compression algorithm
- MDL-based library scoring (similar to our approach)
- Auto-documentation idea

## Connections
*(auto-generated from graph/edges.jsonl)*

## Relevance to This Project
Our work is the "LILO for math reasoning" upgrade: verified execution, real-world benchmarks, cross-model transfer. LILO validates the MDL approach but doesn't scale.
