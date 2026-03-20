# Papers: Template Algebra

Core references for the formal algebraic operations on reasoning templates project,
organized by relevance category.

---

## Category A: Template / Reasoning Reuse (Direct Competitors)

### ReasoningBank: Teaching LLMs to Reason with Memory
- **Authors:** Various
- **Venue:** arXiv 2025
- **URL:** https://arxiv.org/abs/2501.XXXXX
- **Key idea:** Store successful reasoning traces in a memory bank; retrieve most similar trace for new problems via embedding similarity
- **Relevance:** Primary competitor for template-based reasoning improvement
- **Limitation for us:** Pure retrieval — cannot compose fragments from different traces. If no single stored trace matches the new structure, it fails.
- **Our advantage:** We decompose traces into composable templates with algebraic operations

### Metacognitive Reuse: Compressing Past Experience for Efficient LLM Reasoning
- **Authors:** Various
- **Venue:** arXiv 2025
- **URL:** https://arxiv.org/abs/2502.XXXXX
- **Key idea:** Compress recurring reasoning behaviors into reusable compressed token sequences, achieving 46% token reduction
- **Relevance:** Primary competitor for token-efficient reasoning
- **Limitation for us:** Compressed tokens are opaque — cannot be decomposed, composed, or transferred. Lossy compression.
- **Our advantage:** Templates are structured, typed, and composable. Formal algebra over them.

### Framework of Thoughts: Structured Reasoning with Predefined Frameworks
- **Authors:** Various
- **Venue:** arXiv 2025
- **URL:** https://arxiv.org/abs/2504.XXXXX
- **Key idea:** Define structured reasoning frameworks (break-into-subproblems, analogical, etc.) and select appropriate framework per problem
- **Relevance:** Related structured reasoning approach
- **Limitation for us:** Frameworks are manually designed, static, not learned or composed
- **Our advantage:** Templates are automatically extracted and algebraically composable

---

## Category B: Thought / Template Retrieval and Augmentation

### Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models
- **Authors:** Yang et al.
- **Venue:** NeurIPS 2024
- **URL:** https://arxiv.org/abs/2406.04271
- **Key idea:** Maintain a buffer of high-level thought templates; retrieve and instantiate for new problems
- **Relevance:** Closely related template retrieval approach
- **Limitation:** Retrieval-only, no composition; templates are informal (natural language descriptions)
- **Our advantage:** Formal template structure with typed slots; algebraic composition

### Retrieval-Augmented Thought Process (RATP)
- **Authors:** Various
- **Venue:** arXiv 2024
- **URL:** https://arxiv.org/abs/2403.XXXXX
- **Key idea:** Retrieve relevant reasoning examples to augment chain-of-thought
- **Relevance:** Retrieval-augmented reasoning baseline
- **Limitation:** Retrieves full examples, not composable template fragments

### Self-Notes: Does LLM Benefit from Taking Notes During Reasoning?
- **Authors:** Various
- **Venue:** arXiv 2024
- **Key idea:** LLM generates intermediate notes during reasoning to organize thought
- **Relevance:** Intermediate reasoning structure, but unformalized

---

## Category C: Chain-of-Thought Foundations

### Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- **Authors:** Wei et al.
- **Venue:** NeurIPS 2022
- **URL:** https://arxiv.org/abs/2201.11903
- **Key idea:** Include step-by-step reasoning demonstrations in prompt to elicit reasoning
- **Relevance:** Foundation of CoT reasoning; our baseline

### Tree of Thoughts: Deliberate Problem Solving with LLMs
- **Authors:** Yao et al.
- **Venue:** NeurIPS 2023
- **URL:** https://arxiv.org/abs/2305.10601
- **Key idea:** Explore multiple reasoning paths as a tree; evaluate and prune
- **Relevance:** Structured reasoning exploration; our templates encode proven paths

### Graph of Thoughts: Solving Elaborate Problems with Large Language Models
- **Authors:** Besta et al.
- **Venue:** AAAI 2024
- **URL:** https://arxiv.org/abs/2308.09687
- **Key idea:** Represent reasoning as a graph allowing merging and refinement of thoughts
- **Relevance:** Graph structure for reasoning; our templates are also graphs (step DAGs)

---

## Category D: Program Synthesis and Formal Methods for Reasoning

### PAL: Program-Aided Language Models
- **Authors:** Gao et al.
- **Venue:** ICML 2023
- **URL:** https://arxiv.org/abs/2211.10435
- **Key idea:** Generate Python programs instead of natural language reasoning steps
- **Relevance:** Reasoning via program generation; our template programs are a higher-level abstraction

### Faithful Chain-of-Thought Reasoning
- **Authors:** Lyu et al.
- **Venue:** IJCNLP-AACL 2023
- **URL:** https://arxiv.org/abs/2301.13379
- **Key idea:** Generate reasoning in a symbolic language with verified execution
- **Relevance:** Formal reasoning languages; our template DSL is similar in spirit

### Planning with Large Language Models via Corrective Re-prompting
- **Authors:** Various
- **Venue:** arXiv 2024
- **Key idea:** LLMs plan by decomposing into subgoals with corrective feedback
- **Relevance:** Decomposition-based reasoning; our compiler does explicit template decomposition

---

## Category E: Mathematical Reasoning Benchmarks

### GSM8K: Training Verifiers to Solve Math Word Problems
- **Authors:** Cobbe et al.
- **Venue:** arXiv 2021
- **URL:** https://arxiv.org/abs/2110.14168
- **Key idea:** 8.5K grade school math word problems with step-by-step solutions
- **Relevance:** Primary source of reasoning traces for template extraction

### MATH: Measuring Mathematical Problem Solving with the MATH Dataset
- **Authors:** Hendrycks et al.
- **Venue:** NeurIPS 2021
- **URL:** https://arxiv.org/abs/2103.03874
- **Key idea:** 12.5K competition math problems across 7 subjects
- **Relevance:** Harder reasoning benchmark; source of diverse templates

### AIME and AMC Competition Problems
- **URL:** https://artofproblemsolving.com/wiki/
- **Relevance:** Out-of-distribution evaluation on competition math

---

## Category F: Model Training and Infrastructure

### Qwen2.5 / Qwen3.5 Technical Reports
- **Authors:** Alibaba Qwen Team
- **URL:** https://arxiv.org/abs/2412.15115
- **Relevance:** Base models for template extraction (27B) and compiler training (9B)

### DeepSpeed ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
- **Authors:** Rajbhandari et al.
- **Venue:** SC 2020
- **URL:** https://arxiv.org/abs/1910.02054
- **Relevance:** Training infrastructure for fine-tuning large models

### vLLM: Efficient Memory Management for Serving Large Language Models
- **Authors:** Kwon et al.
- **Venue:** SOSP 2023
- **URL:** https://arxiv.org/abs/2309.06180
- **Relevance:** Fast inference serving for template extraction with Qwen3.5-27B

---

## Category G: Compositional Generalization (Evaluation Framework)

### SCAN: Compositional Generalization Benchmarks
- **Authors:** Lake & Baroni
- **Venue:** ICML 2018
- **URL:** https://arxiv.org/abs/1711.00350
- **Key idea:** Systematically test compositional generalization in seq2seq models
- **Relevance:** Inspiration for our compositional test set design

### COGS: Compositional Generalization Challenge
- **Authors:** Kim & Linzen
- **Venue:** EMNLP 2020
- **URL:** https://arxiv.org/abs/2010.05465
- **Key idea:** Structural generalization tests for semantic parsing
- **Relevance:** Methodology for designing held-out composition tests

---

## Reading Priority

| Priority | Papers | Reason |
|----------|--------|--------|
| P0 (must read) | ReasoningBank, Metacognitive Reuse, Buffer of Thoughts | Direct competitors |
| P0 (must read) | CoT, GSM8K, MATH | Foundations and data sources |
| P1 (should read) | Framework of Thoughts, PAL, Tree of Thoughts | Related structured reasoning |
| P1 (should read) | SCAN, COGS | Compositional generalization evaluation methodology |
| P2 (nice to have) | Graph of Thoughts, Self-Notes, RATP | Broader reasoning augmentation context |
