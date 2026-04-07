# Verified Procedural Abstractions Enable Transferable Compositional Math Reasoning

## Problem Anchor
Question: can a frozen, execution-verified library of typed procedural abstractions mined from Qwen3.5-32B improve small-model compositional generalization on math problems whose atomic ingredients are matched but whose higher-order compositions are novel?

Falsifiable thesis:
1. On GSM8K MCD-hard, a frozen verified library transferred to Qwen3.5-9B beats a matched Qwen3.5-32B-CoT-distilled 9B baseline by >=15 absolute accuracy points.
2. MDL compression ratio is a useful diagnostic of MCD transfer and outpredicts library size, mean trace length, and teacher answer accuracy.
3. Typed MCTS repair recovers >=25% of initially failed plans under matched search budget.

Not claimed:
- not a universal reasoning claim;
- not a "compression law";
- not a search method with privileged labels.

## Method
Core repo objects already exist in src/template_dsl.py: Program, Step, Subroutine, SubroutineLibrary, CompositionPlan, Executor, CompositionExecutor.

Definitions:
- Program p=(s1,...,sT): typed DSL steps.
- Subroutine z: verified program fragment with typed slots, internal steps, output.
- Library L={z1,...,zK}: frozen set of mined subroutines.
- Composition plan pi=(c1,...,cm): ordered calls ci=(zi,bi) binding subroutine slots from environment.
- Executor runs steps; CompositionExecutor runs plans over the library.

Pipeline:
1. Qwen3.5-32B generates candidate DSL programs.
2. Keep only parse-valid, type-valid, step-verified, final-answer-correct traces.
3. Abstract constants into typed slots.
4. Cluster by normalized structure.
5. Score by MDL gain and support.
6. Freeze top-K library from train only.
7. Remap train examples into plans over the library.
8. Train student planners to emit plans, not CoT.

Compression:
- C_flat(x): code length of verified flat teacher program.
- C_L(x): code length of shortest verified library composition.
- Example ratio: CR(x;L)=C_flat(x)/C_L(x).
- Split ratio: CR(D;L)=sum_x C_flat(x) / sum_x C_L(x).
- DSL-token serializer is fixed across all variants.
- MDL_gain(z)=sum_train(C_flat(x)-C_{L+{z}}(x)) - C(z).

Portability:
- same frozen library mined once from Qwen3.5-32B on GSM8K-train;
- transferred to Qwen3.5-9B and a second student;
- second student is Qwen3.5-3B if stable, else Llama-3-8B-Instruct;
- no student-specific reminting in portability runs.

## Formal MCD Split Construction
Unit: verified composition plans, not raw text.

Atoms:
- internal primitive op multiset after inlining;
- slot type signature multiset, e.g. (int,float)->float;
- call arity;
- call count;
- provisional subroutine identity for temporary split-building library.

Compounds:
- inlined primitive-op bigrams and trigrams;
- subroutine bigrams (zi,zj);
- typed-call bigrams sig(zi)->sig(zj);
- solution-graph edges from produced variables to downstream consumers;
- solution-graph depth-aware 3-node motifs.

Split constraints:
- train/dev/test = 60/20/20;
- atom TVD(train,test) <= 0.02;
- unseen test compound ratio >= 0.40 on GSM8K-hard;
- dev sampled from train atom family only;
- only verified plans participate.

Overlap audits, reported for every split:
1. Lexical overlap: normalized n-gram Jaccard, numbers stripped. Success: no more than random split +2 points.
2. Template overlap: normalized flat-program signature overlap. Success: >=40% unseen test templates.
3. Operation-sequence overlap: unseen inlined op bigrams/trigrams. Success: trigram unseen >=35%.
4. Solution-graph overlap: unseen dataflow edges/motifs. Success: edge unseen >=30%, motif unseen >=25%.

## Verification Protocol
Step-level, not final-answer-only.

Stage A, static:
- JSON parse;
- Program.from_dict;
- variable availability per step;
- type coercion;
- no forbidden builtins/operators;
- consistent target typing.

Stage B, dynamic:
1. execute each step under environment from prior steps;
2. record expression, output value, typed environment delta;
3. require no exception;
4. require output type matches target_dtype;
5. require deterministic replay of the entire environment trace.

Additional step-level anti-shortcut check:
- perturb irrelevant environment variables and require unchanged output for each step.

Acceptance targets:
- GSM8K >=1500 verified teacher programs;
- MATH stress subset >=1000 verified teacher programs.

## Baselines And Fairness
Required baselines:
1. 32B-CoT-distilled 9B (primary comparison)
2. flat_inline
3. raw_trace_retrieval
4. uncompressed_program_bank
5. random_library
6. frequency_matched_library
7. retrieval_compose
8. cot_budget
9. search_enabled_flat
10. search_enabled_retrieval

Fairness: same split, same student, same train data, same token caps, same search budget (expansions + forward passes + executor calls), no test-time gold.

Causal compression test: compressed library must beat matched-size uncompressed bank by >=5 and frequency-matched library by >=3 on GSM8K MCD-hard.

## Library Audit Protocol
- support histogram, reuse counts, MDL gain, redundancy rate
- semantic coherence: >=70% of subroutines coherent (4/5 supports share role)
- human-interpretable appendix: top 12 subroutines with examples
- spurious-subroutine filters applied before portability experiments

## Search-Time Repair
Repair actions: substitute, modify binding, insert call, delete call, stop. All type-safe.
Anti-cheating: equalized budget, no oracle correctness on test, hyperparameters frozen from dev.

## Failure Analysis Plan
Label >=150 GSM8K + >=100 MATH failures into 8 categories.
Report: count, percentage, representative example, library/planner/search attribution.

## 8-Page Paper Outline
- Intro (0.75p)
- Setup/Formalism (0.9p)
- Verified Library Mining (1.0p)
- Controlled Splits (0.9p)
- Search-Time Repair (0.7p)
- Experiments (1.55p)
- Results (1.35p)
- Analysis/Limitations (0.85p)

## Exact Acceptance Criteria
1. compose beats CoT-distilled by >=15 on GSM8K MCD-hard (Qwen3.5-9B)
2. frozen library improves second student by >=8 over its CoT-distilled baseline
3. compression ratio is strongest predictor, p<0.05
4. repair recovers >=25% of failed plans under matched budget

If any fails, reframe to narrower supported claim.
