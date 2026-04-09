# Field Gap Map

## Unresolved Gaps

### G1: No self-evolving verified library for math reasoning
All existing library-based reasoning (ReasonFlux, LILO, DreamCoder) use fixed/frozen libraries. No system evolves its abstraction library through verified reward signals during training or inference.

### G2: Energy-based compositional reasoning unexplored for math
Compositional Energy Minimization (NeurIPS 2025 Spotlight) demonstrated on N-Queens/SAT/Graph Coloring only. Zero work applying energy-based composition to mathematical reasoning.

### G3: RLVR + library learning unconnected
RLVR (DeepSeek-R1, GRPO) and library learning (LILO, DreamCoder, TheoryCoder-2) are separate research threads. No work uses composition-verification as verifiable reward to evolve libraries.

### G4: Test-time tool building for math unexplored
T3RL shows tool verification at test time works. But no system builds NEW tools at test time — only verifies existing reasoning traces.

### G5: "Does RL expand reasoning?" unanswered for compositional settings
The RLVR debate (expand vs optimize) has only been studied on flat CoT. No evidence for compositional/library-based reasoning settings.

### G6: Cross-model library transfer lacks RLVR adaptation
Frozen library transfer (our current approach) doesn't adapt to the student. No work combines transfer with student-side RLVR fine-tuning.

### G7: MCD evaluation absent from template/library reasoning
ReasonFlux, Buffer of Thoughts, SELF-DISCOVER — none use MCD splits. Compositional generalization claims are uncontrolled.

### G8: Compositional skill synthesis lacks formal verification
Agentic Proposing (Feb 2026) does compositional skill synthesis but without step-level execution verification. Skills are unverified.

## Addressed Gaps
*(none yet)*
