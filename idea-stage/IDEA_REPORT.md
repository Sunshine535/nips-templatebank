# Idea Discovery Report

**Direction**: 在 SEVAL 基础上改进到 MATH SOTA (90%+)
**Date**: 2026-04-24
**Constraint**: 必须基于当前方法改进，无完全相同的已有工作

## Literature Landscape

### SOTA Methods (MATH benchmark)
- rStar-Math (90.0%): MCTS + Process Reward Model (PRM) + mutual self-play
- ReasonFlux (91.2%): Hierarchical reasoning skill library + RL-guided planning
- DeepSeek-R1 (79.8%): Pure GRPO + long CoT
- Qwen3 (87.1%): RL + thinking mode

### Key Insight
所有 MATH 90%+ 的方法都使用了 **大规模测试时计算** (test-time compute scaling):
- rStar-Math: MCTS 搜索 (hundreds of rollouts per problem)
- ReasonFlux: hierarchical search over skill combinations
- Best-of-N with verifier: sample N solutions, verify, vote

SEVAL 目前缺少这个关键组件 — 只用单次生成。

### SEVAL 的独特优势（不应放弃的）
1. **Typed DSL**: 结构化搜索空间，比自由文本 CoT 更紧凑
2. **确定性执行验证**: 完美的 reward oracle，不需要训练 PRM
3. **子程序库**: 提供复用和组合推理的基础

## Ranked Ideas

---

### Idea 1: Execution-Verified Process Reward (EVPR)
**核心**: 用程序的确定性执行结果替代训练的 Process Reward Model

**方法**:
1. 模型生成 N 个候选程序 (N=16-64)
2. 每个程序在 typed DSL executor 中执行
3. 执行结果提供 **step-level** 反馈:
   - 哪一步类型错误？哪一步执行失败？最终答案对不对？
4. 用这些 step-level 信号做 **best-of-N 选择** 或 **beam search**
5. GRPO 训练: reward = 程序执行正确性 (binary, no reward hacking)

**为什么新颖**:
- rStar-Math 训练 PRM（需要大量人工标注的 step-level 数据）
- EVPR 不需要 PRM — 执行就是验证。**零标注成本的 process reward**
- 没有 reward hacking（执行结果是确定性的，不可欺骗）

**与已有工作的区别**:
- vs PAL/PoT: 它们只做单次生成 + 执行，不做搜索
- vs rStar-Math: 它们用训练的 PRM，我们用执行验证（零成本、完美信号）
- vs ReasonFlux: 它们的技能库是静态的，我们的可以在 RL 训练中进化
- vs DeepSeek-R1: 它们用纯 CoT，不能提供 step-level 执行反馈

**可行性**: ⭐⭐⭐⭐⭐ (所有组件已存在于代码库)
**新颖性**: ⭐⭐⭐⭐ (execution-as-PRM 的具体实现是新的)
**SOTA 潜力**: ⭐⭐⭐ (取决于 DSL 覆盖率和搜索效率)

---

### Idea 2: Program-Space MCTS with Execution Oracle (ProgMCTS)
**核心**: 在 typed program 空间上做 MCTS，用执行结果做节点评估

**方法**:
1. 搜索树的每个节点 = 部分程序 (partial program with K steps)
2. 扩展: LLM 提议下一步 (next step in DSL)
3. 评估: 执行部分程序，检查中间结果是否合理
4. 回溯: 将执行成功/失败反馈到搜索树
5. 终止: 完整程序执行得到正确答案

**为什么新颖**:
- rStar-Math 做 MCTS over CoT tokens（非结构化搜索空间）
- ProgMCTS 做 MCTS over **typed program steps**（结构化搜索空间）
- 每一步都可以被执行验证（不需要训练 PRM）
- DSL 类型系统自动剪枝无效分支

**风险**: DSL 表达力有限，可能无法覆盖所有 MATH 问题类型
**可行性**: ⭐⭐⭐ (需要实现 partial program execution)
**新颖性**: ⭐⭐⭐⭐⭐ (没有人在 typed program 空间做过 MCTS)
**SOTA 潜力**: ⭐⭐⭐⭐ (如果 DSL 覆盖率够高)

---

### Idea 3: Dual-Mode Reasoning (CoT + Program Mutual Verification)
**核心**: 同时生成 CoT 和 Program，互相验证

**方法**:
1. 对每个问题，模型生成:
   - K 个 CoT 解答 (自然语言推理链)
   - K 个 typed program 解答 (可执行程序)
2. **交叉验证**:
   - 执行所有程序 → 得到程序答案
   - 从 CoT 提取答案
   - 如果 CoT 和 Program 答案一致 → 高置信度
   - 如果不一致 → 用不一致信号触发修复
3. **自适应选择**: 
   - 对简单问题: 直接用程序执行结果
   - 对 DSL 无法覆盖的问题: 用 CoT 结果
   - 对不一致的问题: 生成更多候选，加权投票

**为什么新颖**:
- 没有人做过 CoT 与可执行程序的 **mutual verification**
- 大多数工作只用一种范式 (CoT 或 Program)
- 互相验证可以发现单一范式的错误

**可行性**: ⭐⭐⭐⭐ (实现简单，只需要并行生成两种格式)
**新颖性**: ⭐⭐⭐⭐ (mutual verification 角度是新的)
**SOTA 潜力**: ⭐⭐⭐⭐⭐ (覆盖率 = CoT ∪ Program，比单一方法宽)

---

### Idea 4: Execution-Guided Self-Refinement (ExeRefine)
**核心**: 程序执行失败时，用执行错误信息指导 LLM 修复程序

**方法**:
1. 生成候选程序
2. 执行 → 如果失败，获得详细错误信息:
   - "Step 3: NameError: 'radius' not defined"
   - "Step 5: TypeError: can't multiply str and int"
   - "Execution OK but answer=42, expected=56"
3. 将错误信息反馈给 LLM: "你的程序在第3步失败了，原因是..., 请修复"
4. 重复最多 K=5 次
5. GRPO 训练: reward = 最终修复成功率

**为什么新颖**:
- Self-Debug (Chen et al., 2023) 做过代码修复，但不是 typed DSL，不用 GRPO
- 执行错误信息是 **精确的、结构化的** (比自然语言 feedback 更有信息量)
- 与 RLVR 结合: 用 GRPO 训练模型成为更好的 "程序修复者"

**可行性**: ⭐⭐⭐⭐⭐ (最容易实现)
**新颖性**: ⭐⭐⭐ (self-debug 类似，但 typed DSL + GRPO 是新组合)
**SOTA 潜力**: ⭐⭐⭐ (修复成功率可能有限)

---

### Idea 5: Library-Augmented Retrieval for Program Synthesis (LARPS)
**核心**: 把子程序库用作检索增强的少样本示例，而不是硬组合

**方法**:
1. 挖掘子程序库 (Phase 0, 已有)
2. 对新问题:
   - 检索最相似的子程序作为 few-shot 示例
   - Prompt: "这里有一些相关的程序模板: [examples]. 参考它们为新问题生成程序"
3. 执行验证 + best-of-N
4. GRPO 训练: 模型学会利用检索到的示例

**为什么新颖**:
- 不同于 SEVAL 的硬组合 (call subroutine with bindings)
- 用子程序作为 **soft guidance** 而非 **hard API call**
- 避免了绑定问题 (slot name mismatch)
- 保留了子程序库的知识复用优势

**可行性**: ⭐⭐⭐⭐⭐ (最简单，不需要 composition executor)
**新颖性**: ⭐⭐⭐ (RAG for program synthesis 有一些相关工作)
**SOTA 潜力**: ⭐⭐⭐ (检索质量是瓶颈)

---

## Recommendation

**推荐组合方案: Idea 1 + Idea 3 + Idea 4**

**论文标题方向**: "Execution-Verified Reasoning: When Deterministic Programs Meet Test-Time Search for Mathematical Problem Solving"

**核心贡献**:
1. **Execution as Process Reward**: 用确定性程序执行替代训练的 PRM — 零成本、完美信号、不可欺骗
2. **Dual-Mode Verification**: CoT + Program 互相验证，扩大问题覆盖率
3. **Execution-Guided Refinement**: 用执行错误信息指导程序修复
4. **GRPO Training**: 用执行验证 reward 训练模型生成更好的程序

**与 SEVAL 的关系**: 在 SEVAL 的 typed DSL + execution verification 基础上，添加 test-time search (best-of-N, refinement) 和 dual-mode verification，去掉 problematic 的 library composition 层。

**为什么这个组合可以达到 SOTA**:
- Best-of-N with execution oracle 已被证明有效 (类似 rStar-Math 的搜索但不需要 PRM)
- Dual-mode 扩大覆盖率 (DSL 无法表达的用 CoT 覆盖)
- GRPO 训练提升单次生成质量
- Refinement 提升困难问题的成功率
