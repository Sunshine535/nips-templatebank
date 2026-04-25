# 0. 仓库可读性判断

我确认仓库可通过 GitHub Web 访问，且能看到公开代码、配置、结果、审查文档和环境文件；但我在本地容器里执行 `git clone` 时遇到 DNS 解析失败，因此**没有本地运行测试、没有复现实验、没有修改代码**。以下结论是基于公开仓库的静态审计、结果文件审查和源码阅读；所有“结果是否真实复现”的判断都不能替代本地可运行复现。

| Item                       |               Found? | Location                                                                      | Notes                                                                                                              |
| -------------------------- | -------------------: | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 仓库可访问                      |                  Yes | GitHub root                                                                   | 公开仓库可读；root 显示 `configs/`, `results/`, `scripts/`, `src/`, `tests/` 等目录。([GitHub][1])                              |
| 完整代码                       |            Partially | `src/`, `scripts/`, `tests/`                                                  | 公开代码可读；但我未能本地 clone，所以未跑静态检查、pytest 或命令复现。                                                                         |
| README                     |                  Yes | `README.md`                                                                   | README 主张“Verified Procedural Abstractions for Compositional Math Reasoning”，并给出抽取、MCD split、训练、评估命令。([GitHub][2]) |
| 论文草稿                       | No clear paper draft | `PROPOSAL.md`, `IDEA_REPORT.md`, `PAPERS.md`                                  | 有 proposal/idea/report，不像完整 NeurIPS paper draft；没有看到 `.tex`/PDF 正文。                                                |
| 训练脚本                       |                  Yes | `scripts/train_template_compiler.py`                                          | README 调用 compose/flat/CoT SFT 训练入口。([GitHub][2])                                                                  |
| 评估脚本                       |                  Yes | `scripts/eval_template_reasoning.py`                                          | README 给出评估命令和 `results/eval_v2` 输出路径。([GitHub][2])                                                                |
| configs                    |                  Yes | `configs/`                                                                    | README 指向 `configs/template_config.yaml`, `configs/pilot_config.yaml`。                                             |
| 日志和结果                      |      Yes, incomplete | `results/`, root pilot JSONs, reports                                         | 有 `eval_results_seed42.json`、pilot JSON、MCD split、verified templates，但缺 checkpoint、原始完整日志、wandb/tensorboard。       |
| baseline                   |              Partial | README / eval script / reports                                                | 有 flat、direct CoT、cot_budget、retrieval/MCTS 等；但 baseline 公平性和 checkpoint 追溯不足。                                     |
| 失败实验记录                     |                  Yes | `AUTO_REVIEW.md`, `ARIS_REVIEW.md`, `PIPELINE_REPORT.md`, `REVIEW_STATE.json` | 多份自审材料明确指出 MCD、synthetic、fallback、weak results 等问题。([GitHub][3])                                                   |
| ablation                   |              Partial | `EXPERIMENTS.md`, result files, scripts                                       | 有计划和部分 pilot/sweep 结果；缺完整 ablation matrix 和 seed 聚合。                                                               |
| requirements / environment |                  Yes | `requirements.txt`, `environment.yml`, `setup.sh`                             | root 可见。([GitHub][1])                                                                                              |

## Missing Items

| Missing Item                 | Why Needed                                                                 | What You Should Upload                                           |
| ---------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 完整 zip 或可 clone 镜像           | 本地 clone DNS 失败，无法运行 pytest、grep、命令复现                                      | repo zip，包括 `.git` 可选                                            |
| 真实 checkpoint / LoRA adapter | 判断 eval 是否加载正确模型、是否 stale checkpoint                                       | `results/*checkpoint*`, planner/flat/CoT adapters, training logs |
| 原始运行日志                       | 判断 seed、config、command、失败栈、显存、resume 情况                                    | `logs/`, `run.log`, wandb/tensorboard export                     |
| 论文草稿                         | claim-code-result 对齐必须检查 abstract/introduction/method/experiment 的实际 claim | `.tex`, PDF, figures, tables                                     |
| 数据 split 索引和 hash            | 检查 train/dev/test leakage、MCD split 语义正确性                                  | dataset cache manifest, split json, hashes                       |
| official baseline 复现记录       | NeurIPS claim 需要公平 baseline                                                | baseline commands, official code commit, logs                    |
| 负面实验完整表                      | 现有 reports 有摘要，但没有完整每 seed/per dataset 数据                                  | csv/json of all failed ablations                                 |

---

# 1. Repository Map

## 1.1 核心判断

这个仓库当前试图解决的问题是：从数学推理数据中抽取可执行程序，压缩成可复用模板/子程序库，再训练小模型生成模板组合计划，以提升 compositional math reasoning 的泛化、token efficiency 和 out-of-distribution 组合能力。README 的核心 framing 是“execution-verified, compression-mined procedures”，并给出从 Qwen teacher 抽取 JSON-AST 程序、构建 MCD split、训练 compose/flat/CoT student、评估 compose/flat/direct CoT/cot_budget 的 pipeline。([GitHub][2])

当前方法的核心假设是：
**如果程序片段经过执行验证并被压缩成模板库，那么 student 学会调用这些模板的组合计划，就能比直接 CoT 或 flat inline program 更好地泛化到 MCD-hard 组合。**

但静态审计显示：当前实现最关键的缺口不是“模板库不够大”或“调参不够”，而是**模板调用没有被显式、语义化地绑定到问题数量和前序调用输出**。许多训练样本把语义不匹配的问题映射到单个模板 ID，且 bindings 为空；当前 composition executor 主要依赖同名变量和 unique-type heuristic，而不是显式 output-to-input dataflow。这个缺失会同时解释“JSON 有效/执行率高但答案接近 0”的现象。([GitHub][4])

## 1.2 Component Map

| Component           | Path                                                 |                                                Purpose | Importance | Notes                                                                                                         |
| ------------------- | ---------------------------------------------------- | -----------------------------------------------------: | ---------- | ------------------------------------------------------------------------------------------------------------- |
| 主 README            | `README.md`                                          |                                    描述方法、环境、pipeline 命令 | High       | 是当前公开主张来源；含 teacher extraction、MCD、train、eval 命令。([GitHub][2])                                                |
| 实验计划                | `EXPERIMENTS.md`                                     |          benchmark、baseline、metric、NeurIPS minimum bar | High       | 设定 GSM8K/MATH/BBH/StrategyQA/Game24 等，要求 3 replications/bootstrap，但结果未完整满足。([GitHub][5])                      |
| Proposal            | `PROPOSAL.md`                                        |                         “Template Algebra” formal idea | Medium     | Aspirational；代码未完全实现 typed graph algebra / constrained compiler。                                              |
| 自审：ARIS             | `ARIS_REVIEW.md`                                     |                                 严厉 reviewer-style 负面审查 | High       | 明确指出 novelty overstated、proposal-code mismatch、no schema constrained decoding、no proper split 等。([GitHub][3]) |
| 自审：AUTO             | `AUTO_REVIEW.md`                                     |                        Round 1/2 reject risk and fixes | High       | 指出 MCTS gold reward cheating、MCD zero unseen compounds、synthetic contamination、weak 4% result 等。([GitHub][6]) |
| Pipeline report     | `PIPELINE_REPORT.md`                                 |                 ATLAS pipeline upgrade and weak result | High       | 记录 synthetic→real N=50 compose 4% vs baselines 0；弱但有诊断价值。([GitHub][7])                                        |
| SEVAL review        | `REVIEW_SEVAL_V2.md`                                 |               RLVR/library evolution/test-time tool 审查 | High       | 指出 flywheel vacuous、bigram abstraction weak、CoT-Pass@K confounded、DSL restrictive。([GitHub][8])               |
| 状态文件                | `REVIEW_STATE.json`                                  |                                        readiness state | High       | last score 3.0 not ready；列出 path_to_6: 1k+ verified programs, real MCD, compose > flat, 3 seeds。([GitHub][9]) |
| 方法计划                | `PLAN.md`                                            |                                            8-week plan | Medium     | 明确 Stage 2 希望 COMPOSE connect outputs to inputs；这恰好是当前缺失机制。([GitHub][10])                                     |
| DSL core            | `src/template_dsl.py`                                |       Program/Step/Subroutine/CompositionPlan/Executor | Critical   | 组合语义核心；当前 implicit env/name/type binding 是核心风险。([GitHub][11])                                                 |
| MCD split           | `src/mcd_split.py`                                   |                                    atom/compound split | Critical   | 当前 flow edge 是 consecutive-call potential flow，不是真实 dataflow。([GitHub][12])                                   |
| Template extraction | `scripts/extract_templates.py`                       | teacher program extraction, library, plan construction | Critical   | 生成 verified programs、library、compose_train/flat_train；当前 label faithfulness 风险最高。([GitHub][13])               |
| Student training    | `scripts/train_template_compiler.py`                 |                                   SFT compose/flat/CoT | High       | README 主训练入口；需审查 seed/resume/validation/schema。                                                               |
| Evaluation          | `scripts/eval_template_reasoning.py`                 |                       compose/flat/CoT/cot_budget eval | Critical   | fallback 混入、adapter missing、retrieval random fallback 都可能污染结论。                                                |
| RLVR evolution      | `src/rlvr_evolution.py`                              |                 SEVAL / library evolution / CoT-Pass@K | Medium     | 有启发，但建立在当前弱 composition semantics 上；不应作为 main path。([GitHub][14])                                             |
| Test-time tools     | `src/test_time_tools.py`                             |                    failure analysis and tool synthesis | Medium     | 目前是 heuristic candidate generation；适合作后续 ablation，不适合作主方法。([GitHub][15])                                      |
| Unit tests          | `tests/test_templatebank.py`                         |                                              DSL/tests | High       | 测试实际编码了“共享 slot 名”而非 output-flow composition 的语义。                                                             |
| Verified programs   | `results/templates_verified/all_programs_stats.json` |                            697 verified programs stats | High       | 显示 697 exec_ok/answer_correct；但 plan labels 不等于 faithful composition。([GitHub][16])                           |
| Compose train data  | `results/templates_verified/compose_train.json`      |                                     planner SFT labels | Critical   | 看到多条样本输出 single call + empty bindings，且 problem 与 template 不匹配。([GitHub][4])                                  |
| Flat train data     | `results/templates_verified/flat_train.json`         |                            flat inline baseline labels | Critical   | flat baseline 也可能在训练不匹配的 inlined template，而非原题程序。([GitHub][17])                                               |
| Library             | `results/templates_verified/subroutine_library.json` |                                    16-template library | High       | 许多模板像 whole-task pattern，slot/interface 未充分 grounded。([GitHub][18])                                           |
| Eval result         | `results/eval_v2/eval_results_seed42.json`           |                                            seed42 eval | Critical   | compose GSM8K 2/100, MCD 0/100；直接否定强 claim。([GitHub][19])                                                     |
| MCD split v2        | `results/mcd_split_v2.json`                          |                                    real verified split | High       | atom_tvd good, compound divergence high；但 compounds 来源有语义风险。([GitHub][20])                                    |
| MCD split 27B       | `results/mcd_split_27b.json`                         |                                          earlier split | Medium     | synthetic_used true artifact，atom_tvd 0.109897 高于 README 阈值。([GitHub][21])                                    |
| Pilot results       | root `pilot_results*.json`                           |                      dynamic/static/fixed budget pilot | Medium     | static/dynamic acc 均 .50；dynamic 只提高 reuse，fixed256 acc .525。([GitHub][22])                                   |
| Related work notes  | `PAPERS.md`                                          |                                             paper list | Medium     | 含 placeholder arXiv IDs，不能作为真实相关工作证据。                                                                         |

## 1.3 必须回答的问题

| Question               | Answer                                                                                                                                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 仓库当前试图解决什么问题           | 用执行验证的可复用程序模板提升数学推理组合泛化和 token efficiency。                                                                                                                                                                               |
| 当前已有方法是什么              | Teacher 抽取 JSON-AST 程序 → compression/mining 建 library → MCD split → train compose/flat/CoT student → eval compose/flat/direct/cot_budget。                                                                                |
| 当前方法核心假设               | 已验证程序片段可作为可组合抽象；student 学会调用模板比直接生成完整 reasoning 更能泛化。                                                                                                                                                                    |
| 声称解决的 prior limitation | 纯 CoT 不可复用、不可执行、不够 compositional；static template 不动态；memory/reasoning buffer 不严格验证。                                                                                                                                      |
| 主要训练入口                 | `scripts/train_template_compiler.py`。                                                                                                                                                                                    |
| 主要评估入口                 | `scripts/eval_template_reasoning.py`。                                                                                                                                                                                    |
| 数据处理                   | `scripts/extract_templates.py`, `src/mcd_split.py`, `results/templates_verified/*`。                                                                                                                                      |
| 模型核心                   | Student planner SFT in train script；DSL core in `src/template_dsl.py`。                                                                                                                                                   |
| loss / objective       | 主要是 SFT NLL；SEVAL 中有 RLVR/reward，但不是主 README pipeline。                                                                                                                                                                   |
| baseline               | flat_inline、direct_cot、cot_budget、pilot static/dynamic/fixed、retrieval/MCTS variants。                                                                                                                                    |
| configs                | `configs/template_config.yaml`, `configs/pilot_config.yaml` 等。                                                                                                                                                           |
| results/logs           | `results/eval_v2`, `results/templates_verified`, `results/extract_27b_500`, root pilot JSONs。                                                                                                                            |
| 论文 claim               | README/PROPOSAL/IDEA_REPORT/CLAUDE/EXPERIMENTS/PIPELINE_REPORT/REVIEW docs。                                                                                                                                              |
| 主线文件                   | `README.md`, `scripts/extract_templates.py`, `src/template_dsl.py`, `src/mcd_split.py`, `scripts/train_template_compiler.py`, `scripts/eval_template_reasoning.py`, `results/templates_verified/*`, `results/eval_v2/*`。 |
| 历史遗留                   | `src/template_algebra.py`, `scripts/eval_template_algebra.py`, early pilot files, synthetic 27B artifacts, some idea-stage/review-stage docs。                                                                            |
| dead code 可能性          | Template Algebra / SEVAL / test-time tools 部分可能未进入 main pipeline；需 Claude Code trace imports。                                                                                                                            |
| 会影响实验结论的文件             | `template_dsl.py`, `extract_templates.py`, `mcd_split.py`, `eval_template_reasoning.py`, train/eval configs, result aggregation scripts。                                                                                 |

---

# 2. Result Reliability Audit

## 2.1 总体结论

当前仓库中**没有足够可靠的正面主结果**可以支撑 NeurIPS 级 claim。最强可追溯结果反而是负面的：`eval_results_seed42.json` 中 compose 在 GSM8K 上只有 2/100，在 MCD_test 上 0/100；flat_inline 在 MCD_test 反而是 5/100。该结果有 JSON 记录、seed 和元信息，但缺 checkpoint/完整 command，因此是 medium-low reliability 的负面证据。([GitHub][19])

另一个重要事实是：`templates_verified` 的 697 个程序本身显示 exec_ok/answer_correct，但对应的 compose/flat 训练样本并不 faithful：多条问题被映射到语义不匹配的单个模板调用，且 `bindings` 为空。这不是单纯结果弱，而是 label construction 机制层面的污染信号。([GitHub][16])

## 2.2 Result Reliability Table

| Result ID | Result Name                                         | Dataset            |                   Metric |                     Claimed Value |                                                  Logged Value | Config                                |    Seed | Command                      | Checkpoint | Status                   | Reliability  | Issue                                                                 |
| --------- | --------------------------------------------------- | ------------------ | -----------------------: | --------------------------------: | ------------------------------------------------------------: | ------------------------------------- | ------: | ---------------------------- | ---------- | ------------------------ | ------------ | --------------------------------------------------------------------- |
| R1        | README main claim: frozen library improves MCD-hard | GSM8K/MCD-hard     |                      acc | `>= +15%` in proposal-style claim |                                                     Not found | README/config                         | Missing | Generic commands only        | Missing    | Missing Log              | unusable     | 没有支持该 claim 的完整 result table。                                         |
| R2        | README/pipeline: compose vs flat/CoT                | GSM8K/MCD          |                      acc |            compose should improve |                          GSM8K compose 0.02; MCD compose 0.00 | Likely `configs/template_config.yaml` |      42 | Missing exact run            | Missing    | Partially Verified       | medium-low   | JSON 负面；缺 checkpoint/command。([GitHub][19])                           |
| R3        | MCD_test flat_inline                                | MCD_test           |                      acc |                          baseline |                                                          0.05 | same                                  |      42 | Missing                      | Missing    | Partially Verified       | medium-low   | flat > compose on MCD；反驳主假设。([GitHub][19])                            |
| R4        | GSM8K cot_budget                                    | GSM8K              |                      acc |                          baseline |                                                          0.03 | same                                  |      42 | Missing                      | Missing    | Partially Verified       | medium-low   | cot_budget > compose 0.02，但都极低。([GitHub][19])                         |
| R5        | templates_verified program verification             | GSM8K              | exec_ok / answer_correct |                 verified programs |                            697/697 exec_ok and answer_correct | Missing                               | Missing | Missing                      | N/A        | Partially Verified       | medium       | 程序 final answer verified；不代表 plan/composition faithful。([GitHub][16]) |
| R6        | compose training labels                             | GSM8K train        |       plan label quality |             should be composition |                       examples are single-call empty bindings | N/A                                   |     N/A | N/A                          | N/A        | Verified artifact defect | high         | 问题与模板语义不匹配，bindings 空。([GitHub][4])                                   |
| R7        | flat training labels                                | GSM8K train        |   baseline label quality |               full inline program |                           examples inline mismatched template | N/A                                   |     N/A | N/A                          | N/A        | Verified artifact defect | high         | flat baseline 也可能训练错目标。([GitHub][17])                                 |
| R8        | MCD split 27B                                       | synthetic/GSM8K    |        atom_tvd / unseen |                        strict MCD |                        atom_tvd 0.109897; synthetic_used true | Missing                               | Missing | Missing                      | N/A        | Possibly Contaminated    | low/unusable | synthetic artifact；atom_tvd 不满足 README strict 阈值。([GitHub][21])       |
| R9        | MCD split v2                                        | templates_verified |        atom_tvd / unseen |                        strict MCD |                                atom_tvd 0.017366, unseen 9/13 | Missing                               | Missing | Missing                      | N/A        | Partially Verified       | medium-low   | 数值上较好，但 compounds 基于弱 plan/flow semantics。([GitHub][20])              |
| R10       | Pilot static vs dynamic                             | pilot dataset      |    acc / utility / reuse |            dynamic memory helpful | static acc .50; dynamic acc .50; reuse .65→.70; fixed256 .525 | `configs/pilot_config.yaml`           | Missing | pilot command in EXPERIMENTS | N/A        | Partially Verified       | low          | dynamic 未提升 acc/utility/tokens；external path 不可复现。([GitHub][22])      |
| R11       | Pipeline ATLAS synthetic→real                       | GSM8K N=50         |                      acc |           compose 4%, baselines 0 |                                                   report only | Missing                               | Missing | Missing                      | Missing    | Partially Verified       | low          | N=50、小数值、synthetic train、无 raw log。([GitHub][7])                      |
| R12       | SEVAL RLVR claims                                   | MATH/MCD-hard      |         +points / Pass@K |                       +10/+20 etc |                                              No metrics found | Missing                               | Missing | Missing                      | Missing    | Missing Log              | unusable     | 自审指出方法/比较 confounded。([GitHub][8])                                    |
| R13       | Related-work novelty claims                         | literature         |                      N/A |                 novelty-confirmed |                               `PAPERS.md` has placeholder IDs | N/A                                   |     N/A | N/A                          | N/A        | Contradicted/Unclear     | unusable     | placeholder arXiv IDs 不能作为学术证据。                                       |

## 2.3 可信度规则

| Evidence Type                                       | Use in Method Discovery                    |
| --------------------------------------------------- | ------------------------------------------ |
| Verified artifact defect, e.g. empty binding labels | Strong diagnostic evidence                 |
| Logged negative eval JSON                           | Medium diagnostic evidence                 |
| Pilot positive-ish reuse only                       | Weak signal, not method evidence           |
| Synthetic 4% report                                 | Historical low-confidence signal           |
| README/proposal claim without log                   | Not usable as evidence                     |
| SEVAL future claims                                 | Not usable as evidence; only idea fragment |

---

# 3. 代码正确性审查

## 3.1 最关键风险

最关键的代码风险是：**当前 CompositionExecutor 并没有真正实现显式 dataflow composition**。它的绑定逻辑大致是：如果 call 提供 binding 就用 binding；否则如果 slot 名在 env 中就用；否则如果同类型候选唯一就自动绑定；否则 fail。执行后更新 env 和 `__last_output__`。这意味着组合依赖变量名碰巧相同或类型唯一，而不是计划中明确指定“上一子程序输出喂给下一子程序输入”。([GitHub][11])

第二个关键风险是：`src/mcd_split.py` 的 flow compound 生成并不是根据真实绑定边，而是对 consecutive calls 添加 potential flow edge；这会让 MCD split 看起来有 compositional compounds，但这些 compounds 不一定对应真实数据流。([GitHub][12])

第三个关键风险是：训练数据里出现大量 empty-binding 单调用模板，且 problem 与模板语义不匹配，这会导致模型学习“模板检索/猜 ID”，而非“解析问题 → 绑定变量 → 组合执行”。([GitHub][4])

## 3.2 Suspected Bug Table

| Priority | File                                            | Function/Class                        | Code Region             | Suspicion                                                                       | Evidence                                                                                              | How to Verify                                                                    | Proposed Fix for Claude Code                                                       | Expected Effect                       | Confidence  |
| -------: | ----------------------------------------------- | ------------------------------------- | ----------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------- | ----------- |
|       P0 | `src/template_dsl.py`                           | `CompositionExecutor.execute`         | binding resolution      | 没有显式 output-to-input binding；依赖 slot name/env/unique type heuristic             | Executor 逻辑显示未建显式 binding refs；`__last_output__` 仅 env 更新。([GitHub][11])                              | 新建 ADD→MUL 测试，要求 MUL 输入明确绑定 ADD 输出；旧逻辑应失败或给错                                     | 新增 `BindingRef`, `DataflowPlan`, `DataflowExecutor`；禁止 silent implicit binding     | 组合语义从“顺序模板调用”变为“显式数据流图”               | high        |
|       P0 | `results/templates_verified/compose_train.json` | data artifact                         | training labels         | 多个问题被映射到语义不匹配模板，bindings `{}`                                                   | mango/coconut 等问题输出 `{"sub_id":"L00","bindings":{}}`；James media→payment/pizza template。([GitHub][4]) | 写 audit 脚本：执行 label plan 并比较原 verified program/gold answer；统计 empty-binding rate | 只保留可 faithful execution 的 explicit dataflow plan；输出 `plan_audit.json`              | 删除污染 supervision；减少高 valid 低 answer   | high        |
|       P0 | `results/templates_verified/flat_train.json`    | data artifact                         | flat labels             | flat baseline 训练的可能是 mismatched template inline program，不是原题完整 verified program | mango 问题输出 L00 fruit/crate 程序。([GitHub][17])                                                          | 对每个 flat label 执行并比对题目 gold                                                      | 重新构造 flat baseline：使用原题 verified full program，不用 mismatched library representative | baseline 公平性上升；避免误判 compose vs flat   | high        |
|       P0 | `src/mcd_split.py`                              | compound extraction                   | flow edge extraction    | MCD compound 不是真实 dataflow，只是 consecutive call potential flow                   | 代码对连续 call 总是添加 flow edge。([GitHub][12])                                                              | 用 empty-binding single-call/multi-call plan 检查 compounds                         | 改为从 `BindingRef(source=call_output)` 生成 flow compounds；无真实 edge 则不算 composition    | MCD-hard 才真正评估 compositional dataflow | high        |
|       P1 | `scripts/eval_template_reasoning.py`            | eval compose/flat                     | fallback metrics        | main accuracy 可能混合 method 和 CoT fallback                                        | eval JSON 有 fallback_rate/fallback_free_accuracy，README claim 容易使用 combined accuracy。([GitHub][19])   | 强制 `--no_fallback` 和 `--fallback_report_only` 比较                                 | 输出 `method_accuracy`, `fallback_rescued_accuracy`, `combined_accuracy_not_main`    | 防止 fallback 掩盖方法失败                    | high        |
|       P1 | `scripts/eval_template_reasoning.py`            | adapter loading                       | model loading           | adapter 缺失可能 silently evaluate base model                                       | 结果极低且缺 checkpoint path/hash；需 fail-hard                                                               | 临时重命名 adapter 目录，eval 应报错                                                        | 增加 `--require_adapters`，记录 model path/hash，adapter exists                          | 防 stale/missing checkpoint 污染         | medium      |
|       P1 | `scripts/train_template_compiler.py`            | training loop                         | seed/resume/validation  | README 使用 seeds，但脚本是否暴露 seed/val/resume 不清晰；auto-resume 可能 stale                | README 说 auto-resume；result 只有 seed42。([GitHub][2])                                                   | grep argparse；运行 `--help`                                                        | 增加 explicit `--seed`, `--no_resume`, data hash, val split, checkpoint manifest     | 可复现性提升                                | medium      |
|       P1 | `src/template_dsl.py`                           | `SubroutineLibrary.add` / fingerprint | fingerprint             | program fingerprint 只看 op/expr，忽略 slot semantics/type/name                      | fingerprint code 依赖 op + expr。([GitHub][11])                                                          | 构造两个 slot semantics 不同但 expr 相同的程序，看是否 dedup                                     | fingerprint 加 slots dtype/role/canonical IO graph；记录 collision                     | 减少错误模板合并                              | high        |
|       P1 | `src/template_dsl.py`                           | `DType.coerce`                        | bool coercion           | `bool("False")` 为 True                                                          | BOOL coercion uses `bool(value)`。([GitHub][11])                                                       | unit test `"False"` should false or reject                                       | 明确 parse bool string 或 reject                                                      | 避免类型 silently wrong                   | medium      |
|       P2 | `scripts/extract_templates.py`                  | `_extract_bindings`                   | number binding          | 按数值出现顺序绑定 slot，弱语义，易 spurious                                                   | extraction prompt/logic weakly binds numbers；训练 label 已暴露空绑定问题。([GitHub][13])                         | compare extracted slot names vs problem spans/entities                           | 引入 quantity grounding table `{qid, span, value, entity, type}`；绑定 refs 指向 qid      | 让计划可解释、可扰动验证                          | high        |
|       P2 | `src/rlvr_evolution.py`                         | `find_patterns`, `abstract_pattern`   | SEVAL library evolution | pattern mining 是 n-gram/bigram over successful programs，可能只 concat，不是真抽象        | review 指出 flywheel vacuous；code 有 bigram/trigram pattern mining。([GitHub][8])                         | 对 evolved library 测试是否引入新 interface-flow 功能                                      | 暂时 freeze SEVAL；等 GIFT dataflow 后再 evolution explicit-flow fragments               | 避免在坏 semantics 上优化                    | medium-high |
|       P2 | `src/test_time_tools.py`                        | tool generation                       | repair heuristics       | 候选工具 sum/product/diff/type adapter/model generation，可能用 gold 验证，机制弱             | code 包含 heuristic strategies。([GitHub][15])                                                           | 确认 eval/test 是否用 gold_answer                                                     | 仅作 ablation；train/dev 可用 gold，test 禁止 gold-guided selection                        | 避免 test leakage / novelty weak        | medium      |

---

# 4. Claim-Code-Result Matrix

| Claim                                                               | Source File                  | Implementation File                                   | Result Evidence                             | Status               | Problem                                                                               | Confidence |
| ------------------------------------------------------------------- | ---------------------------- | ----------------------------------------------------- | ------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------- | ---------- |
| “Execution-verified procedures can be mined from GSM8K”             | README / `EXPERIMENTS.md`    | `scripts/extract_templates.py`, `src/template_dsl.py` | 697 verified programs stats                 | Partially Supported  | final answer verified，不等于 reusable plan faithful。([GitHub][16])                       | medium     |
| “Compose student improves over flat/CoT on MCD-hard”                | README/PROPOSAL              | train/eval scripts                                    | seed42 MCD compose 0, flat 5/100            | Contradicted         | 当前可见结果反向。([GitHub][19])                                                               | medium     |
| “Frozen library yields ≥15% gain”                                   | PROPOSAL/IDEA                | DSL/library/eval                                      | No logged support                           | Unsupported          | aspirational claim；必须删除或改成 hypothesis。                                                | high       |
| “Template algebra supports compose/abstract/specialize typed graph” | PROPOSAL                     | `template_dsl.py`, maybe `template_algebra.py`        | No robust evidence                          | Unsupported          | 当前 composition 是 implicit env/slot heuristic，不是 typed dataflow algebra。([GitHub][11]) | high       |
| “Strict MCD split controls atom distribution”                       | README                       | `src/mcd_split.py`                                    | v2 atom_tvd 0.017366; 27B atom_tvd 0.109897 | Partially Supported  | v2 数值可；但 compound semantics 不是真 dataflow。([GitHub][20])                               | medium     |
| “Compression / MDL predicts generalization”                         | README/EXPERIMENTS           | library mining                                        | No direct correlation result                | Unsupported          | 没有看到 MDL-vs-accuracy analysis。                                                        | high       |
| “Search-time repair recovers ≥25% failed cases”                     | README/IDEA                  | MCTS/test-time tools                                  | No reliable log                             | Unsupported          | 自审中承认 earlier MCTS reward/gold issues。([GitHub][6])                                   | medium     |
| “SEVAL RLVR-evolved library improves MATH MCD-hard by +10”          | `CLAUDE.md`, SEVAL docs      | `src/rlvr_evolution.py`                               | Missing metrics                             | Not Testable         | review 指出 comparison confounded and abstraction weak。([GitHub][8])                    | medium     |
| “Test-time tool building recovers failures”                         | `CLAUDE.md`, SEVAL           | `src/test_time_tools.py`                              | Missing reliable metrics                    | Not Testable         | heuristics plausible but not validated; leakage needs audit。([GitHub][15])            | medium     |
| “Dynamic memory beats static/fixed budgets”                         | `EXPERIMENTS.md` pilot       | pilot script                                          | dynamic acc .50 = static .50; fixed256 .525 | Contradicted / Mixed | 只有 reuse 提升，不是 accuracy/utility 提升。([GitHub][22])                                     | high       |
| “Full logs/configs support reproducibility”                         | `EXPERIMENTS.md` requirement | repo                                                  | partial results only                        | Unsupported          | checkpoint、command、seed aggregation、hash 缺失。                                          | high       |
| `PAPERS.md` novelty landscape                                       | `PAPERS.md`                  | N/A                                                   | placeholder IDs                             | Unclear/Unsupported  | 不能把 `2501.XXXXX` 作为真实论文。                                                              | high       |

---

# 5. Phenomenon Ledger

| ID     | Observation                                                        | Type                              | Where Found                                      | Setting                 | Metric            | Compared To               | Reliability | What It Suggests                                                        | What It Rules Out                                                      | Confidence  |
| ------ | ------------------------------------------------------------------ | --------------------------------- | ------------------------------------------------ | ----------------------- | ----------------- | ------------------------- | ----------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------- | ----------- |
| PHE-01 | compose GSM8K accuracy 2/100, valid 96%, exec 81%                  | Negative / Anomalous              | `eval_results_seed42.json`                       | seed42, library size 16 | acc/valid/exec    | flat/direct/cot_budget    | medium-low  | syntax/JSON/execution 不是主要瓶颈；semantic binding 是瓶颈                       | “只要提高 parse validity 就能赢”                                              | medium      |
| PHE-02 | compose MCD accuracy 0/100 while exec 93%                          | Negative / Anomalous              | same                                             | MCD_test                | acc/exec          | flat 5/100                | medium-low  | MCD 组合语义失败；execution ≠ reasoning correctness                            | “compose 已经 learned transferable composition”                          | medium      |
| PHE-03 | flat_inline MCD 5/100 > compose 0/100                              | Negative                          | same                                             | MCD_test                | acc               | compose                   | medium-low  | 组合调用路径比 flat 更脆弱                                                        | “library call abstraction 本身优于 inline”                                 | medium      |
| PHE-04 | cot_budget GSM8K 3/100 > compose 2/100                             | Negative                          | same                                             | GSM8K                   | acc               | compose                   | medium-low  | 当前 compose 没攻击真正瓶颈                                                      | “compose 比 simple budget baseline 更强”                                  | medium      |
| PHE-05 | verified programs 697/697 exec_ok and answer_correct               | Positive but bounded              | `all_programs_stats.json`                        | GSM8K                   | exec/answer       | N/A                       | medium      | 单题程序抽取/执行有价值，可作为 source of primitives                                   | “program verification automatically gives faithful composition labels” | medium      |
| PHE-06 | compose_train 多为 single-call empty bindings                        | Negative / Bug-like               | `compose_train.json`                             | train labels            | label quality     | desired plan              | high        | planner supervision 未教 grounding/binding                                | “模型学不到组合只是因为模型小”                                                       | high        |
| PHE-07 | problem 与 selected template 语义明显不匹配                                | Negative / Anomalous              | `plans_with_programs.json`, `compose_train.json` | train labels            | semantic match    | gold problem              | high        | library matching 用 structural/op sequence 而非 semantic interface         | “模板 ID 是语义正确子程序”                                                       | high        |
| PHE-08 | flat_train 也 inline mismatched template                            | Negative / Baseline contamination | `flat_train.json`                                | baseline labels         | label quality     | fair flat                 | high        | flat baseline 不公平/不可靠                                                   | “flat baseline 已经严格实现”                                                 | high        |
| PHE-09 | pilot dynamic memory acc = static acc = .50; reuse .70 > .65       | Mixed                             | pilot JSON                                       | pilot                   | acc/utility/reuse | static/fixed              | low         | retrieval/memory 可提升 reuse，但不自动提升 correctness                           | “memory reuse 就是 accuracy gain”                                        | medium      |
| PHE-10 | fixed256 pilot acc .525 > dynamic/static .50                       | Negative for dynamic              | pilot JSON                                       | pilot                   | acc               | dynamic/static            | low         | token budget/simple baseline 可能强；必须比较强 baseline                         | “dynamic memory 已经优于 simple budget”                                    | medium      |
| PHE-11 | synthetic 27B extraction used synthetic fallback                   | Possibly contaminated             | extraction_meta                                  | extract_27b_500         | synthetic_used    | real data                 | high        | early artifacts 不应进入 main evidence                                      | “27B split/result 可作为真实主证据”                                            | high        |
| PHE-12 | MCD split v2 atom_tvd good but compounds few/noisy                 | Mixed                             | `mcd_split_v2.json`, code                        | verified                | atom_tvd/compound | desired MCD               | medium      | split machinery可用，但 compound definition 要重写                             | “当前 MCD 已测试真实 semantic composition”                                    | medium-high |
| PHE-13 | MCD flow edges added for consecutive calls                         | Negative / Methodology bug        | `src/mcd_split.py`                               | split feature           | compound          | real flow                 | high        | MCD 需要 explicit binding edges                                           | “call adjacency = compositional dataflow”                              | high        |
| PHE-14 | CompositionExecutor uses implicit slot/env/type binding            | Negative / Mechanism gap          | `template_dsl.py`                                | execution               | semantics         | explicit flow             | high        | 必须新增 explicit binding refs                                              | “现有 executor 已实现 COMPOSE output-input”                                 | high        |
| PHE-15 | self-review score 3/10 not ready                                   | Negative meta-evidence            | `REVIEW_STATE.json`, ARIS/AUTO                   | project state           | readiness         | NeurIPS bar               | medium      | 仓库作者/agents 已识别 major gaps                                              | “只需 polish paper”                                                      | high        |
| PHE-16 | SEVAL review says pattern abstraction vacuous/bigram-like          | Negative                          | `REVIEW_SEVAL_V2.md`, code                       | SEVAL                   | mechanism         | claimed flywheel          | medium      | RLVR/evolution 不能先作为主线，需先修 representation                               | “evolution over current library is sufficient”                         | medium      |
| PHE-17 | test-time tools are heuristic repairs                              | Mixed / Weak                      | `src/test_time_tools.py`                         | failure repair          | recovery          | systematic synthesis      | medium      | 可作为 later ablation；不能主 claim                                            | “tool building 已是 robust missing mechanism”                            | medium      |
| PHE-18 | README protocol asks 3 seeds/bootstrap, result only seed42 visible | Negative reproducibility          | README/results                                   | eval                    | seed aggregation  | required protocol         | high        | 必须先修 logging/aggregation                                                | “当前结果可 claim stable”                                                   | high        |
| PHE-19 | high exec but near-zero answer                                     | Anomalous                         | eval JSON                                        | compose                 | exec vs acc       | expected correlation      | medium      | execution validity lacks semantic faithfulness                          | “executor correctness implies answer correctness”                      | high        |
| PHE-20 | whole-task-ish templates with support counts                       | Mixed                             | subroutine_library                               | library                 | support/mdl       | compositional abstraction | medium      | compression finds repeated forms but may not expose reusable interfaces | “support count = reusable compositional module”                        | medium      |

---

# 6. Design Constraints

| Constraint ID | Derived From Observation | Constraint Type    | Meaning                                                         | Implication for New Method                                                                     | Confidence  |
| ------------- | ------------------------ | ------------------ | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ----------- |
| C-01          | PHE-05                   | Must Preserve      | Verified executable programs are valuable source data           | Keep DSL/program verification as source of primitives, not final method proof                  | medium      |
| C-02          | PHE-06/PHE-07            | Must Fix           | Training labels must be faithful to problem semantics           | Generate only explicit grounded dataflow plans that reproduce original program/gold            | high        |
| C-03          | PHE-14                   | Must Fix           | Composition must use explicit output-to-input edges             | Replace implicit env/slot heuristic with `BindingRef` plan DAG                                 | high        |
| C-04          | PHE-01/PHE-02/PHE-19     | Must Explain       | High valid/exec + low acc means syntax is solved, semantics not | New method must log semantic faithfulness, active binding, final answer correctness separately | high        |
| C-05          | PHE-13                   | Must Fix           | MCD compounds must be real dataflow compounds                   | Split on explicit flow edges, not consecutive calls                                            | high        |
| C-06          | PHE-03/PHE-08            | Must Control       | Baselines may be contaminated                                   | Rebuild flat baseline from original verified full programs and evaluate method-only            | high        |
| C-07          | PHE-09/PHE-10            | Must Avoid         | Reuse alone is not enough                                       | Do not claim memory/template reuse unless it improves method-only accuracy or calibration      | medium      |
| C-08          | PHE-11                   | Must Not Claim     | Synthetic artifacts cannot support real benchmark claims        | Archive synthetic 27B results or label as historical                                           | high        |
| C-09          | PHE-18                   | Must Stabilize     | Need multi-seed and CI                                          | Add seed control, data hash, checkpoint hash, aggregation with std/CI                          | high        |
| C-10          | PHE-16                   | Must Avoid         | Evolution over broken semantics is meaningless                  | Freeze SEVAL as ablation until explicit dataflow works                                         | medium-high |
| C-11          | PHE-17                   | Must Control       | Test-time repair can leak or overfit                            | Gold answer allowed only for train/dev verification, never test-time selection                 | medium-high |
| C-12          | PHE-12                   | Must Generalize    | MCD should test unseen explicit structures                      | New split must enforce atom balance and unseen dataflow motifs                                 | medium      |
| C-13          | PHE-20                   | Must Differentiate | Compression alone is too close to prior library induction       | Novelty must be grounded interface-flow faithfulness, not “we mine templates”                  | high        |
| C-14          | R1/R2 claim mismatch     | Must Not Claim     | Current strong claims unsupported                               | Rewrite README/paper claims as provisional hypotheses until validated                          | high        |
| C-15          | PHE-01/PHE-02            | Must Test          | New method must beat old best positive fragment, not just flat  | Add A/B/C controls: old fragment only, new without mechanism, full new                         | high        |

---

# 7. Negative-to-Insight Analysis

| Negative Observation                   | Failed Assumption                                   | Why the Assumption Failed                                           | What Mechanism Is Missing                                      | New Design Requirement                                                              |
| -------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| compose MCD 0/100 with 93% exec        | Executable plan implies correct reasoning           | Executor can execute semantically wrong plan                        | Explicit semantic dataflow + final-output faithfulness         | Plan must bind every slot to quantity or prior output and pass execution/gold audit |
| GSM8K compose 2/100 despite high valid | JSON validity is main bottleneck                    | Model outputs syntactically valid but semantically ungrounded calls | Grounded quantity-to-slot binding                              | Add quantity grounding and binding loss/logs                                        |
| single-call empty-binding labels       | Template selection alone can supervise composition  | Labels do not teach variable binding or multi-call dependency       | Interface binding supervision                                  | Regenerate training data from verified full program traces                          |
| problem-template mismatch              | Structural op sequence match is enough              | Arithmetic shape ignores entities/roles/question target             | Entity/role-aware interface matching                           | Include slot role, source span, quantity id, output role                            |
| flat baseline uses mismatched template | Baseline comparison is fair                         | Baseline label pipeline shares same contamination                   | Clean baseline construction                                    | Rebuild flat from original verified program, not representative library template    |
| MCD compounds from adjacency           | Call sequence represents compositional structure    | Adjacent calls may have no data dependency                          | Real flow-edge extraction                                      | MCD split over explicit binding graph motifs                                        |
| dynamic memory reuse without acc gain  | Reusing prior template improves correctness         | Reuse can select wrong template or omit binding                     | Usefulness-gated reuse                                         | Acceptance gate based on executable faithful dataflow, not retrieval score          |
| SEVAL bigram abstraction critique      | Evolving frequent call n-grams creates abstractions | Frequent sequences may not encode reusable interfaces               | Abstraction with typed inputs/outputs and holdout faithfulness | Evolve only fragments with explicit interface and active binding                    |
| test-time repair heuristic             | Simple sum/product/diff recovers reasoning          | Heuristics do not diagnose missing variable binding                 | Systematic failure-to-interface gap detection                  | Use verifier to locate unbound/ambiguous slots, not generic arithmetic guesses      |
| only seed42 visible                    | Best seed/one run enough                            | Instability unknown                                                 | Reproducibility protocol                                       | 3+ seeds, CI, command/config/checkpoint manifest                                    |

---

# 8. Method Synthesis Table

| Evidence Fragment            | Source in Repo                                       | What It Reveals                                          | Generalized Principle                                  | Use in New Method?        | How to Transform It                                                          |
| ---------------------------- | ---------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------ | ------------------------- | ---------------------------------------------------------------------------- |
| 697 verified programs        | `results/templates_verified/all_programs_stats.json` | Teacher/executor can produce correct full programs       | Verification is useful, but only at full-program level | Yes                       | Use as source traces to mine subroutines and derive faithful dataflow labels |
| 16-template library          | `subroutine_library.json`                            | Compression finds recurring arithmetic skeletons         | Reuse needs typed semantic interfaces                  | Yes, transformed          | Recompute library signatures with slots, roles, dtype, output semantics      |
| compose_train empty bindings | `compose_train.json`                                 | Current supervision is not grounded                      | Composition requires explicit binding labels           | Yes, as negative evidence | Replace with `compose_train_gift.json` only after dataflow audit             |
| flat_train mismatch          | `flat_train.json`                                    | Baseline pipeline contaminated                           | Baselines must be constructed from original truth      | Yes, as fix target        | Rebuild flat baseline from original verified program                         |
| high exec low acc eval       | `eval_results_seed42.json`                           | Execution syntax not enough                              | Need semantic faithfulness metrics                     | Yes                       | Add active-binding and final-output faithfulness checks                      |
| MCD v2 atom balance          | `mcd_split_v2.json`                                  | Split machinery can satisfy atom balance numerically     | MCD useful if compounds are real                       | Yes, transformed          | Redefine compounds from dataflow motifs                                      |
| Pilot dynamic reuse          | pilot JSON                                           | Retrieval can increase reuse but not accuracy            | Reuse must be gated by correctness/faithfulness        | Maybe                     | Keep as ablation; add usefulness gate                                        |
| MCTS/search repair           | `PIPELINE_REPORT`, scripts                           | Search may explore alternatives                          | Search is secondary; representation first              | Later ablation            | Search over explicit dataflow plans only                                     |
| SEVAL RLVR                   | `src/rlvr_evolution.py`, review                      | Evolution idea plausible but currently weak              | Library evolution needs real interfaces                | Later ablation            | Freeze until GIFT core works                                                 |
| Test-time tools              | `src/test_time_tools.py`                             | Failure repair can propose local tools                   | Repair should target missing binding/interface gaps    | Later ablation            | Convert to dataflow gap repair, no test gold                                 |
| Proposal template algebra    | `PROPOSAL.md`/`PLAN.md`                              | Intended mechanism was typed compose/abstract/specialize | The good idea is explicit typed composition            | Yes                       | Implement minimal version as GIFT, not broad algebra claim                   |

---

# 9. Missing Mechanism Diagnosis

1. **Missing Mechanism Name:**
   **Grounded Interface-Flow Binding and Faithfulness Verification**

2. **One-Sentence Diagnosis:**
   The repository has verified single-problem programs and recurring arithmetic skeletons, but it lacks an explicit mechanism that binds problem quantities to subroutine interfaces and connects subroutine outputs to later subroutine inputs, so “composition” degenerates into ungrounded template selection or concatenation.

3. **Evidence From Positive Results:**
   The 697 verified programs show that executable symbolic traces can be obtained and checked against answers; this supports keeping execution verification as a foundation.([GitHub][16])

4. **Evidence From Negative Results:**
   compose gets 2/100 on GSM8K and 0/100 on MCD while maintaining high valid/executable rates, implying that the issue is not merely syntax or parser failure but semantic binding failure.([GitHub][19])

5. **Evidence From Unstable / Mixed Results:**
   Pilot dynamic memory improves reuse rate but not accuracy or utility, which suggests reusable memories/templates are only helpful when selected and bound correctly.([GitHub][22])

6. **Evidence From Failed Ablations / Self-Review:**
   Reviews explicitly criticize weak abstraction, bad MCD, synthetic contamination, and proposal-code mismatch; SEVAL review says pattern abstraction is bigram-like/vacuous unless it becomes parameterized and verified.([GitHub][3])

7. **Why Existing Method Cannot Solve It:**
   Current `CompositionExecutor` does not require explicit dataflow edges; it resolves slots from env/name/type heuristics. Current training labels often have empty bindings. This cannot teach or enforce faithful compositional reasoning.([GitHub][11])

8. **Why Simple Tuning Cannot Solve It:**
   More epochs, larger model, or different LR can improve JSON imitation, but the target labels themselves do not encode correct binding or dataflow. Tuning a model on empty-binding mismatched labels will optimize the wrong behavior.

9. **Why Existing Best Positive Fragment Is Insufficient:**
   The best positive fragments are weak: pilot reuse improves without accuracy gain; synthetic N=50 compose 4% is low-confidence; verified full programs do not imply faithful composition. None adds explicit grounded interface-flow semantics.

10. **What New Mechanism Must Do:**
    It must represent each plan as a typed DAG where every subroutine input is bound either to a grounded problem quantity/span or to a specific previous call output; it must execute using only those explicit refs; and it must filter/train/evaluate only plans whose explicit dataflow is faithful.

11. **Confidence:**
    **High** that this mechanism is missing; **medium-low** that the proposed method will empirically outperform after implementation, because no GIFT experiment has been run yet.

---

# 10. New MAIN METHOD PATH

## New MAIN METHOD PATH: **GIFT — Grounded Interface-Flow Template Composition**

1. **Method Name Placeholder:**
   **GIFT: Grounded Interface-Flow Template Composition**

2. **One-Sentence Core Idea:**
   Train and evaluate a planner that emits executable subroutine-call DAGs with explicit typed bindings from problem quantities and previous call outputs, and accept only plans that pass dataflow faithfulness checks.

3. **Core Missing Mechanism It Adds:**
   Explicit grounded input/output binding plus faithfulness verification.

4. **What Phenomena It Explains:**
   It explains why verified full programs exist but compose fails; why valid/executable JSON is high but answer accuracy near zero; why MCD split can look structured but fail; why memory reuse alone does not improve accuracy.

5. **What Negative Results It Fixes:**
   Empty-binding labels, semantic template mismatch, fake flow compounds, fallback-masked metrics, flat baseline contamination.

6. **What Existing Positive Signals It Generalizes:**
   It preserves executable DSL verification and compression-mined templates, but turns them into typed interface-flow primitives instead of ID-only templates.

7. **Why Existing Best Path Is Not Enough:**
   The existing best fragments do not force correct slot binding or output-to-input dependencies. They can reuse templates while still solving the wrong problem.

8. **Core Mechanism:**
   A plan is no longer `[{sub_id, bindings:{}}]`; it is a typed DAG:

   * call node: `c_t = subroutine_id`;
   * slot binding: `slot_j <- quantity:q_i`, `slot_j <- call_output:c_k`, or explicitly typed constant;
   * final answer: selected call output;
   * verifier: type, coverage, acyclicity, execution, answer match during training/dev, and active-binding perturbation.

9. **New Objective / Loss:**
   `L_total = L_plan + λ_bind L_bind + λ_type L_type + λ_exec L_exec + λ_gate L_gate + λ_comp L_comp`

10. **New Architecture or Module:**
    Add `BindingRef`, `DataflowPlan`, `PlanCall`, `DataflowExecutor`, `PlanFaithfulnessAuditor`, and `QuantityGrounder`.

11. **New Training Procedure:**
    Generate faithful dataflow labels from verified full programs; filter unfaithful labels; train planner to emit canonical JSON DAG; validate on method-only execution accuracy and binding metrics.

12. **New Evaluation Protocol:**
    Report method-only accuracy as primary; fallback only as secondary rescue metric; require adapter/checkpoint hashes; evaluate A/B/C controls:

* A. Existing Best Positive Fragment Only
* B. GIFT without explicit dataflow / faithfulness gate
* C. Full GIFT

13. **What Existing Components It Reuses:**
    DSL primitives, verified full-program extraction, library idea, MCD split scaffold, train/eval script scaffolds.

14. **What Existing Components It Deletes:**
    No silent deletion of negative results; delete or mark invalid only unsupported headline claims and placeholder literature entries.

15. **What Existing Components It Rewrites:**
    CompositionExecutor, plan labels, flat baseline construction, MCD compound extraction, eval metric handling, checkpoint loading, seed/logging.

16. **What Existing Components It Keeps Only as Ablation:**
    old compose empty-binding pipeline, retrieval compose, MCTS repair, dynamic memory, SEVAL evolution.

17. **What Existing Components It Keeps Only as Baseline:**
    direct CoT, cot_budget, faithful flat inline, official PAL/BoT-style baselines where feasible.

18. **Why This Is Not Merely the Existing Best Path:**
    Existing path selects template IDs and executes implicit environment bindings. GIFT changes the representation, supervision, executor, split definition, metrics, and ablations around explicit dataflow. It is a mechanism-level change, not a narrower choice among existing branches.

19. **Why This Could Produce Real Positive Results:**
    It aligns training labels with the real causal structure of arithmetic reasoning: identify quantities, bind them to roles, compose intermediate outputs, and verify final answer.

20. **Why This Is Mechanism-Level Different from Prior Work:**
    The novelty target is not “templates” or “program execution” alone; those overlap with prior work. The differentiating claim must be: **execution-verified reusable abstractions with explicit grounded interface-flow supervision and MCD evaluation over true dataflow motifs.**

21. **Main Risk:**
    Faithful dataflow label coverage may be too low; DSL may be too restrictive for MATH; official baselines like PAL/BoT/Faithful CoT may dominate if matched fairly.

22. **Minimal Falsification Experiment:**
    On 50–100 verified GSM8K examples, if Full GIFT cannot one-batch overfit, cannot produce >70% valid active bindings, or does not beat both A and B on method-only accuracy across 3 seeds, stop and pivot.

23. **Confidence:**
    **Medium-low** as an empirical path; **high** as the correct mechanism-level diagnosis.

---

# 11. Formal Method Description

## 11.1 Problem Setup

Given a math word problem `x`, extract grounded quantities:

[
Q(x)={q_i=(span_i, value_i, entity_i, dtype_i, role_i)}_{i=1}^{n}
]

Given a library of executable subroutines:

[
\mathcal{L} = {s_k=(I_k, O_k, P_k)}
]

where `I_k` are typed input slots, `O_k` is a typed output, and `P_k` is executable DSL code.

The model predicts a plan DAG:

[
G=(V,E,B)
]

where each node (v_t) calls a subroutine (s_{k_t}), and every input slot has an explicit binding:

[
B(t,j) \in {\text{Quantity}(q_i), \text{CallOutput}(v_m), \text{Constant}(c)}, \quad m<t
]

## 11.2 Existing Method Failure

Existing composition allows plan calls with empty bindings and relies on implicit env/type/name matching. That makes execution possible but not semantically faithful. The observed result is high valid/executable rate with near-zero answer accuracy.([GitHub][19])

## 11.3 New Insight

A reusable template is not useful unless its interface is grounded. The missing object is not a better template ID; it is a verified **interface-flow binding graph**.

## 11.4 Method Overview

GIFT has five parts:

1. Quantity grounding.
2. Typed subroutine interface extraction.
3. Explicit dataflow plan generation.
4. Faithfulness verification and active-binding audit.
5. Method-only evaluation with A/B/C controls.

## 11.5 Algorithm

**Algorithm: GIFT — Grounded Interface-Flow Template Composition**

**Input:**
Problem `x`; extracted quantities `Q`; subroutine library `L`; planner model `πθ`; verifier `V`; acceptance threshold `τ`.

**Output:**
Answer `ŷ`; executable dataflow plan `G`; audit logs.

**Steps:**

1. Extract grounded quantities `Q(x)` with spans, values, types, entity labels, and roles.
2. Generate canonical JSON dataflow plan:

   ```json
   {
     "calls": [
       {
         "call_id": "c0",
         "sub_id": "L03",
         "bindings": {
           "num_items": {"source": "quantity", "id": "q1"},
           "unit_price": {"source": "quantity", "id": "q2"}
         }
       },
       {
         "call_id": "c1",
         "sub_id": "L07",
         "bindings": {
           "subtotal": {"source": "call_output", "call_id": "c0"},
           "tax": {"source": "quantity", "id": "q3"}
         }
       }
     ],
     "final": {"source": "call_output", "call_id": "c1"}
   }
   ```
3. Typecheck: all slots covered; no cycles; dtype matches; final output reachable.
4. Execute via `DataflowExecutor`, resolving only explicit refs.
5. Audit active binding: perturb each quantity binding or intermediate output and verify final output changes when expected.
6. Accept if valid, executable, calibrated, and faithful; otherwise mark failure or fallback separately.

## 11.6 Objective

[
L_{\text{total}} =
L_{\text{plan}}

* \lambda_{\text{bind}} L_{\text{bind}}
* \lambda_{\text{type}} L_{\text{type}}
* \lambda_{\text{exec}} L_{\text{exec}}
* \lambda_{\text{gate}} L_{\text{gate}}
* \lambda_{\text{comp}} L_{\text{comp}}
  ]

where:

| Term              | Meaning                                               | Phenomenon / Constraint Addressed                        |
| ----------------- | ----------------------------------------------------- | -------------------------------------------------------- |
| (L_{\text{plan}}) | NLL over canonical call sequence / DAG serialization  | planner must learn structure, not just answer            |
| (L_{\text{bind}}) | CE over quantity refs and previous-output refs        | fixes empty-binding labels                               |
| (L_{\text{type}}) | penalty/CE for invalid type/slot/reachability         | fixes implicit type heuristic                            |
| (L_{\text{exec}}) | train/dev verifier reward or loss for executed answer | aligns objective with correctness                        |
| (L_{\text{gate}}) | confidence/acceptance calibration                     | prevents fallback/invalid plans being counted as success |
| (L_{\text{comp}}) | optional compression prior                            | keeps useful library bias but not as main claim          |

## 11.7 Training Pipeline

1. Start from verified full programs.
2. Mine candidate subroutines with explicit input/output slots.
3. Align subroutines to full-program steps.
4. Construct explicit dataflow plans.
5. Execute plan; require final answer match.
6. Run active-binding perturbation.
7. Save:

   * `compose_train_gift.json`
   * `flat_train_faithful.json`
   * `plan_audit.json`
   * `library_gift.json`
8. Train planner with seed, data hash, checkpoint hash.
9. Validate method-only accuracy and binding metrics.

## 11.8 Inference / Evaluation Pipeline

Primary metrics:

| Metric                       | Meaning                                             |
| ---------------------------- | --------------------------------------------------- |
| `method_accuracy`            | answer correct without fallback                     |
| `plan_valid_rate`            | JSON/schema valid                                   |
| `typecheck_rate`             | all slots/refs valid                                |
| `exec_rate`                  | plan executes                                       |
| `active_binding_rate`        | perturbing used inputs affects output appropriately |
| `true_multicall_flow_rate`   | percentage with at least one `call_output` edge     |
| `fallback_rescue_rate`       | secondary only                                      |
| `combined_accuracy_not_main` | reported but not used as claim                      |

## 11.9 Expected Empirical Signature

GIFT should show:

1. Higher method-only accuracy than old compose.
2. Lower empty-binding rate.
3. Higher active-binding rate.
4. True multi-call flow plans on MCD-hard.
5. Full GIFT > GIFT without dataflow > old fragment, with confidence intervals.

## 11.10 Required Ablations

| Ablation                                 | Purpose                                                            |
| ---------------------------------------- | ------------------------------------------------------------------ |
| Existing Best Positive Fragment Only     | Tests whether old template reuse alone explains gains              |
| GIFT without explicit `call_output` refs | Tests dataflow mechanism                                           |
| GIFT without active-binding gate         | Tests faithfulness verification                                    |
| GIFT with random library IDs             | Tests template semantics                                           |
| Faithful flat inline                     | Tests whether library calls add value over full program generation |
| No compression prior                     | Tests whether MDL/compression matters                              |
| Official PAL/Faithful CoT/BoT baseline   | Tests novelty and practical strength                               |

---

# 12. Related Work and Novelty Risk

## 12.1 Important related work

| Paper                                | Year / Venue | Code                          | Mechanism                                                                                                  | Why Close                                         | Difference from GIFT                                                                 | Novelty Risk | Required Differentiation Experiment                                              |
| ------------------------------------ | ------------ | ----------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------ | -------------------------------------------------------------------------------- |
| Buffer of Thoughts                   | 2024         | likely available / paper repo | Stores high-level thought templates in a meta-buffer and retrieves/adapts them for reasoning.([arXiv][23]) | Very close to reusable reasoning templates/memory | GIFT must be executable typed dataflow, not text thought retrieval                   | High         | Same tasks, same model, compare BoT-style retrieval vs executable dataflow plans |
| PAL: Program-Aided Language Models   | 2022/2023    | official code exists          | LM writes programs, interpreter executes.([arXiv][24])                                                     | Executable math reasoning                         | GIFT reuses mined subroutines with explicit interface-flow, not one-off full program | High         | Compare against PAL with official prompt/code and matched token budget           |
| Faithful Chain-of-Thought            | 2023         | paper/code likely             | NL reasoning translated into symbolic chain executed by solver.([arXiv][25])                               | Faithfulness + symbolic execution                 | GIFT adds reusable library and MCD over dataflow motifs                              | Medium-high  | Faithfulness metrics plus compositional OOD split                                |
| Self-Discover                        | 2024         | likely                        | LLM composes atomic reasoning modules into task-specific structures.([arXiv][26])                          | Self-composed reasoning modules                   | GIFT modules are executable with typed dataflow and trained bindings                 | Medium-high  | Compare generated reasoning structures vs executable dataflow calls              |
| Tree of Thoughts                     | 2023         | yes                           | Search over intermediate thoughts.([arXiv][27])                                                            | Search-time reasoning                             | GIFT is not search alone; it is grounded executable planning                         | Medium       | Include ToT-style search baseline if compute feasible                            |
| Graph of Thoughts                    | 2023         | yes                           | Graph-structured LLM reasoning units.([arXiv][28])                                                         | Graph of reasoning dependencies                   | GIFT graph nodes are typed executable subroutines with verified bindings             | Medium-high  | Compare graph reasoning text vs executable typed graph                           |
| LILO / library learning              | 2023/2024    | yes                           | Iterative library learning and compression for code/program synthesis.([arXiv][29])                        | Library induction / reusable abstractions         | GIFT focuses on NL math grounding and explicit dataflow evaluation                   | High         | Show benefit from grounded binding, not only compression                         |
| Retrieval-of-Thought                 | 2025         | OpenReview/arXiv              | Retrieves previous reasoning as composable thought graph with semantic/sequential edges.([arXiv][30])      | Very close to composable thought reuse            | GIFT must emphasize executable DSL, typed interfaces, active-binding verification    | High         | Same retrieval budget; compare thought graph vs executable dataflow graph        |
| Learning Composable Chain-of-Thought | 2025         | arXiv                         | Trains models on composable CoT format for atomic tasks.([arXiv][31])                                      | Compositional CoT supervision                     | GIFT differs by program execution and library interfaces                             | Medium-high  | Use composable CoT baseline and compare OOD dataflow MCD                         |
| CFQ / MCD split methodology          | 2020         | dataset/tooling               | Compound divergence with atom distribution control.([OpenReview][32])                                      | MCD evaluation method                             | GIFT must define compounds over true dataflow motifs                                 | Medium       | Report atom TVD, compound divergence, and dataflow motif unseen rate             |

## 12.2 Novelty risk verdict

Novelty risk is **high** if the paper says “we mine templates and reuse them.” That is too close to Buffer of Thoughts, PAL, Faithful CoT, LILO/DreamCoder-style library learning, and newer composable thought retrieval.

The defensible novelty route is narrower and sharper:

> Existing reusable-reasoning methods retrieve text thoughts or generate full programs, but they do not train an executable planner over mined subroutines with explicit grounded interface-flow supervision and evaluate OOD generalization over true dataflow compounds.

Claims that must **not** be made now:

* “SOTA”
* “NeurIPS-level proven”
* “template reuse alone solves MCD”
* “verified program extraction implies faithful composition”
* “SEVAL RLVR improves MATH by +10”
* “dynamic memory improves accuracy” based on current pilot

Claims that can be made **only if experiments pass**:

* GIFT improves method-only accuracy over old compose, faithful flat, and matched CoT/PAL-style baselines.
* Explicit dataflow binding is causal, shown by A/B/C ablations.
* True dataflow MCD split reveals failures hidden by adjacency-based compounds.
* Active-binding faithfulness correlates with OOD accuracy.

---

# 13. Keep / Delete / Rewrite / Archive Plan

| Item                              | Type             | File / Directory / Claim / Experiment                   | Current Role              | Problem Under New MAIN PATH             | Action                                              | Reason                                             |
| --------------------------------- | ---------------- | ------------------------------------------------------- | ------------------------- | --------------------------------------- | --------------------------------------------------- | -------------------------------------------------- |
| DSL primitives                    | Code             | `src/template_dsl.py` `Program`, `Step`, basic executor | executable foundation     | useful but composition semantics flawed | KEEP + REWRITE composition parts                    | Keep verified execution, replace plan semantics    |
| CompositionExecutor               | Code             | `src/template_dsl.py`                                   | main composition executor | implicit binding                        | REWRITE                                             | Needs explicit `BindingRef`                        |
| CompositionPlan schema            | Code/data schema | `src/template_dsl.py`, train JSON                       | plan format               | allows empty bindings                   | REWRITE                                             | Must encode quantity/call refs                     |
| `inline_program`                  | Code             | `src/template_dsl.py`                                   | flat baseline conversion  | may inline wrong/misbound templates     | REWRITE                                             | Preserve faithful dataflow or use original program |
| verified full programs            | Data             | `results/templates_verified/all_programs*`              | source programs           | provenance incomplete but valuable      | KEEP                                                | Use as input after audit                           |
| old compose labels                | Data             | `compose_train.json`                                    | SFT target                | unfaithful empty binding                | ARCHIVE / KEEP ONLY AS HISTORICAL NEGATIVE EVIDENCE | Do not train main method on it                     |
| old flat labels                   | Data             | `flat_train.json`                                       | baseline target           | contaminated                            | ARCHIVE / REWRITE                                   | Rebuild fair flat                                  |
| subroutine library                | Data             | `subroutine_library.json`                               | 16 templates              | weak interfaces                         | MERGE INTO NEW METHOD after audit                   | Recompute interface signatures                     |
| MCD split code                    | Code             | `src/mcd_split.py`                                      | split generation          | fake flow edges                         | REWRITE                                             | Compounds from explicit dataflow                   |
| MCD split 27B                     | Data             | `results/mcd_split_27b.json`                            | earlier artifact          | synthetic_used, bad atom_tvd            | ARCHIVE                                             | Not main evidence                                  |
| MCD split v2                      | Data             | `results/mcd_split_v2.json`                             | better split              | compounds semantically weak             | FREEZE then REWRITE                                 | Use only after dataflow re-split                   |
| eval seed42 result                | Result           | `results/eval_v2/eval_results_seed42.json`              | current evidence          | negative, incomplete provenance         | KEEP ONLY AS HISTORICAL NEGATIVE EVIDENCE           | Important diagnostic                               |
| pilot results                     | Result           | `pilot_results*.json`                                   | memory result             | low reproducibility, no acc gain        | KEEP ONLY AS HISTORICAL NEGATIVE/MIXED EVIDENCE     | Shows reuse ≠ correctness                          |
| Pipeline 4% result                | Result/report    | `PIPELINE_REPORT.md`                                    | weak positive             | synthetic, N=50                         | ARCHIVE                                             | Do not use as main positive                        |
| training script                   | Code             | `scripts/train_template_compiler.py`                    | SFT training              | seed/resume/val issues                  | REWRITE                                             | Add reproducibility and GIFT schema                |
| eval script                       | Code             | `scripts/eval_template_reasoning.py`                    | evaluation                | fallback/checkpoint metric risks        | REWRITE                                             | Method-only primary                                |
| SEVAL RLVR                        | Code             | `src/rlvr_evolution.py`                                 | future extension          | built on weak semantics                 | KEEP ONLY AS ABLATION                               | Resume after GIFT core                             |
| test-time tools                   | Code             | `src/test_time_tools.py`                                | repair                    | heuristic/leakage risk                  | KEEP ONLY AS ABLATION                               | Convert later to dataflow gap repair               |
| Template Algebra proposal         | Claim/docs       | `PROPOSAL.md`                                           | broad method claim        | not implemented                         | FREEZE / WEAKEN                                     | Do not claim until implemented                     |
| README strong claims              | Claim/docs       | `README.md`                                             | public thesis             | unsupported                             | REWRITE                                             | Replace with reproducible status                   |
| `PAPERS.md` placeholder citations | Docs             | `PAPERS.md`                                             | related work              | fake/placeholder IDs                    | DELETE or MARK UNVERIFIED                           | Academic integrity                                 |
| tests                             | Code             | `tests/test_templatebank.py`                            | current unit tests        | encode wrong composition behavior       | REWRITE / ADD TESTS                                 | Need explicit dataflow tests                       |
| result aggregation scripts        | Code             | `collect_results.sh`, eval aggregation                  | tables                    | likely incomplete seed/CI               | REWRITE                                             | Add CI/std/hash                                    |
| `src/template_algebra.py`         | Code             | legacy algebra                                          | unclear main              | likely dead/aspirational                | ARCHIVE or KEEP ONLY AS HISTORICAL                  | Avoid confusing mainline                           |

---

# 14. Claude Code Implementation Plan

## Task 1: Freeze and label unreliable old evidence

**Purpose:** Prevent old weak/contaminated results from shaping the new method.
**Which Phenomenon / Constraint It Addresses:** PHE-11, PHE-18, C-08, C-14.
**Why It Supports New MAIN METHOD PATH:** GIFT must be evaluated cleanly, not inherited from synthetic/empty-binding claims.
**Files to Inspect:** `README.md`, `PIPELINE_REPORT.md`, `IDEA_REPORT.md`, `PROPOSAL.md`, `results/*`, root pilot JSONs.
**Files to Edit:** Add `archive/README_unreliable_results.md`, optionally move or copy manifests only.
**Files to Delete / Archive:** Archive metadata for `results/extract_27b_500`, old pilot, old eval; do not delete raw files silently.
**Functions / Classes:** N/A.
**Exact Change:** Add a manifest table labeling each old result as `historical`, `unreliable`, `negative`, or `diagnostic`; add warning comments in README.
**Do Not Change:** Do not modify raw JSON result values.
**Verification Command:**

```bash
python - <<'PY'
from pathlib import Path
assert Path("archive/README_unreliable_results.md").exists()
print("archive manifest exists")
PY
```

**Expected Result:** Clear provenance status for old artifacts.
**Failure Means:** Old claims may continue contaminating future decisions.
**Rollback Condition:** If archive path breaks scripts, revert moves and keep manifest-only labels.
**Priority:** P0
**Confidence:** high

---

## Task 2: Add failing tests for explicit dataflow composition

**Purpose:** Make the current missing mechanism testable.
**Which Phenomenon / Constraint It Addresses:** PHE-14, C-03.
**Why It Supports New MAIN METHOD PATH:** GIFT requires explicit previous-output binding.
**Files to Inspect:** `tests/test_templatebank.py`, `src/template_dsl.py`.
**Files to Edit:** `tests/test_dataflow_plan.py`.
**Exact Change:** Add tests where ADD output must feed MUL input:

```python
# expected: (2 + 3) * 4 = 20
# old implicit slot-name behavior should not count as success
```

**Do Not Change:** Do not weaken existing tests yet; mark incompatible tests after new semantics added.
**Verification Command:**

```bash
pytest -q tests/test_dataflow_plan.py
```

**Expected Result:** Initially fails before Task 3, passes after Task 3.
**Failure Means:** New mechanism is not implemented.
**Rollback Condition:** None; tests define required semantics.
**Priority:** P0
**Confidence:** high

---

## Task 3: Implement explicit `BindingRef`, `DataflowPlan`, and `DataflowExecutor`

**Purpose:** Replace implicit env binding with explicit interface-flow execution.
**Which Phenomenon / Constraint It Addresses:** PHE-01, PHE-02, PHE-14, C-02, C-03, C-04.
**Files to Inspect:** `src/template_dsl.py`.
**Files to Edit:** Prefer new `src/dataflow_plan.py`; minimally edit `src/template_dsl.py` to interoperate.
**Exact Change:** Add:

* `BindingRef(source: Literal["quantity","call_output","constant"], id/call_id/value, dtype)`
* `PlanCall(call_id, sub_id, bindings)`
* `DataflowPlan(calls, final)`
* `DataflowExecutor.execute(plan, quantities, library)`
* strict typecheck: all required slots covered, no implicit env fallback.
  **Do Not Change:** Do not change primitive `Program` execution semantics unless tests require.
  **Verification Command:**

```bash
pytest -q tests/test_dataflow_plan.py tests/test_templatebank.py
```

**Expected Result:** New dataflow tests pass; old tests either pass or are explicitly updated with migration note.
**Failure Means:** Core GIFT cannot proceed.
**Rollback Condition:** If primitive executor breaks verified programs, isolate new executor in separate file.
**Priority:** P0
**Confidence:** high

---

## Task 4: Build plan faithfulness auditor

**Purpose:** Filter out empty-binding and semantically mismatched plans.
**Which Phenomenon / Constraint It Addresses:** PHE-06, PHE-07, C-02.
**Files to Inspect:** `scripts/extract_templates.py`, `results/templates_verified/plans_with_programs.json`.
**Files to Edit:** Add `scripts/audit_dataflow_plans.py` and import GIFT executor.
**Exact Change:** For each candidate plan, log:

* `slot_coverage`
* `typecheck_ok`
* `exec_ok`
* `answer_correct`
* `active_binding_rate`
* `true_multicall_flow`
* `empty_binding_rate`
  **Do Not Change:** Do not overwrite existing result files.
  **Verification Command:**

```bash
python scripts/audit_dataflow_plans.py \
  --input results/templates_verified/plans_with_programs.json \
  --library results/templates_verified/subroutine_library.json \
  --output results/debug_gift/plan_audit.json \
  --max_examples 50
```

**Expected Result:** Audit exposes current old plans as mostly unfaithful/empty-binding.
**Failure Means:** Either data schema unknown or executor mismatch; stop and inspect.
**Rollback Condition:** None.
**Priority:** P0
**Confidence:** high

---

## Task 5: Regenerate faithful GIFT training data

**Purpose:** Replace bad supervision.
**Which Phenomenon / Constraint It Addresses:** PHE-06, PHE-07, PHE-08.
**Files to Inspect:** `scripts/extract_templates.py`.
**Files to Edit:** `scripts/extract_templates.py` or new `scripts/build_gift_data.py`.
**Exact Change:** Generate:

* `results/gift/compose_train_gift.json`
* `results/gift/flat_train_faithful.json`
* `results/gift/library_gift.json`
* `results/gift/plan_audit.json`
  Only include examples whose explicit dataflow plan executes to the original answer.
  **Do Not Change:** Do not delete old `compose_train.json`.
  **Verification Command:**

```bash
python scripts/build_gift_data.py \
  --programs results/templates_verified/all_programs.json \
  --library results/templates_verified/subroutine_library.json \
  --output_dir results/gift \
  --max_examples 200 \
  --require_active_binding
```

**Expected Result:** Nonzero faithful coverage with audit stats.
**Failure Means:** If faithful coverage <30%, current library mining is unusable; pivot to mining smaller step-level primitives.
**Rollback Condition:** Keep old data archived, do not train GIFT.
**Priority:** P0
**Confidence:** medium

---

## Task 6: Rewrite MCD split over true dataflow motifs

**Purpose:** Make MCD evaluate real composition.
**Which Phenomenon / Constraint It Addresses:** PHE-12, PHE-13, C-05.
**Files to Inspect:** `src/mcd_split.py`.
**Files to Edit:** `src/mcd_split.py`, possibly `scripts/build_mcd_split.py`.
**Exact Change:** Compounds are:

* subroutine pair with actual `call_output` edge,
* slot-role flow motif,
* final-output dependency motif.
  No edge means no flow compound.
  **Do Not Change:** Do not report adjacency compounds as MCD flow.
  **Verification Command:**

```bash
python scripts/build_mcd_split.py \
  --plans results/gift/compose_train_gift.json \
  --output results/gift/mcd_split_gift.json \
  --max_atom_tvd 0.02 \
  --seed 42
```

**Expected Result:** atom_tvd logged, unseen true-flow compounds logged.
**Failure Means:** Not enough true dataflow motifs; library/plans too shallow.
**Rollback Condition:** Use IID split only for debugging; no MCD claim.
**Priority:** P0
**Confidence:** high

---

## Task 7: Fix evaluation metrics and checkpoint loading

**Purpose:** Prevent contaminated method claims.
**Which Phenomenon / Constraint It Addresses:** PHE-01, PHE-18, C-06, C-09.
**Files to Inspect:** `scripts/eval_template_reasoning.py`.
**Files to Edit:** `scripts/eval_template_reasoning.py`.
**Exact Change:** Add:

* `--require_adapters`
* `--no_fallback`
* `method_accuracy`
* `fallback_rescued_accuracy`
* `combined_accuracy_not_main`
* checkpoint path/hash
* data hash
* seed
* environment info
  **Do Not Change:** Do not silently change datasets/metrics.
  **Verification Command:**

```bash
python scripts/eval_template_reasoning.py \
  --config configs/template_config.yaml \
  --methods compose \
  --max_samples 10 \
  --no_fallback \
  --require_adapters \
  --output_dir results/debug_eval
```

**Expected Result:** Fails loudly if adapter missing; otherwise logs method-only metrics.
**Failure Means:** Current eval is not trustworthy.
**Rollback Condition:** Keep old script as `eval_template_reasoning_legacy.py`.
**Priority:** P0
**Confidence:** high

---

## Task 8: Fix training reproducibility

**Purpose:** Make multi-seed and checkpoint provenance real.
**Which Phenomenon / Constraint It Addresses:** PHE-18, C-09.
**Files to Inspect:** `scripts/train_template_compiler.py`.
**Files to Edit:** same.
**Exact Change:** Add explicit `--seed`, `--no_resume`, `--resume_from`, val split, data hash, checkpoint manifest, exact command logging.
**Do Not Change:** Do not tune hyperparameters for positive results before sanity checks.
**Verification Command:**

```bash
python scripts/train_template_compiler.py \
  --config configs/template_config.yaml \
  --train_file results/gift/compose_train_gift.json \
  --mode gift \
  --seed 42 \
  --no_resume \
  --max_steps 5 \
  --output_dir results/debug_train_gift_seed42
```

**Expected Result:** Writes manifest with seed/data hash/checkpoint hash.
**Failure Means:** Reproducibility not fixed.
**Rollback Condition:** None.
**Priority:** P1
**Confidence:** medium-high

---

## Task 9: Add minimal GIFT config switches

**Purpose:** Keep new path isolated and ablatable.
**Files to Inspect:** `configs/template_config.yaml`.
**Files to Edit:** `configs/gift_minimal.yaml`.
**Exact Change:** Add switches:

```yaml
method: gift
plan_schema: dataflow_v1
require_explicit_bindings: true
require_active_binding: true
fallback_policy: report_only
mcd_compounds: true_dataflow
```

**Do Not Change:** Existing configs except adding deprecation notes.
**Verification Command:**

```bash
python - <<'PY'
import yaml
cfg=yaml.safe_load(open("configs/gift_minimal.yaml"))
assert cfg["method"]=="gift"
assert cfg["require_explicit_bindings"] is True
print("gift config ok")
PY
```

**Expected Result:** config parses.
**Priority:** P1
**Confidence:** high

---

## Task 10: Run minimal A/B/C verification queue

**Purpose:** Test whether GIFT adds causal benefit beyond old positive fragments.
**Files to Inspect:** all above.
**Files to Edit:** result aggregation script.
**Verification Command:**

```bash
bash scripts/run_gift_minimal_ablation.sh \
  --config configs/gift_minimal.yaml \
  --seeds 42 123 456 \
  --max_train 200 \
  --max_eval 100
```

**Expected Result:** Table comparing:
A. Existing Best Positive Fragment Only
B. GIFT Without New Mechanism
C. Full GIFT
**Failure Means:** If C does not beat A/B or only wins with fallback, do not continue to full benchmark.
**Rollback Condition:** Stop and inspect dataflow coverage.
**Priority:** P1
**Confidence:** medium

---

# 15. Minimal Verification Experiments

| Priority | Experiment                               | Hypothesis                                     | Command                                                                                                                                                            | Config           | Dataset            | Seeds      | Metric                         | Success Criterion                                             | Failure Interpretation          |
| -------: | ---------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- | ------------------ | ---------- | ------------------------------ | ------------------------------------------------------------- | ------------------------------- |
|        0 | Smoke test                               | repo imports and core tests run                | `pytest -q tests`                                                                                                                                                  | N/A              | N/A                | N/A        | pass/fail                      | all non-legacy tests pass                                     | codebase not stable             |
|        0 | Data sanity check                        | main data is non-synthetic and hashable        | `python scripts/audit_dataflow_plans.py --input results/templates_verified/plans_with_programs.json --output results/debug_gift/plan_audit.json --max_examples 50` | N/A              | templates_verified | N/A        | synthetic flag, empty bindings | synthetic false for main; empty-binding rate logged           | data artifact unusable          |
|        0 | Metric sanity check                      | method accuracy separable from fallback        | `python scripts/eval_template_reasoning.py --config configs/template_config.yaml --max_samples 10 --no_fallback --require_adapters`                                | template_config  | GSM8K small        | 42         | method vs fallback             | script fails if adapter missing; no combined main metric      | old eval unreliable             |
|        0 | One-batch overfit                        | planner can learn GIFT schema                  | `python scripts/train_template_compiler.py --mode gift --train_file results/gift/compose_train_gift.json --overfit_one_batch --seed 42`                            | gift_minimal     | 1 batch            | 42         | exact plan match / exec acc    | >95% exact or executable match                                | schema/training broken          |
|        0 | Checkpoint loading check                 | eval uses intended adapter                     | `python scripts/eval_template_reasoning.py --require_adapters --print_checkpoint_hash ...`                                                                         | gift_minimal     | debug              | 42         | hash logged                    | adapter path/hash present                                     | stale/missing checkpoint        |
|        1 | Reproduce current negative result        | old compose failure is real enough to diagnose | `python scripts/eval_template_reasoning.py --legacy --config configs/template_config.yaml --seed 42 --output_dir results/repro_old_seed42`                         | legacy           | GSM8K/MCD small    | 42         | acc/exec                       | roughly matches 2/100,0/100 within sample noise               | old artifact not reproducible   |
|        1 | Reproduce current best positive fragment | old best signal is not ignored                 | `bash scripts/run_legacy_best_fragment.sh --seed 42 --max_eval 100`                                                                                                | legacy           | pilot/GSM8K        | 42         | acc/reuse                      | logs old reuse/weak acc                                       | cannot compare A                |
|        1 | New mechanism activation check           | GIFT creates real dataflow                     | `python scripts/build_gift_data.py --max_examples 200 --require_active_binding`                                                                                    | gift_minimal     | verified programs  | N/A        | true_flow_rate, active_binding | true_flow_rate >30%, active_binding >70%                      | library/plan mining too shallow |
|        1 | New MAIN METHOD minimal test             | Full GIFT can solve small heldout              | `bash scripts/run_gift_minimal.sh --seeds 42 123 456 --max_train 200 --max_eval 100`                                                                               | gift_minimal     | GSM8K heldout      | 42/123/456 | method_acc                     | mean > old compose and faithful flat debug baseline           | no initial benefit              |
|        1 | Ablation: remove dataflow                | explicit dataflow is causal                    | `bash scripts/run_gift_ablation.sh --variant no_call_output_refs ...`                                                                                              | gift_no_flow     | same               | 42/123/456 | method_acc                     | Full GIFT > no_flow                                           | dataflow not causal             |
|        1 | A. Existing Best Positive Fragment Only  | old positive fragment alone insufficient       | `bash scripts/run_gift_ablation.sh --variant old_fragment_only ...`                                                                                                | old_fragment     | same               | 42/123/456 | method_acc                     | Full GIFT > A                                                 | new method just old fragment    |
|        1 | B. GIFT Without New Mechanism            | architecture without binding not enough        | `bash scripts/run_gift_ablation.sh --variant gift_no_faithfulness ...`                                                                                             | gift_no_mech     | same               | 42/123/456 | method_acc, active_binding     | Full GIFT > B and active_binding higher                       | gains not from mechanism        |
|        1 | C. Full New MAIN METHOD                  | full mechanism works                           | `bash scripts/run_gift_ablation.sh --variant full_gift ...`                                                                                                        | gift_minimal     | same               | 42/123/456 | method_acc                     | C best or statistically tied with best but more faithful      | insufficient method evidence    |
|        2 | Small baseline comparison                | GIFT beats simple strong baselines             | `bash scripts/run_small_baselines.sh --methods faithful_flat,direct_cot,cot_budget,gift`                                                                           | gift_minimal     | GSM8K small        | 42/123/456 | acc/tokens                     | GIFT improves method_acc or token-efficiency without fallback | baseline stronger               |
|        2 | Multi-seed stability                     | gain is not seed cherry-pick                   | same aggregation                                                                                                                                                   | same             | same               | 42/123/456 | mean/std/CI                    | CI of C-A > 0 or meaningful effect                            | unstable                        |
|        2 | Expansion gate to full benchmark         | small success generalizes                      | `bash scripts/run_full_gift_gate.sh --dry_run && bash ...`                                                                                                         | gift_full        | GSM8K/MCD full     | 3+         | acc/CI                         | only run if P0/P1 pass                                        | premature full benchmark        |
|        2 | Official baseline reproduction           | compare fairly                                 | `bash scripts/run_official_baselines.sh --baselines pal,bot,faithful_cot`                                                                                          | baseline configs | same               | 3+         | acc/tokens                     | official or close reproduction                                | cannot claim superiority        |
|        2 | Unified environment comparison           | no environment mismatch                        | `python scripts/collect_env_and_results.py`                                                                                                                        | all              | all                | all        | env hash                       | same env/model/data hash                                      | unfair comparison               |
|        3 | Robustness/generalization                | GIFT works beyond IID                          | `bash scripts/run_gift_robustness.sh --splits iid,mcd_trueflow`                                                                                                    | gift_full        | true-flow MCD      | 3+         | acc/active_binding             | MCD drop smaller than baselines                               | not compositional               |
|        3 | Statistical significance                 | avoid cherry-picking                           | `python scripts/bootstrap_results.py --input results/gift/*.json`                                                                                                  | all              | all                | 3+         | CI/p-value/effect              | paired bootstrap supports main claim                          | weak evidence                   |

---

# 16. Baseline and SOTA Plan

| Baseline                             | Why Required                       | Official Code           | Dataset                       | Metric           | Reproduction Requirement                     | Fairness Risk                                            |
| ------------------------------------ | ---------------------------------- | ----------------------- | ----------------------------- | ---------------- | -------------------------------------------- | -------------------------------------------------------- |
| Direct CoT                           | simplest standard baseline         | prompt baseline enough  | GSM8K/MATH/MCD                | accuracy/tokens  | same model, same decoding budget             | weak prompt can understate baseline                      |
| cot_budget                           | controls token budget              | repo                    | GSM8K/MCD                     | acc/tokens       | same model and budget                        | current cot_budget already beats compose in seed42 GSM8K |
| Faithful flat inline                 | controls full program generation   | repo, rebuilt           | GSM8K/MCD                     | method_acc/exec  | use original verified full programs          | old flat labels contaminated                             |
| Existing Best Positive Fragment Only | proves not old path                | repo legacy             | same                          | acc/reuse        | freeze old implementation                    | may be unfair if old artifact broken; still diagnostic   |
| PAL                                  | executable program baseline        | official code exists    | GSM8K/MATH                    | accuracy/exec    | official prompt/code, same model if possible | different interpreter/prompt budget                      |
| Faithful CoT                         | symbolic faithful baseline         | paper/code if available | GSM8K/MATH                    | acc/faithfulness | reproduce official setup                     | solver/dataset mismatch                                  |
| Buffer of Thoughts                   | reusable thought-template memory   | paper/code if available | GSM8K/MATH/BBH                | acc/tokens       | official or faithful reproduction            | very close mechanism; must compare                       |
| Self-Discover                        | compositional reasoning modules    | official if available   | BBH/GSM8K                     | acc/tokens       | official prompt + same model                 | prompt engineering differences                           |
| Tree/Graph of Thoughts               | search/graph reasoning             | official if feasible    | Game24/BBH/GSM8K              | acc/compute      | matched compute budget                       | compute unfairness                                       |
| LILO/library induction               | mechanism-level library baseline   | official if feasible    | program synthesis/math subset | acc/reuse        | may need task adaptation                     | not same task; compare mechanism carefully               |
| Retrieval-of-Thought                 | closest composable reasoning reuse | if available            | reasoning datasets            | acc/reuse        | official or reimplementation                 | high novelty overlap                                     |

---

# 17. Paper Thesis Reconstruction

1. **New Paper Thesis:**
   Reusable reasoning templates only help compositional generalization when their interfaces are explicitly grounded and their outputs are connected through verified dataflow; GIFT turns mined executable programs into faithful interface-flow plans and evaluates them on true dataflow MCD splits.

2. **Main Technical Contribution:**
   A dataflow-grounded template composition framework: typed subroutine interfaces, explicit quantity/output bindings, faithfulness auditing, and MCD evaluation over true flow motifs.

3. **Main Empirical Claim:**
   Only if experiments pass: Full GIFT improves method-only accuracy and stability over old template reuse, faithful flat inline programs, and matched CoT/PAL/BoT-style baselines on IID and true-flow MCD splits.

4. **What Previous Failures Taught Us:**
   High execution validity is not enough; ungrounded template IDs and empty bindings can execute while solving the wrong problem.

5. **What We Should Not Claim:**
   SOTA, +15%, +25% repair, SEVAL +10 MATH, dynamic memory accuracy gains, or broad “template algebra” unless verified.

6. **What We Can Claim If Experiments Pass:**
   Explicit interface-flow binding is a causal mechanism for compositional generalization, supported by A/B/C ablations and active-binding metrics.

7. **Required Baselines:**
   Direct CoT, cot_budget, faithful flat, old compose, PAL, Faithful CoT, BoT, Self-Discover/ToT/GoT where appropriate.

8. **Required Ablations:**
   no dataflow refs, no active-binding gate, no compression prior, old fragment only, faithful flat, random library.

9. **Required Robustness Tests:**
   true-flow MCD, IID-to-MCD transfer, seed stability, model-size transfer, DSL category stress tests.

10. **Reviewer Likely Objections:**
    “This is PAL + templates,” “BoT already stores templates,” “MCD split artificial,” “baselines weak,” “program extraction labels are noisy,” “novelty is engineering.”

11. **How GIFT Answers Them:**
    By showing dataflow-specific causal ablations, true-flow MCD metrics, official baselines, and active-binding faithfulness.

12. **What Would Make This NeurIPS-Strong:**
    Clean 3+ seed results, official baselines, clear novelty vs BoT/PAL/Faithful CoT, reproducible scripts, meaningful OOD gains, negative results disclosed.

13. **What Would Make This Rejected:**
    Same weak seed42 results, unsupported claims, fake related work, no official baselines, fallback-masked accuracy, or no proof that dataflow mechanism causes gains.

14. **What Would Be Required for Oral-Level Strength:**
    Strong multi-dataset evidence that explicit interface-flow is a general principle across models/tasks, plus compelling mechanistic analysis.

15. **What Would Be Required for Best-Paper-Level Strength:**
    A broadly applicable abstraction-learning result with strong theory or scaling law, robust cross-domain transfer, and a benchmark-quality true-flow MCD dataset adopted beyond this repo.

---

# 18. Reviewer Risk Assessment

| Risk                       | Why Reviewer May Object                            | Evidence Needed                       | How GIFT Addresses It                                | Remaining Weakness                  |
| -------------------------- | -------------------------------------------------- | ------------------------------------- | ---------------------------------------------------- | ----------------------------------- |
| Novelty risk               | Templates/program execution/memory already studied | Related work and mechanism comparison | Focus on explicit grounded dataflow + active-binding | Still close to PAL/BoT/Faithful CoT |
| Incremental risk           | Could look like engineering cleanup                | Causal ablation                       | A/B/C shows dataflow mechanism matters               | If gains small, rejected            |
| Baseline weakness          | Current baselines very weak                        | Official baselines                    | Add PAL/BoT/Faithful CoT                             | Compute burden                      |
| Reproducibility            | Missing checkpoints/logs                           | command/config/hash/seed              | Add manifests and CI                                 | Needs local rerun                   |
| Cherry-picking             | only seed42 visible                                | 3+ seeds, all datasets                | Required aggregation                                 | Compute                             |
| Negative result hiding     | many negative docs exist                           | disclose diagnostic failures          | Keep archive, explain pivot                          | Need careful writing                |
| Overclaiming               | README/proposal too strong                         | weakened claims                       | rewrite thesis                                       | Must resist “SOTA”                  |
| Unclear mechanism          | current compose semantics weak                     | binding/active logs                   | explicit DataflowPlan                                | Need readable examples              |
| Ablation insufficiency     | old ablations not causal                           | A/B/C + no-gate/no-flow               | direct mechanism tests                               | Coverage may be low                 |
| Dataset limitation         | GSM8K-only risk                                    | MATH/BBH/Game24 if DSL supports       | start with true-flow MCD                             | DSL may not handle MATH             |
| Compute unfairness         | search baselines costly                            | matched budget logs                   | report tokens/latency                                | Hard to compare search              |
| Implementation reliability | current bugs likely                                | unit tests and audits                 | fail-hard eval                                       | Need full local validation          |
| Related work omission      | `PAPERS.md` placeholders                           | verified bibliography                 | cite real closest papers                             | Must remove fake refs               |

---

# 19. Final Decision

## 1. One-Sentence Verdict

从所有现象推导出的唯一主线应是 **GIFT: Grounded Interface-Flow Template Composition**：把当前模板调用从 ungrounded template-ID selection 改造成显式数量绑定、显式 output-to-input dataflow、可执行且可扰动验证的 faithful plan DAG。

## 2. Current Most Likely Root Cause

最可能根因不是单个调参问题，而是组合了：

* **missing mechanism:** 缺少 grounded interface-flow binding；
* **data/label pipeline bug:** compose/flat training labels 不 faithful；
* **evaluation reliability bug:** fallback、adapter/checkpoint、seed/logging 不足；
* **method assumption failure:** “verified full program + compressed template = compositional abstraction” 这个假设不成立；
* **insufficient evidence:** 没有多 seed、official baselines、完整 checkpoint/log。

## 3. Why This Is Not Just the Existing Best Path

旧路径复用模板 ID；GIFT 复用的是**显式接口与数据流机制**。旧路径可以输出 `bindings:{}`；GIFT 要求每个 slot 绑定到问题 quantity 或前序 call output。旧路径 MCD 可以由 call adjacency 构造；GIFT 只用真实 dataflow motifs。旧路径可被 fallback 混合；GIFT primary metric 是 method-only accuracy。

## 4. Phenomena Explained

GIFT 解释：

* 697 verified programs 为什么有价值但不足；
* compose valid/exec 高但 accuracy 低；
* MCD compose 0/100；
* flat baseline 也被污染；
* dynamic memory reuse 提升但 acc 不提升；
* SEVAL evolution 在当前 semantics 上为何 vacuous；
* MCD split 数值好但语义不可靠。

## 5. Mechanism Missing in Current Method

**显式 grounded interface-flow binding and faithfulness verification.**

## 6. New Mechanism

* `BindingRef(quantity | call_output | constant)`
* `DataflowPlan`
* `DataflowExecutor`
* active-binding perturbation audit
* true-flow MCD split
* method-only evaluation

## 7. What to Delete / Archive / Rewrite

* Archive: synthetic 27B split, pilot weak result, pipeline 4% synthetic report as historical evidence.
* Rewrite: CompositionExecutor, plan labels, flat labels, MCD compounds, eval metrics, training reproducibility.
* Freeze: SEVAL/test-time tools until GIFT core passes.
* Delete/mark unverified: placeholder citations and unsupported README/idea claims.
* Keep: verified full programs, DSL primitives, baseline scaffolds.

## 8. First Five Claude Code Tasks

1. Freeze/archive unreliable claims and result artifacts with manifest.
2. Add explicit dataflow composition tests.
3. Implement `BindingRef`, `DataflowPlan`, `DataflowExecutor`.
4. Build plan faithfulness auditor for old and new labels.
5. Regenerate faithful GIFT training data and fair flat baseline.

## 9. Minimal Experiments

1. Smoke tests.
2. Data sanity and empty-binding audit.
3. Metric/fallback sanity.
4. One-batch overfit.
5. Checkpoint loading check.
6. Reproduce old negative.
7. Reproduce old best positive fragment.
8. GIFT mechanism activation.
9. A/B/C minimal comparison.
10. 3-seed stability.
11. Small official baseline comparison.
12. true-flow MCD expansion gate.

## 10. Continue / Stop / Pivot Criteria

**Continue if:**

* faithful GIFT coverage is nontrivial, ideally >30%;
* one-batch overfit passes;
* active-binding rate >70%;
* Full GIFT beats both old fragment and no-mechanism ablation on method-only accuracy across 3 seeds.

**Stop if:**

* GIFT cannot overfit one batch;
* explicit dataflow plans rarely execute;
* Full GIFT only wins when fallback is counted;
* C does not beat A/B.

**Pivot if:**

* faithful dataflow coverage is too low because library mining is too coarse;
* official PAL/BoT/Faithful CoT dominates at matched compute;
* DSL cannot cover target datasets beyond GSM8K.

## 11. NeurIPS-Level Gap

Current gap is large: clean dataflow labels, reliable evaluation, official baselines, multi-seed results, novelty differentiation, and rewritten claims are all required.

## 12. Oral / Best Paper Gap

Oral-level would require broad evidence that explicit grounded dataflow is a general reusable-reasoning principle across datasets/models. Best-paper-level would require a much deeper scientific contribution: strong theory or scaling evidence, new benchmark quality, and robust transfer beyond arithmetic word problems.

## 13. Confidence

* **High confidence** in the diagnosis that current composition lacks explicit grounded dataflow.
* **Medium confidence** that GIFT is the right next path.
* **Low-to-medium confidence** that it will produce NeurIPS-strength results without substantial new engineering and clean experiments.

---

# 20. Final Claude Code Instruction

```text
Claude Code, execute the following plan.

You must implement the New MAIN METHOD PATH defined in the GPT-5.5 Pro diagnosis report:

GIFT — Grounded Interface-Flow Template Composition.

Do not invent a different method.
Do not optimize for superficial positive results.
Do not weaken baselines.
Do not delete negative results silently.
Do not change metrics or datasets unless explicitly instructed.
Do not rewrite unrelated files.

Your tasks are:

1. Freeze and label unreliable old evidence.
   - Add archive/README_unreliable_results.md.
   - Mark old synthetic, pilot, seed42, and pipeline artifacts as historical/diagnostic/unreliable where appropriate.
   - Do not modify raw JSON values.

2. Add tests for explicit dataflow composition.
   - Create tests/test_dataflow_plan.py.
   - Include an ADD -> MUL case where the second call must consume the first call output.
   - The expected result must be (2 + 3) * 4 = 20.
   - The test must fail under implicit slot-name binding and pass only with explicit call_output binding.

3. Implement the GIFT dataflow schema.
   - Add BindingRef, PlanCall, DataflowPlan, and DataflowExecutor.
   - Prefer a new src/dataflow_plan.py.
   - Do not break primitive Program/Step execution.
   - DataflowExecutor must reject missing bindings and must not silently use env/name/type fallback.

4. Add a plan faithfulness auditor.
   - Add scripts/audit_dataflow_plans.py.
   - Log slot_coverage, typecheck_ok, exec_ok, answer_correct, empty_binding_rate, true_multicall_flow, and active_binding_rate.
   - Run it first on existing results/templates_verified/plans_with_programs.json.
   - Save results/debug_gift/plan_audit.json.

5. Regenerate faithful GIFT data.
   - Add scripts/build_gift_data.py or extend scripts/extract_templates.py.
   - Output:
     results/gift/compose_train_gift.json
     results/gift/flat_train_faithful.json
     results/gift/library_gift.json
     results/gift/plan_audit.json
   - Include only examples whose explicit dataflow plan executes to the original verified answer.
   - If faithful coverage is below 30%, stop and report.

6. Rewrite MCD compound extraction.
   - Modify src/mcd_split.py so flow compounds come only from explicit call_output binding edges.
   - Do not count consecutive calls as flow.
   - Save results/gift/mcd_split_gift.json with atom_tvd and true-flow unseen compound stats.

7. Fix evaluation reliability.
   - Modify scripts/eval_template_reasoning.py.
   - Add --require_adapters, --no_fallback, checkpoint hash, data hash, seed logging.
   - Report method_accuracy as primary.
   - Report fallback_rescued_accuracy and combined_accuracy_not_main separately.
   - Fail loudly if an adapter/checkpoint is missing unless an explicit --allow_base_model flag is passed.

8. Fix training reproducibility.
   - Modify scripts/train_template_compiler.py.
   - Add --seed, --no_resume, --resume_from, validation split, data hash, command manifest, checkpoint manifest.
   - Do not auto-resume unless explicitly requested.

9. Add configs/gift_minimal.yaml.
   - method: gift
   - plan_schema: dataflow_v1
   - require_explicit_bindings: true
   - require_active_binding: true
   - fallback_policy: report_only
   - mcd_compounds: true_dataflow

10. Run the minimal verification queue only after tasks 1–9 pass.
    - Smoke test.
    - Data sanity check.
    - Metric sanity check.
    - One-batch overfit.
    - Checkpoint loading check.
    - Reproduce current negative seed42 result if possible.
    - Reproduce current best positive fragment if possible.
    - Run A/B/C:
      A. Existing Best Positive Fragment Only
      B. New MAIN METHOD Without New Mechanism
      C. Full New MAIN METHOD

For every task:
- make the smallest necessary change;
- show the diff;
- run the specified verification command;
- save logs;
- report failures;
- stop if verification fails;
- do not proceed to full benchmark until minimal tests pass.

At the end, output:
- files changed;
- files archived;
- configs added;
- commands run;
- logs;
- result table;
- failed checks;
- unresolved issues;
- whether Full New MAIN METHOD beats:
  A. Existing Best Positive Fragment Only,
  B. New MAIN METHOD Without New Mechanism,
  C. Full New MAIN METHOD.
```

[1]: https://github.com/Sunshine535/nips-templatebank "GitHub - Sunshine535/nips-templatebank: TemplateBank++: Dynamic Reasoning Template Memory (NeurIPS 2026) · GitHub"
[2]: https://github.com/Sunshine535/nips-templatebank/blob/main/README.md "nips-templatebank/README.md at main · Sunshine535/nips-templatebank · GitHub"
[3]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/ARIS_REVIEW.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/results/templates_verified/compose_train.json "raw.githubusercontent.com"
[5]: https://github.com/Sunshine535/nips-templatebank/blob/main/EXPERIMENTS.md "nips-templatebank/EXPERIMENTS.md at main · Sunshine535/nips-templatebank · GitHub"
[6]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/AUTO_REVIEW.md "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/PIPELINE_REPORT.md "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/REVIEW_SEVAL_V2.md "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/REVIEW_STATE.json "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/PLAN.md "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/src/template_dsl.py "https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/src/template_dsl.py"
[12]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/src/mcd_split.py "https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/src/mcd_split.py"
[13]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/scripts/extract_templates.py "https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/scripts/extract_templates.py"
[14]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/src/rlvr_evolution.py "raw.githubusercontent.com"
[15]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/src/test_time_tools.py "raw.githubusercontent.com"
[16]: https://github.com/Sunshine535/nips-templatebank/blob/main/results/templates_verified/all_programs_stats.json "nips-templatebank/results/templates_verified/all_programs_stats.json at main · Sunshine535/nips-templatebank · GitHub"
[17]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/results/templates_verified/flat_train.json "raw.githubusercontent.com"
[18]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/results/templates_verified/subroutine_library.json "raw.githubusercontent.com"
[19]: https://github.com/Sunshine535/nips-templatebank/blob/main/results/eval_v2/eval_results_seed42.json "nips-templatebank/results/eval_v2/eval_results_seed42.json at main · Sunshine535/nips-templatebank · GitHub"
[20]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/results/mcd_split_v2.json "raw.githubusercontent.com"
[21]: https://raw.githubusercontent.com/Sunshine535/nips-templatebank/main/results/mcd_split_27b.json "raw.githubusercontent.com"
[22]: https://github.com/Sunshine535/nips-templatebank/blob/main/results/templatebank_pilot_20260227_150036.json "nips-templatebank/results/templatebank_pilot_20260227_150036.json at main · Sunshine535/nips-templatebank · GitHub"
[23]: https://arxiv.org/abs/2406.04271 "https://arxiv.org/abs/2406.04271"
[24]: https://arxiv.org/abs/2211.10435 "https://arxiv.org/abs/2211.10435"
[25]: https://arxiv.org/abs/2301.13379 "https://arxiv.org/abs/2301.13379"
[26]: https://arxiv.org/abs/2402.03620 "https://arxiv.org/abs/2402.03620"
[27]: https://arxiv.org/abs/2305.10601 "https://arxiv.org/abs/2305.10601"
[28]: https://arxiv.org/abs/2308.09687 "https://arxiv.org/abs/2308.09687"
[29]: https://arxiv.org/abs/2310.19791 "https://arxiv.org/abs/2310.19791"
[30]: https://arxiv.org/abs/2509.21743 "https://arxiv.org/abs/2509.21743"
[31]: https://arxiv.org/abs/2505.22635 "https://arxiv.org/abs/2505.22635"
[32]: https://openreview.net/forum?id=SygcCnNKwr "https://openreview.net/forum?id=SygcCnNKwr"
