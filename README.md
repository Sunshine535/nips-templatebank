# Subroutine Composition for Mathematical Reasoning

## 项目简介

从 GSM8K + MATH 数据集中挖掘类型化子程序库 (typed subroutine library)，训练 planner 模型组合子程序解决数学问题。在 CFQ-style Maximum Compound Divergence (MCD) split 上评估，验证可复用子程序能否改善 program-compound generalization。

**Review 状态**: Round 2, Score 3.0/10（需要更多实验数据）

## 环境安装

```bash
cd /workspace/nips-templatebank
python3 -m venv .venv
source .venv/bin/activate
pip install torch
pip install transformers datasets accelerate trl peft evaluate wandb \
    outlines sentence-transformers rank-bm25 numpy scipy matplotlib pandas pyyaml
```

## 快速开始

```bash
source .venv/bin/activate

# Smoke test（合成数据）
SMOKE=1 bash run.sh --smoke

# 单独训练 compose planner
torchrun --nproc_per_node=4 scripts/train_template_compiler.py --mode compose

# 单独训练 flat baseline
torchrun --nproc_per_node=4 scripts/train_template_compiler.py --mode flat
```

## 完整实验流程（4 阶段）

```bash
# 一键全流程
bash run.sh

# 分步执行：
# Stage 1: Teacher 抽取程序 + 构建子程序库
python3 scripts/extract_templates.py --config configs/template_config.yaml

# Stage 2: 构建 MCD split
python3 scripts/build_mcd_split.py --config configs/template_config.yaml

# Stage 3: 训练（多卡 torchrun）
torchrun --nproc_per_node=4 scripts/train_template_compiler.py --mode compose
torchrun --nproc_per_node=4 scripts/train_template_compiler.py --mode flat

# Stage 4: 评估
python3 scripts/eval_template_reasoning.py --config configs/template_config.yaml
```

## 断点续训

- 训练支持 `--resume auto`（自动从最新 checkpoint 恢复）
- Pipeline 使用 `results/.phase_markers/` 跟踪完成状态
- 强制重跑：`FORCE_RERUN=1 bash run.sh`

## 项目结构

```
src/
  template_dsl.py        # 核心 DSL: Program, Step, Op, Subroutine, CompositionPlan
  mcd_split.py           # CFQ-style MCD split 构建
  template_algebra.py    # Legacy template algebra
scripts/
  extract_templates.py   # Stage 1: Teacher → programs → library
  build_mcd_split.py     # Stage 2: MCD split
  train_template_compiler.py  # Stage 3: 训练 (torchrun)
  eval_template_reasoning.py  # Stage 4: 评估
configs/
  template_config.yaml   # 全部配置
results/                 # 实验结果
```

## 下一步（达到 6/10 的路径）

1. 1k+ gold-verified GSM8K programs
2. Real MCD split with unseen compounds > 0
3. Compose > flat by >= 5 absolute points
4. 3 seeds with confidence intervals
5. 一键可复现
