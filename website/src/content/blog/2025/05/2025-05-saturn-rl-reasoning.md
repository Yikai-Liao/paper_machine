---
title: "SATURN: SAT-based Reinforcement Learning to Unleash Language Model Reasoning"
pubDatetime: 2025-05-22T08:23:10+00:00
slug: "2025-05-saturn-rl-reasoning"
type: "arxiv"
id: "2505.16368"
score: 0.6000027208690414
author: "grok-3-latest"
authors: ["Hao Zhu", "Ge Li", "Huanyu Liu", "Kechi Zhang", "Jia Li", "Yihong Dong"]
tags: ["LLM", "Reinforcement Learning", "Curriculum Learning", "Reasoning", "Logical Task"]
institution: ["Peking University"]
description: "SATURN 提出基于SAT问题的强化学习框架，通过可扩展、可验证和难度可控的任务设计，显著提升大型语言模型的推理能力，并在数学和编程任务中展现泛化性。"
---

> **Summary:** SATURN 提出基于SAT问题的强化学习框架，通过可扩展、可验证和难度可控的任务设计，显著提升大型语言模型的推理能力，并在数学和编程任务中展现泛化性。 

> **Keywords:** LLM, Reinforcement Learning, Curriculum Learning, Reasoning, Logical Task

**Authors:** Hao Zhu, Ge Li, Huanyu Liu, Kechi Zhang, Jia Li, Yihong Dong

**Institution(s):** Peking University


## Problem Background

当前强化学习（RL）任务在提升大型语言模型（LLM）推理能力方面面临三大局限：可扩展性（Scalability）、可验证性（Verifiability）和难度可控性（Controllable Difficulty）。
现有任务（如数学、编程或逻辑推理）依赖人工标注或昂贵的模型合成数据，难以大规模扩展；输出结果难以自动验证；且缺乏精细的难度控制，无法支持从易到难的渐进式学习。
SATURN 框架通过引入布尔可满足性问题（SAT）作为RL任务，旨在解决这些问题，利用SAT的可编程生成、线性时间验证和参数化难度控制特性，构建高效的训练和评估体系。

## Method

*   **核心思想：** 利用布尔可满足性问题（SAT）作为强化学习任务，通过多阶段课程学习（Curriculum Learning）框架，持续提升大型语言模型的推理能力，同时确保任务的可扩展性、可验证性和难度可控性。
*   **框架设计：** SATURN 包含两个主要循环：
    *   **课程估计循环（Curriculum Estimation Loop）：** 根据模型在验证集上的性能（pass@1指标），动态调整SAT任务难度，决定是否进入更高难度阶段。
    *   **模型训练循环（LLMs Training Loop）：** 在当前难度下生成训练数据，使用GRPO（Generalized Reward Policy Optimization）优化模型，奖励函数结合输出格式正确性和答案准确性。
*   **SAT实例构建：** 通过参数（n, k, l）定义SAT问题结构，其中n为每个子句的变量数，k为总变量数，l为子句数；算法 `SAT_Construction` 实现大规模、随机且满足已知解的实例生成，确保多样性和可扩展性。
*   **难度估计：** 提出难度函数 D(n, k, l) = log2(k) + 2*log2(l) - n + k/n，结合稀疏性（搜索成本与解空间比例）和结构复杂性（变量交互和子句宽度），为课程学习提供细粒度指导。
*   **训练细节：** 使用简单有效的奖励机制（格式错误为-1，格式正确但答案错误为0，正确答案为1），通过KL散度惩罚确保训练稳定性；课程学习从易到难，基于性能阈值动态调整难度。
*   **补充性：** SATURN 不替代数学或编程任务，而是作为逻辑推理的补充训练策略，可与其他任务结合以增强整体推理能力。

## Experiment

*   **有效性：** 在 SATURN-2.6k 数据集上，SATURN-1.5B 和 SATURN-7B 模型在未见过的更难SAT测试集上分别实现了 pass@3 提升 +14.0 和 +28.1，表明方法在目标任务上的显著改进。
*   **泛化性：** 在数学和编程任务（如 AIME, MATH-500, LiveCodeBench）上，SATURN-1.5B 和 SATURN-7B 平均得分分别提升 +4.9 和 +1.8，显示推理能力向其他领域的迁移效果。
*   **对比优势：** 与现有RL任务（如 Logic-RL）相比，SATURN 在数学和编程任务上进一步提升了 +8.8%，证明其作为补充任务的价值和优越性。
*   **实验设置合理性：** SATURN-2.6k 数据集包含1500个训练实例、160个同难度测试实例和1000个更难测试实例，支持系统性评估难度变化对性能的影响；消融实验验证了课程学习和难度控制的重要性；实验覆盖多个基准模型和任务类型，设置全面且结果可信。
*   **训练开销：** 主要开销在于多阶段训练和SAT实例生成，但通过可编程生成避免了人工标注成本，整体计算需求在合理范围内（基于8个A100 GPU）。

## Further Thoughts

SATURN 提出利用NP完全问题（如SAT）作为强化学习任务的通用基底，这一思路启发我们可以通过将复杂推理问题归约为标准逻辑问题，解决数据生成和验证难题，同时通过参数化设计实现难度控制，为课程学习提供新范式；此外，SAT任务培养的自验证（Self-Verification）行为在数学和编程任务中的迁移效果，提示逻辑任务设计可以针对性增强模型的反思和纠错能力，未来可探索更多此类任务（如其他NP问题）以进一步提升推理行为的多样性和鲁棒性。