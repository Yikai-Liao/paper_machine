---
title: "SATURN: SAT-based Reinforcement Learning to Unleash Language Model Reasoning"
pubDatetime: 2025-05-22T08:23:10+00:00
slug: "2025-05-saturn-reasoning-rl"
type: "arxiv"
id: "2505.16368"
score: 0.6000027208690414
author: "grok-3-latest"
authors: ["Hao Zhu", "Ge Li", "Huanyu Liu", "Kechi Zhang", "Jia Li", "Yihong Dong"]
tags: ["LLM", "Reinforcement Learning", "Curriculum Learning", "Reasoning", "Scalability"]
institution: ["Peking University"]
description: "本文提出 **SATURN**，一个基于SAT问题的强化学习框架，通过可扩展、可验证和难度可控的任务设计，显著提升大型语言模型的推理能力并泛化至数学和编程任务。"
---

> **Summary:** 本文提出 **SATURN**，一个基于SAT问题的强化学习框架，通过可扩展、可验证和难度可控的任务设计，显著提升大型语言模型的推理能力并泛化至数学和编程任务。 

> **Keywords:** LLM, Reinforcement Learning, Curriculum Learning, Reasoning, Scalability

**Authors:** Hao Zhu, Ge Li, Huanyu Liu, Kechi Zhang, Jia Li, Yihong Dong

**Institution(s):** Peking University


## Problem Background

当前强化学习（RL）任务在提升大型语言模型（LLM）推理能力时面临三大局限：可扩展性（Scalability）、可验证性（Verifiability）和难度可控性（Controllable Difficulty）。
现有任务（如数学、编程或逻辑推理）依赖人工标注或昂贵LLM合成数据，难以大规模扩展；模型输出难以自动验证；且缺乏细粒度难度控制，无法通过从易到难的课程学习逐步提升推理能力。
论文提出基于布尔可满足性问题（SAT）的RL框架 **SATURN**，以解决这些关键问题。

## Method

*   **核心思想:** 利用SAT问题作为强化学习任务，通过课程学习（Curriculum Learning）从易到难训练LLM，逐步提升其推理能力。
*   **SAT实例构建:** 设计 `SAT_Construction` 算法，生成满足特定难度参数（n, k, l）的SAT实例，其中 n 是每个子句的变量数，k 是总变量数，l 是子句总数，确保任务的可扩展性。
*   **难度估计:** 提出公式 D(n, k, l) = log2(k) + 2*log2(l) - n + k/n，结合稀疏性和结构复杂性，精确估计任务难度，用于课程学习中的难度分级。
*   **课程学习框架:** 包含两个交替循环：
    *   **课程估计循环（Curriculum Estimation Loop）:** 动态生成验证集，评估模型性能（pass@1），若超过阈值则提升难度，否则进入训练循环。
    *   **模型训练循环（LLMs Training Loop）:** 在当前难度下生成训练数据，使用强化学习（GRPO，Generalized Reward Policy Optimization）优化模型，奖励函数结合格式正确性和答案正确性。
*   **训练稳定性:** 通过基于性能的难度过渡机制，确保训练过程稳定，避免难度跳跃过大导致模型无法适应。
*   **关键优势:** SAT任务可程序化生成（无限数据）、线性时间验证（高可靠性）、参数化难度控制（适合课程学习），弥补了现有RL任务的不足。

## Experiment

*   **有效性:** 在 **SATURN-2.6k** 数据集上，**SATURN-1.5B** 和 **SATURN-7B** 在未见过的更难SAT测试集上分别实现 pass@3 提升 +14.0 和 +28.1，表明方法在目标任务上的显著改进。
*   **泛化性:** 在数学和编程任务（如 AIME, MATH-500, LiveCodeBench）上，两个模型平均得分分别提升 +4.9 和 +1.8，证明从SAT任务中学到的推理能力可迁移到其他领域。
*   **对比优势:** 相比现有RL任务构建方法（如 Logic-RL），**SATURN** 在数学和编程任务上进一步提升 +8.8%，显示其作为补充任务的价值。
*   **实验设置合理性:** 实验涵盖不同难度SAT任务、多个基准数据集，并与多种基线模型对比，设置全面；消融研究验证了课程学习和难度控制的重要性。
*   **局限性与分析:** 随着训练阶段增加，数学和编程任务提升趋于饱和，可能受限于知识局限和上下文窗口瓶颈；但整体数据显著性较高，难度估计与模型表现高度相关（R² 值 0.5-0.7）。

## Further Thoughts

SAT问题作为NP完全问题，可以作为逻辑推理的通用基础，是否可将其他复杂任务（如规划、决策）转化为SAT形式训练LLM？
**SATURN** 的自适应课程学习机制是否可推广到视觉推理或多模态任务，实现个性化训练？
SAT任务增强了模型自验证能力，是否可通过设计特定任务针对性提升模型反思或纠错能力？
未来是否可结合自然语言逻辑任务与SAT任务，形成混合训练框架，同时提升逻辑推理和语言理解？