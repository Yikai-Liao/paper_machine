---
title: "On Entropy Control in LLM-RL Algorithms"
pubDatetime: 2025-09-03T17:23:19+00:00
slug: "2025-09-adaptive-entropy-llm"
type: "arxiv"
id: "2509.03493"
score: 0.5183223852214205
author: "grok-3-latest"
authors: ["Han Shen"]
tags: ["LLM", "Reinforcement Learning", "Entropy Control", "Policy Optimization", "Exploration"]
institution: ["Ant Group"]
description: "本文提出 AEnt 方法，通过钳制熵计算空间和自适应熵奖励系数，解决了 LLM-RL 中传统熵正则化的偏差问题，并在数学推理任务中显著优于基线。"
---

> **Summary:** 本文提出 AEnt 方法，通过钳制熵计算空间和自适应熵奖励系数，解决了 LLM-RL 中传统熵正则化的偏差问题，并在数学推理任务中显著优于基线。 

> **Keywords:** LLM, Reinforcement Learning, Entropy Control, Policy Optimization, Exploration

**Authors:** Han Shen

**Institution(s):** Ant Group


## Problem Background

在强化学习（RL）中，适当的熵控制对于算法有效性至关重要。传统熵正则化方法（如在 PPO、SAC 等算法中使用的熵奖励）在机器人和游戏任务中表现良好，但在大型语言模型（LLM）的 RL 训练中效果微弱甚至无效。原因是 LLM 的响应空间极其庞大且最优输出稀疏，导致传统熵正则化引入较大偏差，无法有效促进探索和优化。本文旨在分析这一问题并提出改进方法。

## Method

*   **核心思想:** 针对 LLM-RL 中传统熵正则化的偏差问题，提出一种新的熵控制方法 AEnt（Adaptive Entropy with token space clamping），通过限制熵计算的 token 空间和动态调整熵奖励系数，减少偏差并提升探索效率。
*   **具体实现:** 
    *   **Clamped Entropy（钳制熵）:** 传统熵计算基于整个词汇表，而 AEnt 将熵计算限制在一个较小的、输入相关的 token 子集上（通过选择概率最高的 token 集合并重新归一化策略），从而减少对低概率、非最优 token 的关注，降低熵正则化带来的偏差。
    *   **Adaptive Coefficient（自适应系数）:** 传统方法使用固定熵奖励系数，而 AEnt 根据当前策略的钳制熵值动态调整系数，通过设定熵值的上下限，确保熵维持在合理范围，避免过高（导致过多无意义探索）或过低（导致探索不足）。
    *   **算法流程:** 在每个训练步骤中，结合策略优化目标（如 GRPO）和钳制熵奖励进行优化，并根据熵值更新系数，确保训练稳定性。
*   **关键点:** AEnt 不改变底层策略优化算法，仅通过熵计算和系数调整改进探索行为，适用于多种 LLM-RL 场景。

## Experiment

*   **有效性:** 实验在数学推理任务中进行，使用 Qwen2.5-math-1.5b 和 DeepSeek-R1-distilled-Qwen-1.5b 模型，在 MATH 和 OpenR1-math 数据集上测试。结果显示 AEnt 在多个基准（如 MATH-Hard、AIME24）上显著优于基线方法 GRPO 和传统熵正则化（EntReg），例如在 MATH 数据集上测试分数从 GRPO 的 0.524 提升至 0.552。
*   **稳定性:** Figure 4 表明 AEnt 的熵值和响应长度在训练中保持稳定，而 GRPO 出现熵崩溃，EntReg 熵值波动，显示 AEnt 有效避免了策略过早收敛或无意义探索。
*   **合理性:** 实验设置覆盖不同模型和数据集，评估指标包括多个数学推理基准，测试分数通过多次尝试平均获得，设置全面合理。
*   **消融研究:** Figure 5 和 Figure 6 验证了自适应系数和钳制百分比对性能的影响，自适应系数显著提升了响应长度控制和熵稳定性，钳制百分比的选择对性能有一定影响但整体稳健。
*   **开销:** AEnt 主要增加了钳制熵计算和系数调整的计算量，但未显著增加训练复杂度。

## Further Thoughts

论文提出的‘钳制熵’概念非常有启发性，通过限制熵计算到高概率 token 子集，减少了无意义探索的偏差，这一思路可推广到其他高维 RL 任务中，如图像生成或多模态模型训练。此外，自适应系数的设计启发我们可以在 RL 中引入更多动态调整机制，根据任务特性或训练阶段调整超参数。另一个值得探索的方向是，是否可以通过基于语义相关性或任务特定知识的 token 选择策略，进一步优化钳制空间的选择，从而提升性能。