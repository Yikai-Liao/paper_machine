---
title: "BNPO: Beta Normalization Policy Optimization"
pubDatetime: 2025-06-03T13:28:57+00:00
slug: "2025-06-beta-normalization-optimization"
type: "arxiv"
id: "2506.02864"
score: 0.5597829847761947
author: "grok-3-latest"
authors: ["Changyi Xiao", "Mengdi Zhang", "Yixin Cao"]
tags: ["LLM", "Reinforcement Learning", "Policy Optimization", "Reward Normalization", "Reasoning"]
institution: ["Fudan University", "Meituan Group"]
description: "本文提出 BNPO 方法，通过 Beta 分布动态规范化奖励函数，显著降低策略梯度估计方差，提升大型语言模型在推理任务中的训练稳定性和性能。"
---

> **Summary:** 本文提出 BNPO 方法，通过 Beta 分布动态规范化奖励函数，显著降低策略梯度估计方差，提升大型语言模型在推理任务中的训练稳定性和性能。 

> **Keywords:** LLM, Reinforcement Learning, Policy Optimization, Reward Normalization, Reasoning

**Authors:** Changyi Xiao, Mengdi Zhang, Yixin Cao

**Institution(s):** Fudan University, Meituan Group


## Problem Background

大型语言模型（LLMs）在推理任务中通过强化学习提升性能时，面临现有策略优化方法（如 REINFORCE 和 GRPO）缺乏奖励规范化或仅采用静态规范化策略的问题。
这种静态方法无法适应训练过程中策略分布的动态变化，导致梯度估计方差较高，进而影响训练稳定性。

## Method

*   **核心思想:** 提出 Beta Normalization Policy Optimization (BNPO)，通过 Beta 分布动态规范化奖励函数，适应策略模型在训练中的变化，降低梯度估计方差。
*   **奖励建模:** 针对二值奖励（0 或 1），将其视为 Bernoulli 分布随机变量，其期望 p(q) 进一步建模为 Beta 分布 f_D(p(q); a, b)，反映奖励的概率分布特性。
*   **动态规范化:** 引入另一个 Beta 分布 f_N(p(q); α, β) 作为规范化项，构建优势函数 A(q, o) = [R(q, o) - p(q)] / f_N(p(q); α, β)，其中 p(q) 作为基线，f_N 起到规范化作用。
*   **参数调整:** 使用蒙特卡洛采样估计 p(q) 的均值和方差，计算 f_D 的参数 a 和 b，并根据理论推导设置 f_N 的参数 α = 1 + a/3 和 β = 1 + b/3，以最小化梯度方差。
*   **优势分解:** 为处理复杂奖励系统，将多个二值奖励函数分别规范化后求平均，避免不同奖励间的干扰，提升方法适用性。
*   **理论支持:** 通过 Theorem 1 证明 BNPO 在特定参数设置下能有效降低梯度估计方差，并展示其对 REINFORCE 和 GRPO 的泛化能力（通过特定 α, β 值还原为这两种方法）。

## Experiment

*   **有效性:** BNPO 在 Qwen2.5-Math-1.5B 和 Qwen2.5-Math-7B 模型上均取得最高平均性能（pass@1），如在 7B 模型上平均准确率达 47.8%，优于 REINFORCE (47.2%)、ReMax (47.6%)、GRPO (47.1%) 和 REINFORCE++ (46.8%)，尤其在 AMC23 数据集上提升显著（68.8% vs. 64.5% GRPO）。
*   **稳定性:** 通过梯度范数分析，BNPO 表现出更稳定的训练动态，梯度波动较小，而 GRPO 和 REINFORCE 等方法波动明显。
*   **实验设置合理性:** 实验覆盖不同规模模型（1.5B 和 7B），使用 MATH 数据集训练，在 MATH500、AMC23、AIME2024、AIME2025 等多数据集上评估，超参数在各方法间一致，确保公平性。
*   **局限性:** BNPO 在部分数据集（如 MATH500）上未显著优于 GRPO，可能与任务特性有关；优势分解在格式奖励上的性能提升有限，因奖励快速饱和。

## Further Thoughts

BNPO 的动态奖励规范化思路可启发其他强化学习任务中处理高方差问题的方法，尤其在奖励分布随训练变化的场景；此外，Beta 分布建模奖励期望的方式是否可扩展至其他分布（如 Gaussian）以适应连续奖励场景？优势分解机制是否可进一步用于多目标优化或个性化推荐，确保不同目标间的平衡？