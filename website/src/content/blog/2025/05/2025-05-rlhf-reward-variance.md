---
title: "Accelerating RLHF Training with Reward Variance Increase"
pubDatetime: 2025-05-29T08:54:06+00:00
slug: "2025-05-rlhf-reward-variance"
type: "arxiv"
id: "2505.23247"
score: 0.5489813284946536
author: "grok-3-latest"
authors: ["Zonglin Yang", "Zhexuan Gu", "Houduo Qi", "Yancheng Yuan"]
tags: ["LLM", "RLHF", "Reward Variance", "Optimization", "Post-Training"]
institution: ["The Hong Kong Polytechnic University"]
description: "本文提出了一种通过奖励调整模型增加奖励方差的 GRPOVI 算法，显著加速了 RLHF 训练，同时以高效的 O(n log n) 算法解决了非凸优化问题。"
---

> **Summary:** 本文提出了一种通过奖励调整模型增加奖励方差的 GRPOVI 算法，显著加速了 RLHF 训练，同时以高效的 O(n log n) 算法解决了非凸优化问题。 

> **Keywords:** LLM, RLHF, Reward Variance, Optimization, Post-Training

**Authors:** Zonglin Yang, Zhexuan Gu, Houduo Qi, Yancheng Yuan

**Institution(s):** The Hong Kong Polytechnic University


## Problem Background

大型语言模型（LLM）在后训练阶段通过人类反馈的强化学习（RLHF）对齐人类价值观和偏好，但训练效率仍是一个挑战。
近期研究表明，初始策略模型的奖励方差（Reward Variance）越高，RLHF 训练速度越快。
基于此，作者旨在开发一种方法，通过增加奖励方差来加速 RLHF 训练，特别是在 Group Relative Policy Optimization (GRPO) 框架下，解决训练速度慢的关键问题。

## Method

*   **核心思想:** 提出一个奖励调整模型（Reward Adjustment Model），通过增加奖励方差来加速 RLHF 训练，同时保持奖励期望和响应间的相对偏好不变。
*   **具体实现步骤:**
    *   对于每个提示（Prompt），从初始策略模型生成一组响应（Responses），并计算原始奖励值和生成概率。
    *   构建一个非凸优化问题，目标是最大化调整后奖励的方差，约束条件包括：调整后的奖励值在原始范围（m, M）内，保持奖励期望不变，以及维持响应间的顺序关系（即偏好一致性）。
    *   针对非凸优化问题（一般为 NP 难），通过显式刻画可行集的极点（Extreme Points），设计了两种算法：枚举搜索算法（O(n²)）和单遍搜索算法（O(n log n)），后者更高效，能快速找到全局最优解。
    *   将调整后的奖励集成到 GRPO 框架中，形成 GRPOVI（GRPO with Reward Variance Increase）算法，用于 RLHF 训练。
*   **理论保障:** 通过 Theorem 1 证明，调整后的奖励确实能增加响应空间上的奖励方差。
*   **关键优势:** 方法仅需在初始阶段调整奖励，额外计算开销小，且不改变原始模型结构，易于集成到现有 RLHF 流程中。

## Experiment

*   **算法效率验证:** 针对奖励调整模型的求解，单遍搜索算法（O(n log n)）在响应数量较大时显著优于枚举搜索算法（O(n²)），且两者均能达到全局最优解，表明计算开销可控。
*   **RLHF 训练效果:** 在 UltraFeedback 数据集上，使用 GRM-Gemma-2-2B 和 GRM-Llama-3.2-3B 奖励模型，GRPOVI 算法在训练和测试集上的平均奖励（由 ArmoRM 模型评估）均显著高于原始 GRPO 算法，验证了增加奖励方差能有效加速训练。
*   **实验设置合理性:** 实验包含多次独立训练以减少随机性影响，数据集划分（训练、测试、SFT）和奖励标准化符合常规做法，设置全面且合理。
*   **额外开销:** GRPOVI 仅增加了一个 O(n log n) 的搜索算法，计算开销微乎其微，与 GRPO 的每迭代训练时间相当。

## Further Thoughts

论文中调整后奖励呈现的三值结构（Ternary Reward Structure）是一个有趣的启发，即奖励值最多只有三种（高、中、低），类似于规则-based 奖励机制，这不仅增加了奖励方差，还解释了类似 DeepSeek-R1 中规则奖励的经验有效性。这启发我思考：是否可以通过设计更复杂的奖励分布形状（如偏态或多峰分布）来进一步优化 RLHF 训练效果？此外，奖励方差的增加是否会影响训练稳定性，尤其是在噪声数据或分布偏移的情况下，这也是一个值得深入研究的方向。