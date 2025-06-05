---
title: "Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts"
pubDatetime: 2025-06-02T19:03:00+00:00
slug: "2025-06-selective-rollout-efficiency"
type: "arxiv"
id: "2506.02177"
score: 0.6876771522366457
author: "grok-3-latest"
authors: ["Haizhong Zheng", "Yang Zhou", "Brian R. Bartoldson", "Bhavya Kailkhura", "Fan Lai", "Jiawei Zhao", "Beidi Chen"]
tags: ["LLM", "Reinforcement Learning", "Rollout Efficiency", "Data Selection", "Reasoning"]
institution: ["Carnegie Mellon University", "Lawrence Livermore National Laboratory", "University of Illinois Urbana-Champaign", "Meta AI"]
description: "本文提出 GRESO 算法，通过奖励动态在 rollout 前过滤无信息提示，显著提升了大型语言模型推理任务中强化学习的训练效率，同时保持模型性能。"
---

> **Summary:** 本文提出 GRESO 算法，通过奖励动态在 rollout 前过滤无信息提示，显著提升了大型语言模型推理任务中强化学习的训练效率，同时保持模型性能。 

> **Keywords:** LLM, Reinforcement Learning, Rollout Efficiency, Data Selection, Reasoning

**Authors:** Haizhong Zheng, Yang Zhou, Brian R. Bartoldson, Bhavya Kailkhura, Fan Lai, Jiawei Zhao, Beidi Chen

**Institution(s):** Carnegie Mellon University, Lawrence Livermore National Laboratory, University of Illinois Urbana-Champaign, Meta AI


## Problem Background

大型语言模型（LLM）在推理任务中通过强化学习（RL）技术（如 PPO 和 GRPO）取得了显著进展，但 rollout 阶段（在线生成训练数据）的计算开销巨大，尤其是在扩展 rollout 以采样更多提示（prompts）时，许多无信息提示（uninformative prompts）即零方差提示（zero-variance prompts）对训练无贡献，造成了资源浪费。
论文旨在解决如何在 rollout 前识别并跳过这些无信息提示，以提高训练效率，同时不牺牲模型性能的关键问题。

## Method

*   **核心思想:** 提出 GRESO（GRPO with Efficient Selective Rollout），一种在线预 rollout 过滤算法，通过分析奖励训练动态（reward training dynamics）预测并跳过无信息提示，减少计算开销。
*   **具体实现:**
    *   **奖励动态追踪:** 记录每个提示在训练过程中的奖励历史（包括每次采样的奖励集合），判断其是否为零方差提示（即所有响应奖励相同，无学习信号）。
    *   **概率过滤策略:** 根据提示的历史奖励动态计算过滤概率（filtering probability），决定是否跳过该提示；通过概率方式平衡探索（exploration）和利用（exploitation），避免过早丢弃潜在有价值的提示。
    *   **自适应探索概率:** 针对简单（easy）和困难（hard）提示分别设置基础探索概率，并根据目标零方差比例动态调整，确保在不同训练阶段和模型能力下的灵活性。
    *   **自适应采样批次大小:** 根据当前训练批次需求动态调整 rollout 批次大小，避免不必要的计算浪费，例如当仅需少量有效提示时减少采样规模。
*   **关键优势:** 相比传统动态采样（Dynamic Sampling）在 rollout 后过滤的方式，GRESO 在 rollout 前预测无信息提示，显著降低计算成本，同时通过概率机制保留一定的探索空间。

## Experiment

*   **有效性:** GRESO 在多个数学推理基准数据集（如 MATH500, AMC, Gaokao 等）上与动态采样（Dynamic Sampling, DS）相比，准确率几乎无下降，甚至在某些数据集上略有提升（如 Qwen2.5-Math-1.5B 在 Gaokao 上从 64.2% 提升到 66.2%）。
*   **效率提升:** GRESO 显著减少了 rollout 次数（最高减少 3.35 倍），并实现了高达 2.4 倍的 rollout 阶段墙钟时间加速和 2.0 倍的总训练时间加速，例如在 Qwen2.5-Math-7B 上，rollout 时间从 155.9 小时减少到 65.5 小时。
*   **实验设置合理性:** 实验覆盖了不同规模模型（1.5B 到 7B）、不同数据集（DAPO+MATH 和 OPEN-R1 30k 子集）以及六个基准，设置全面且具有代表性；消融研究进一步验证了各组件（如自适应批次大小）的贡献，增强了结果的可信度。
*   **局限性:** GRESO 目前仅针对零方差提示过滤，未对剩余提示的价值进行更细粒度的评估，可能存在进一步优化的空间。

## Further Thoughts

论文中关于提示价值的时间一致性（temporal consistency）的观察非常具有启发性，不仅适用于 RL 训练中的 rollout 选择，还可以推广到监督微调或预训练阶段的数据筛选；此外，GRESO 的概率过滤策略平衡探索与利用的思想，可启发我们在其他动态优化问题中设计类似机制，如自适应调整学习率或正则化强度；针对不同难度提示分别调整探索率的策略也可能对处理异构数据或任务具有广泛适用性。