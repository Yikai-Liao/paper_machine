---
title: "Iterative Distillation for Reward-Guided Fine-Tuning of Diffusion Models in Biomolecular Design"
pubDatetime: 2025-07-01T05:55:28+00:00
slug: "2025-07-iterative-distillation-diffusion"
type: "arxiv"
id: "2507.00445"
score: 0.44144973152249384
author: "grok-3-latest"
authors: ["Xingyu Su", "Xiner Li", "Masatoshi Uehara", "Sunwoo Kim", "Yulai Zhao", "Gabriele Scalia", "Ehsan Hajiramezanali", "Tommaso Biancalani", "Degui Zhi", "Shuiwang Ji"]
tags: ["Diffusion Model", "Reward Guidance", "Fine-Tuning", "Off-Policy Learning", "Biomolecular Design"]
institution: ["Texas A&M University", "EvolutionaryScale", "Seoul National University", "Princeton University", "Genentech", "University of Texas Health Science Center at Houston"]
description: "本文提出 VIDD 框架，通过迭代蒸馏和离线策略微调扩散模型，稳定高效地优化非可微奖励函数下的生物分子设计任务，显著提升生成性能。"
---

> **Summary:** 本文提出 VIDD 框架，通过迭代蒸馏和离线策略微调扩散模型，稳定高效地优化非可微奖励函数下的生物分子设计任务，显著提升生成性能。 

> **Keywords:** Diffusion Model, Reward Guidance, Fine-Tuning, Off-Policy Learning, Biomolecular Design

**Authors:** Xingyu Su, Xiner Li, Masatoshi Uehara, Sunwoo Kim, Yulai Zhao, Gabriele Scalia, Ehsan Hajiramezanali, Tommaso Biancalani, Degui Zhi, Shuiwang Ji

**Institution(s):** Texas A&M University, EvolutionaryScale, Seoul National University, Princeton University, Genentech, University of Texas Health Science Center at Houston


## Problem Background

扩散模型在建模复杂高维数据分布（如图像、生物分子结构）方面表现出色，但在生物分子设计等实际应用中，需针对特定下游目标（如蛋白质结合亲和力、结构约束）进行优化，这些目标常通过非可微奖励函数定义；现有强化学习方法（如PPO）在微调扩散模型时因在线策略性质和反向KL散度目标，导致不稳定性、样本效率低和模式崩塌问题，亟需一种稳定高效的微调方法。

## Method

*   **核心思想:** 提出 VIDD（Value-guided Iterative Distillation for Diffusion Models）框架，通过迭代蒸馏软最优策略（Soft-Optimal Policies）来微调扩散模型，使其在非可微奖励函数下实现目标导向生成，同时保持训练稳定性和样本效率。
*   **具体实现:** 方法分为三个阶段：
    *   **Roll-In 阶段:** 采用离线策略（Off-Policy）生成训练数据，使用预训练模型和当前学生模型的混合策略采样轨迹，动态调整探索与利用比例，确保覆盖设计空间并逐步聚焦高奖励区域。
    *   **Roll-Out 阶段:** 基于当前模型（Roll-Out Policy）模拟软最优策略，采样去噪轨迹，并通过去噪预测直接近似软值函数（Soft Value Function），避免复杂 Monte Carlo 采样，提高计算效率。
    *   **Distillation 阶段:** 通过最小化软最优策略（教师策略）与当前模型（学生策略）之间的前向KL散度（Forward KL Divergence），采用值加权最大似然估计（Value-Weighted MLE）更新模型参数；引入懒惰更新机制（Lazy Update），定期刷新 Roll-Out 策略以增强稳定性。
*   **关键优势:** 离线策略允许更广泛的探索，前向KL散度目标避免模式崩塌，值函数近似提升计算效率，整体框架适用于非可微奖励场景。

## Experiment

*   **有效性:** VIDD 在蛋白质序列设计、DNA 调控序列设计和小分子设计任务上显著优于基线方法（如PPO、DDPP、标准微调），例如在蛋白质二级结构匹配任务中，β-sheet 比例从 0.05 提升至 0.69，小分子对接分数从 7.2 提升至 9.4。
*   **稳定性与样本效率:** VIDD 表现出更高的样本效率，收敛速度快于基线（如图3所示），在相同奖励查询次数下达到更高性能。
*   **自然性与多样性:** 在优化奖励的同时，VIDD 保持了生成样本的自然性（如蛋白质 pLDDT 分数、小分子 NLL）和多样性（Diversity 指标），避免了模式崩塌。
*   **实验设置合理性:** 实验覆盖多种生物分子设计任务，奖励函数设计符合实际需求（如非可微物理模拟评分），基线选择全面（包括推理时方法 Best-of-N 和 RL 方法 DDPO），评价指标兼顾奖励、自然性和多样性，设置较为全面；但奖励函数质量对结果有较大影响，若信号不准确或噪声大，可能导致优化问题。

## Further Thoughts

VIDD 的混合 Roll-In 策略在探索与利用平衡上的创新可推广至其他生成模型或强化学习任务；前向KL散度目标避免模式崩塌的特性启发我们在其他优化问题中探索类似目标；值函数近似的高效计算方法可应用于需要值估计的场景；此外，奖励函数质量对结果的影响提示未来可结合多目标优化或奖励塑造（Reward Shaping）设计更鲁棒的奖励机制。