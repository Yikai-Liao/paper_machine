---
title: "floq: Training Critics via Flow-Matching for Scaling Compute in Value-Based RL"
pubDatetime: 2025-09-08T16:31:09+00:00
slug: "2025-09-floq-flow-matching-rl"
type: "arxiv"
id: "2509.06863"
score: 0.691117616422529
author: "grok-3-latest"
authors: ["Bhavya Agrawalla", "Michal Nauman", "Khush Agarwal", "Aviral Kumar"]
tags: ["Reinforcement Learning", "Value Function", "Iterative Computation", "Flow Matching", "Test-Time Scaling"]
institution: ["Carnegie Mellon University", "University of Warsaw"]
description: "本文提出 `floq`，一种基于流匹配的 Q 函数参数化方法，通过迭代计算和中间监督显著提升了强化学习中 Q 值估计的性能和计算扩展性。"
---

> **Summary:** 本文提出 `floq`，一种基于流匹配的 Q 函数参数化方法，通过迭代计算和中间监督显著提升了强化学习中 Q 值估计的性能和计算扩展性。 

> **Keywords:** Reinforcement Learning, Value Function, Iterative Computation, Flow Matching, Test-Time Scaling

**Authors:** Bhavya Agrawalla, Michal Nauman, Khush Agarwal, Aviral Kumar

**Institution(s):** Carnegie Mellon University, University of Warsaw


## Problem Background

现代机器学习中，迭代计算（Iterative Computation）通过对中间步骤提供密集监督（如语言模型中的下一个 token 预测或扩散模型中的逐步去噪），显著提升了模型对复杂函数的建模能力。然而，在强化学习（RL）尤其是基于价值的 RL 中，Q 函数通常以单体架构表示，缺乏迭代计算和中间监督，导致难以充分利用深度网络容量，尤其在离线 RL 场景下，Q 值估计的准确性和泛化能力不足。本文旨在探索如何通过迭代计算改进 Q 值估计，提升 RL 性能。

## Method

*   **核心思想:** 将 Q 函数参数化为一个时间依赖的速度场（Velocity Field），通过数值积分（Numerical Integration）将初始噪声逐步转化为标量 Q 值，从而实现迭代计算，并通过调整积分步数（Integration Steps）动态控制计算容量。
*   **具体实现:** 从均匀分布 Unif[l, u] 中采样初始噪声，通过速度场进行多次迭代计算，逐步逼近目标 Q 值分布（理想为以真实 Q 值中心的 Dirac-Delta 分布）。训练时采用流匹配目标（Flow-Matching Objective），对每个中间步骤提供监督，确保速度场匹配基于时序差分（TD）目标的动态变化，同时使用目标速度场（Target Velocity Field）进行稳定训练。
*   **设计优化:** 针对训练中非平稳目标和标量 Q 值的特殊性，提出以下优化：
    *   **初始噪声分布范围:** 设置噪声范围 [l, u]，确保与目标 Q 值范围有足够重叠，避免流轨迹过直或过曲，影响性能。
    *   **分类输入表示:** 对中间插值输入（Interpolant）采用分类表示（Categorical Representation），通过 HL-Gauss 编码缓解输入非平稳性带来的训练不稳定。
    *   **傅里叶基时间嵌入:** 使用傅里叶基嵌入（Fourier-Basis Embedding）表示时间变量，使速度场在不同积分步骤中表现差异化，避免退化为单体架构。
*   **关键优势:** 与传统单体 Q 网络不同，`floq` 通过迭代计算和中间监督增强了表达能力，同时支持测试时计算扩展（Test-Time Scaling），即通过增加积分步数提升 Q 函数容量。

## Experiment

*   **有效性:** 在 OGBench 基准测试集的 50 个离线 RL 任务上，`floq` 平均性能比最先进的单体 Q 网络方法（如 FQL）提高了近 1.8 倍，尤其在困难任务（如 `antmaze-giant`, `hmmaze-large`）上表现突出；在在线微调任务中，`floq` 提供了更强的初始化并更快收敛到更高性能。
*   **优越性:** 对比单体 Q 网络、ResNet 架构和集成方法（Ensembles），`floq` 在计算匹配（Compute-Matched）条件下展现出更强的性能，表明迭代计算和中间监督的独特优势；增加积分步数通常提升性能，但过多的步数可能导致过拟合或数值积分误差。
*   **实验设置合理性:** 实验涵盖多种基线方法和消融研究（Ablation Studies），详细分析了初始噪声范围、输入表示和时间嵌入等设计选择的影响，设置较为全面；统计分析（如中位数、IQM 分数）进一步确认了性能提升的显著性。
*   **局限性:** 过多的积分步数可能导致 TD 误差不稳定，需仔细调参；OGBench 任务虽具挑战性，但是否完全代表真实世界 RL 问题仍需验证。

## Further Thoughts

迭代计算与中间监督的结合在 Q 值估计中展现了巨大潜力，是否可以推广到策略网络或其他 RL 组件，甚至非 RL 领域的回归任务？此外，测试时计算扩展（Test-Time Scaling）通过增加积分步数提升容量，是否可以设计自适应机制，根据任务难度动态调整步数？最后，流轨迹曲率（Curvature）与性能的关系提示迭代计算中可能存在‘最优复杂性’，如何量化并优化这种复杂性值得进一步探索。