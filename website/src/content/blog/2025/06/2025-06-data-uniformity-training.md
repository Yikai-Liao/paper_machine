---
title: "Data Uniformity Improves Training Efficiency and More, with a Convergence Framework Beyond the NTK Regime"
pubDatetime: 2025-06-30T17:58:30+00:00
slug: "2025-06-data-uniformity-training"
type: "arxiv"
id: "2506.24120"
score: 0.5897140681980937
author: "grok-3-latest"
authors: ["Yuqing Wang", "Shangding Gu"]
tags: ["LLM", "Data Uniformity", "Training Efficiency", "Convergence Framework", "Neural Network"]
institution: ["Johns Hopkins University", "UC Berkeley"]
description: "本文提出数据均匀性作为通用数据选择原则，通过理论和实验证明其能显著提升大型语言模型训练效率并保持性能，同时提供超越 NTK 的收敛框架支持复杂架构优化。"
---

> **Summary:** 本文提出数据均匀性作为通用数据选择原则，通过理论和实验证明其能显著提升大型语言模型训练效率并保持性能，同时提供超越 NTK 的收敛框架支持复杂架构优化。 

> **Keywords:** LLM, Data Uniformity, Training Efficiency, Convergence Framework, Neural Network

**Authors:** Yuqing Wang, Shangding Gu

**Institution(s):** Johns Hopkins University, UC Berkeley


## Problem Background

数据选择在数据驱动决策（包括大型语言模型，LLMs）中至关重要，但传统方法通常依赖任务特定的规则，缺乏通用的量化原则。
论文探讨是否存在一种通用的数据选择策略，能够在复杂任务中（尤其缺乏先验知识时）一致提升训练效率和模型性能，聚焦于数据均匀性（Data Uniformity）的影响。

## Method

*   **核心思想:** 通过选择更均匀分布的数据（即数据点间最小距离 _h_min 较大），加速神经网络训练并减少近似误差。
*   **理论分析:** 
    *   证明更均匀的分布对应更大的 _h_min，并通过理论框架展示 _h_min 增大能加速梯度下降（Gradient Descent, GD）的收敛速度（通过分析训练动态），同时减少神经网络与真实函数间的近似误差（通过数据依赖的 Bramble-Hilbert 引理）。
    *   提出一个超越神经切线核（Neural Tangent Kernel, NTK）范式的收敛分析框架，适用于包括Transformer在内的多种复杂架构，不依赖Lipschitz平滑性假设，基于多项式广义平滑性（Polynomial Generalized Smoothness）和局部松弛耗散性（Local Relaxed Dissipativity）。
    *   提供残差连接（Residual Connections）和函数组合（Function Composition）在保持模型表达能力（Non-Degeneracy）方面的理论解释，基于测度论和微分拓扑。
*   **数据选择策略:** 在实验中，采用贪婪算法构建均匀数据子集，通过最大化已选数据点之间的最小余弦距离，确保数据空间的广泛覆盖。
*   **适用性:** 方法适用于监督微调场景，旨在减少训练数据量同时保持性能，特别针对大型语言模型的效率优化。

## Experiment

*   **有效性:** 均匀数据子集（Uniform Subset）显著提升了训练效率，例如在 LLaMA-1-13B 上，10k 均匀子集达到 0.31 损失阈值仅需 148.2 分钟，而随机子集和完整数据集分别需 281.2 和 561.6 分钟。
*   **性能表现:** 均匀子集在下游任务（如 ARC Challenge 和 TruthfulQA MC）上表现与完整数据集相当或略优，例如在 WizardLM 数据集上，10k 均匀子集的 ARC Challenge 准确率为 43.09%，接近完整 20k 数据集的 43.63%。
*   **实验设置合理性:** 实验覆盖多种优化策略（_ℓ_2-SGD 和 Cross-Entropy with Adam）、模型规模（LLaMA-1 7B 和 13B）和数据集（TeaMs-RL 和 WizardLM），通过可视化（PCA 投影）验证均匀子集的覆盖优势，控制变量（如数据集大小）以隔离 _h_min 影响，设置全面且合理。
*   **结论:** 方法在训练效率上提升明显，性能未受显著损失，证明均匀数据选择是一种高效策略。

## Further Thoughts

数据均匀性作为通用数据选择原则的潜力令人启发，未来可探索动态调整均匀性（如训练中逐步优化 _h_min）以进一步提升效率；此外，超越 NTK 的收敛框架为复杂架构优化提供了新思路，或许可应用于其他领域如强化学习；残差连接的非退化性解释也启发了对更深网络设计的理论支持，可能推广到新型架构设计中。