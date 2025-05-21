---
title: "SRLoRA: Subspace Recomposition in Low-Rank Adaptation via Importance-Based Fusion and Reinitialization"
pubDatetime: 2025-05-18T14:12:40+00:00
slug: "2025-05-srlora-subspace-recomposition"
type: "arxiv"
id: "2505.12433"
score: 0.7225939981977073
author: "grok-3-latest"
authors: ["Haodong Yang", "Lei Wang", "Md Zakir Hossain"]
tags: ["LLM", "Low-Rank Adaptation", "Parameter Efficiency", "Subspace Recomposition", "Fine-Tuning"]
institution: ["Australian National University", "Griffith University", "Curtin University", "Data61/CSIRO"]
description: "本文提出 SRLoRA，通过基于重要性的融合和 SVD 方向重新初始化动态重构 LoRA 子空间，显著提升了参数高效微调的表达能力和收敛速度，同时保持参数效率。"
---

> **Summary:** 本文提出 SRLoRA，通过基于重要性的融合和 SVD 方向重新初始化动态重构 LoRA 子空间，显著提升了参数高效微调的表达能力和收敛速度，同时保持参数效率。 

> **Keywords:** LLM, Low-Rank Adaptation, Parameter Efficiency, Subspace Recomposition, Fine-Tuning

**Authors:** Haodong Yang, Lei Wang, Md Zakir Hossain

**Institution(s):** Australian National University, Griffith University, Curtin University, Data61/CSIRO


## Problem Background

大型预训练模型的全参数微调成本高昂，限制了其在多任务适应和资源受限环境中的应用。
低秩适应（LoRA）作为一种参数高效微调（PEFT）方法，通过引入两个低秩矩阵更新权重，降低了计算负担，但其更新空间被限制在固定的低秩子空间内，导致表达能力不足，影响下游任务性能。
本文旨在解决 LoRA 子空间固定的局限性，提升其适应能力，同时保持参数效率。

## Method

*   **核心思想:** 提出 SRLoRA（Subspace Recomposition in Low-Rank Adaptation），通过基于重要性的融合和重新初始化机制，动态调整 LoRA 的更新子空间，在不增加可训练参数数量的前提下增强表达能力。
*   **具体步骤:**
    *   **SVD 初始化:** 对预训练权重矩阵进行奇异值分解（SVD），初始时使用前 r 个主要方向初始化 LoRA 矩阵 A 和 B，确保起始子空间与预训练模型的重要方向对齐。
    *   **重要性评估:** 基于梯度-权重乘积计算每个 LoRA 组件（B 的列和 A 的对应行）的敏感性分数，并通过指数移动平均（EMA）平滑技术和不确定性估计，得到稳定且可靠的重要性分数，用于识别低重要性组件。
    *   **融合低重要性组件:** 定期将低重要性组件的贡献融合到冻结的预训练权重中，释放对应的可训练参数容量。
    *   **重新初始化子空间:** 利用预训练权重的 SVD 中未使用的下一个主要方向，重新初始化被融合的组件，同时从冻结权重中减去新初始化的投影以避免子空间重复，确保模型表示的多样性。
    *   **动态循环:** 在训练过程中持续执行上述融合与重新初始化操作，保持参数数量恒定，但不断探索新的子空间，提升适应能力。
*   **关键优势:** SRLoRA 突破了 LoRA 静态子空间的限制，通过动态重构机制实现了更灵活的适应，同时保持了计算效率。

## Experiment

*   **有效性:** 在 GLUE 基准测试中，SRLoRA 在多个任务上（如 SST-2 准确率 96.1%，RTE 准确率 82.1%）优于标准 LoRA（SST-2 95.9%，RTE 80.4%）和 PiSSA（SST-2 95.7%，RTE 82.0%），且训练损失曲线显示其初期收敛速度更快。
*   **视觉任务表现:** 在 Vision Transformer 上，SRLoRA 在复杂任务如 CIFAR-100 上准确率提升显著（92.51% vs LoRA 的 90.06%），但在简单任务如 MNIST 上表现略逊（94.83% vs LoRA 的 98.89%），表明其更适用于需要丰富表达能力的任务。
*   **实验设置合理性:** 实验覆盖了语言（GLUE）和视觉（CIFAR-100, STL-10, MNIST）任务，与基线方法在超参数（如学习率、批量大小）上保持一致，确保了公平比较；同时，数据集选择涵盖了不同复杂度和领域，验证了方法的通用性。
*   **局限性与分析:** 虽然 SRLoRA 在复杂任务上提升明显，但在简单任务上表现波动，提示其适用性可能与任务需求相关；此外，融合和重新初始化的额外计算开销虽小，但仍需优化。

## Further Thoughts

SRLoRA 的动态子空间重构机制启发我们思考如何将类似思想应用于其他 PEFT 方法或在线学习场景，通过任务驱动的子空间调整提升模型适应性；
其重要性评估方法结合 EMA 平滑技术，为模型剪枝或知识蒸馏提供了新的参数选择思路；
此外，利用 SVD 未使用方向刷新子空间的策略，提示我们可以进一步探索其他矩阵分解技术或结合任务特定信息优化子空间选择策略。