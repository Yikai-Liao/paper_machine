---
title: "Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning"
pubDatetime: 2025-05-15T14:34:36+00:00
slug: "2025-05-multi-task-optimization"
type: "arxiv"
id: "2505.10347"
score: 0.5568496531004892
author: "grok-3-latest"
authors: ["Gabriel S. Gama", "Valdir Grassi Jr."]
tags: ["Multi-Task Learning", "Optimization Strategy", "Gradient Conflict", "Task Balancing", "Loss Weighting"]
institution: ["University of São Paulo"]
description: "本文通过全面实证分析，验证了专门多任务优化器（SMTOs）在复杂多任务场景中的显著优势，揭示了统一损失表现接近的原因，并探索了固定权重的潜力，为优化策略选择提供了重要参考。"
---

> **Summary:** 本文通过全面实证分析，验证了专门多任务优化器（SMTOs）在复杂多任务场景中的显著优势，揭示了统一损失表现接近的原因，并探索了固定权重的潜力，为优化策略选择提供了重要参考。 

> **Keywords:** Multi-Task Learning, Optimization Strategy, Gradient Conflict, Task Balancing, Loss Weighting

**Authors:** Gabriel S. Gama, Valdir Grassi Jr.

**Institution(s):** University of São Paulo


## Problem Background

多任务学习（MTL）在处理复杂现实问题时需平衡多个任务的学习，但任务间的梯度冲突和梯度范数差异常导致优化困难；专门的多任务优化器（SMTOs）被设计用来解决这些问题，然而近期研究质疑其有效性，认为其性能提升可能源于基线模型超参数优化不足或缺乏正则化；本文旨在验证这些质疑，探索 SMTOs 是否在更复杂的多任务场景下优于统一损失（Uniform Loss），以及固定权重是否能媲美动态优化的 SMTOs。

## Method

* **核心思想：** 通过广泛的实证评估，比较统一损失（Unitary Scalarization, Unit. Scal.）、专门多任务优化器（SMTOs）以及固定权重策略在多任务学习中的性能表现，揭示不同优化策略在任务复杂性变化下的适用性。
* **具体实现：**
  * **SMTOs 分类与选择：** 将 SMTOs 分为基于梯度（Gradient-based，如 MGDA、PCGrad）和基于损失（Loss-based，如 FAMO、Auto-Lambda）两类，评估了 16 种方法，并通过 Multi-MNIST 数据集的初步筛选选出表现优异的 SMTOs（如 IMTL、Nash-MTL）进行深入测试。
  * **实验设计：** 逐步增加任务复杂性，从简单的两任务问题（Multi-MNIST）到复杂的三任务高信息量场景（Cityscapes 的 19 类分割+实例分割+视差估计）和多任务量子化学问题（QM9），观察优化策略表现；采用网格搜索优化超参数，确保公平对比。
  * **固定权重测试：** 从 SMTOs 中提取动态权重，应用指数移动平均后作为固定权重，用于验证其是否能替代动态优化，减少计算开销。
* **关键点：** 不修改模型架构，仅通过优化策略调整任务平衡，重点分析任务间干扰（通过梯度余弦相似度等指标）和复杂性对性能的影响。

## Experiment

* **有效性：** 在简单任务场景（如 Multi-MNIST 两任务问题）中，统一损失与 SMTOs 性能相当（∆mtm 差距较小），但在复杂场景（如 Cityscapes 三任务配置和 QM9 数据集）中，SMTOs（如 Nash-MTL、FAMO）显著优于统一损失，∆mtm 提升明显，例如在 QM9 上 Nash-MTL 的 ∆mtm 为 -53.62% 对比统一损失的 -135.7%。
* **固定权重表现：** 从 SMTOs 提取的固定权重在复杂场景中可达到接近动态 SMTOs 的性能（如 QM9 上固定权重 ∆mtm 为 -53.66%，接近 Nash-MTL 的 -53.62%），但稳定性较差，且寻找最优权重计算成本高。
* **实验设置合理性：** 实验覆盖从简单到复杂的多种任务配置，数据集包括 Multi-MNIST、Cityscapes 和 QM9，超参数通过网格搜索优化以确保公平性；通过 ∆mtm 和均值排名（MR）等指标评估多任务性能，设置较为全面，但未涉及 NLP 和强化学习领域，存在一定局限性。
* **开销：** SMTOs 尤其是基于梯度的方法（如 IMTL）在高任务数场景下计算成本较高，部分方法（如 RotoGrad）甚至因数值不稳定而无法收敛；固定权重虽减少动态计算，但前期权重搜索成本较大。

## Further Thoughts

论文揭示了任务复杂性与优化策略效果的关联，启发我们根据任务特性选择合适的优化方法，例如在低干扰场景中使用简单统一损失以节省计算资源；固定权重接近 SMTOs 性能的发现提示，未来可探索更高效的权重搜索技术（如元学习或群体优化）来替代动态 SMTOs；此外，过参数化模型缓解多任务优化限制的观点为研究模型容量与任务平衡的关系提供了新方向。