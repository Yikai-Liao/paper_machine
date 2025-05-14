---
title: "CAT Merging: A Training-Free Approach for Resolving Conflicts in Model Merging"
pubDatetime: 2025-05-11T13:24:09+00:00
slug: "2025-05-conflict-aware-merging"
type: "arxiv"
id: "2505.06977"
score: 0.7174828946593633
author: "grok-3-latest"
authors: ["Wenju Sun", "Qingyong Li", "Yangli-ao Geng", "Boyang Li"]
tags: ["Multi-Task Learning", "Model Merging", "Knowledge Conflict", "Parameter Trimming"]
institution: ["Beijing Jiaotong University", "Nanyang Technological University"]
description: "本文提出 Conflict-Aware Task Merging (CAT Merging)，一种无需训练的框架，通过参数特定的修剪策略有效缓解多任务模型合并中的知识冲突，显著提升了合并模型的整体性能。"
---

> **Summary:** 本文提出 Conflict-Aware Task Merging (CAT Merging)，一种无需训练的框架，通过参数特定的修剪策略有效缓解多任务模型合并中的知识冲突，显著提升了合并模型的整体性能。 

> **Keywords:** Multi-Task Learning, Model Merging, Knowledge Conflict, Parameter Trimming

**Authors:** Wenju Sun, Qingyong Li, Yangli-ao Geng, Boyang Li

**Institution(s):** Beijing Jiaotong University, Nanyang Technological University


## Problem Background

多任务模型合并是一种无需额外训练即可将多个专家模型整合为统一模型的范式，但现有方法（如 Task Arithmetic）在合并任务向量时常因知识冲突（Knowledge Conflict）导致性能下降，表现为任务间的不平衡或矛盾，某些任务的特定知识可能被覆盖或压制。
论文旨在解决这一关键问题，通过缓解冲突来提升合并模型在多个任务上的整体性能。

## Method

*   **核心思想:** 提出 Conflict-Aware Task Merging (CAT Merging)，一种无需训练的框架，通过修剪任务向量中容易引发冲突的成分来缓解知识冲突，同时尽量保留任务特定知识。
*   **具体实现:**
    *   **特征层面分析:** 逐层分析任务向量对特征的影响，识别冲突来源。
    *   **线性权重修剪:** 对于线性层的权重参数，构建一个‘移除基’（Removal Basis），通过投影操作剔除与其他任务冲突的成分，优化目标是最大化冲突减少与知识保留的平衡。
    *   **归一化参数修剪:** 对于归一化层的缩放（Scaler）和偏移（Shift）参数，采用二进制掩码（Masking）操作，选择性地移除冲突成分，基于特征和任务向量的逐元素影响进行优化。
    *   **优化平衡:** 通过一个超参数 λ 权衡冲突减少与知识保留，使用少量无标签样本（通常每任务 2-3 个）进行前向推理以收集特征数据，无需额外训练。
*   **关键创新:** 不同于传统方法仅修剪低重要性成分，CAT Merging 特别针对高重要性但可能引发冲突的成分进行处理，确保任务间更平衡的整合。

## Experiment

*   **有效性:** CAT Merging 在多个任务类型上显著优于现有方法，例如在 ViT-B/32 视觉任务上平均准确率达 78.3%，比最先进的 PCB Merging 高 2.5%；在 ViT-L/14 上达 89.6%，高 2.0%；在 NLP 任务（RoBERTa 模型）上，8 个任务中 6 个表现最佳，平均性能提升至 62.56%。
*   **全面性与合理性:** 实验覆盖视觉、NLP 和视觉-语言任务，使用多种数据集（如 SUN397、MNIST、GLUE）和模型架构（ViT、RoBERTa、BLIP），与多种基线（如 Task Arithmetic、Ties-Merging）对比，设置全面；消融研究验证了线性权重、缩放和偏移修剪策略的必要性。
*   **鲁棒性:** 敏感性分析表明方法对样本数量（即使每任务仅 1 个样本）和超参数（如 λ 和 c）变化不敏感，性能稳定。
*   **局限性:** 尽管平均性能提升明显，但在某些特定数据集（如 Cars）上并非最佳，可能是由于追求任务平衡而非单一任务最优。

## Further Thoughts

CAT Merging 的冲突感知修剪策略启发我们可以在其他模型优化场景（如知识蒸馏或模型剪枝）中优先处理冲突成分，而非仅关注冗余；此外，针对不同参数类型设计特定修剪方法的思路，可以进一步结合模型架构的模块化特性，探索更细粒度的优化策略；最后，其无训练特性提示在资源受限场景下，可以开发更多依赖少样本或无标签数据的方法来提升模型性能。