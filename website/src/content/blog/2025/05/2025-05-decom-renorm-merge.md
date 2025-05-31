---
title: "Decom-Renorm-Merge: Model Merging on the Right Space Improves Multitasking"
pubDatetime: 2025-05-29T05:37:53+00:00
slug: "2025-05-decom-renorm-merge"
type: "arxiv"
id: "2505.23117"
score: 0.7211959034807057
author: "grok-3-latest"
authors: ["Yuatyong Chaichana", "Thanapat Trachu", "Peerat Limkonchotiwat", "Konpat Preechakul", "Tirasan Khandhawit", "Ekapol Chuangsuwanich"]
tags: ["Model Merging", "Multitask Learning", "Parameter Interference", "SVD Decomposition", "Weight Delta"]
institution: ["Chulalongkorn University", "AI Singapore", "UC Berkeley", "Mahidol University"]
description: "本文提出 Decom-Renorm-Merge 方法，通过奇异值分解和重归一化构建共享表示空间，显著提升了多任务模型合并的性能。"
---

> **Summary:** 本文提出 Decom-Renorm-Merge 方法，通过奇异值分解和重归一化构建共享表示空间，显著提升了多任务模型合并的性能。 

> **Keywords:** Model Merging, Multitask Learning, Parameter Interference, SVD Decomposition, Weight Delta

**Authors:** Yuatyong Chaichana, Thanapat Trachu, Peerat Limkonchotiwat, Konpat Preechakul, Tirasan Khandhawit, Ekapol Chuangsuwanich

**Institution(s):** Chulalongkorn University, AI Singapore, UC Berkeley, Mahidol University


## Problem Background

多任务学习旨在构建能在多个任务上表现优异的单一模型，但传统方法面临高昂的训练成本和数据获取难题；模型合并（Model Merging）作为替代方案，试图将独立微调的模型融合为多任务模型，却因权重矩阵中特征排列变化和神经元多义性导致参数干扰（Parameter Interference），影响合并效果。

## Method

* **核心思想**：提出 Decom-Renorm-Merge (DRM) 方法，通过奇异值分解（SVD）将不同模型的权重增量（Weight Delta，即微调模型与预训练模型的参数差异）投影到一个共享表示空间，在此空间内减少干扰并进行合并，而非直接在原始参数空间操作。
* **具体步骤**：
  * **分解（Decompose）**：将各任务的权重增量矩阵水平（DRM-H）或垂直（DRM-V）拼接后，使用 SVD 分解为共享基向量矩阵（U）、奇异值矩阵（Σ）和任务特定的基向量矩阵（V），从而将不同模型协调到同一表示空间。
  * **重归一化（Renormalize）**：对任务特定的基向量进行单位长度归一化，并将原始范数重新分配到奇异值中，以稳定后续剪枝和合并操作，避免因分解导致的向量不正交性引发的偏差。
  * **剪枝（Prune）**：在归一化后的基向量矩阵中，仅保留前 k% 高幅度条目，丢弃低幅度部分，减少参数干扰。
  * **合并（Merge）**：在共享空间内，采用符号选举（Sign Election）和不相交平均（Disjoint Averaging）等技术合并权重增量，最后重建回原始参数空间，得到多任务模型。
* **创新点**：强调在‘正确空间’中合并，通过数学分解协调特征表示，避免直接逐元素操作带来的干扰，同时无需额外训练即可实现高效融合。

## Experiment

* **有效性**：DRM 在多种模型和任务上显著优于基线方法，例如在 ViT-B/32 上，DRM-H 比最强基线提升 5.0%；在 DeBERTa-Base 上提升 9.3%；在 LoRA 微调的 Llama3.1-8B 上提升 1.9%，表明方法能有效减少参数干扰并提升多任务性能。
* **全面性与合理性**：实验覆盖了不同架构（ViT、DeBERTa、T5、Llama3.1-8B）和任务类型（图像分类、自然语言理解），包括有无验证集调参的场景，设置全面；此外，DRM 在任务数量增加时性能下降较慢，显示出良好的扩展性。
* **关键发现**：重归一化是方法成功的关键，移除此步骤会导致性能大幅下降（如 DeBERTa-Base 上下降 8.8%），验证了稳定共享空间的重要性。
* **局限性**：实验未探索对未见任务的泛化能力，也未涉及非 Transformer 架构（如 CNN、RNN），可能限制方法的适用范围。

## Further Thoughts

共享表示空间的概念非常具有启发性，通过数学分解（如 SVD）协调不同模型的特征表示，不仅适用于模型合并，还可能为模型压缩、知识迁移等领域提供新思路；此外，重归一化技术在稳定分解后基向量分布方面的作用，提示我们可以在其他依赖矩阵分解的算法中引入类似机制，以提升稳定性。