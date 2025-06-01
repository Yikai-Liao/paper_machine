---
title: "Decom-Renorm-Merge: Model Merging on the Right Space Improves Multitasking"
pubDatetime: 2025-05-29T05:37:53+00:00
slug: "2025-05-decom-renorm-merge"
type: "arxiv"
id: "2505.23117"
score: 0.7211959034807057
author: "grok-3-latest"
authors: ["Yuatyong Chaichana", "Thanapat Trachu", "Peerat Limkonchotiwat", "Konpat Preechakul", "Tirasan Khandhawit", "Ekapol Chuangsuwanich"]
tags: ["LLM", "Model Merging", "Multitask Learning", "Representation Space", "Interference Reduction"]
institution: ["Chulalongkorn University", "AI Singapore", "UC Berkeley", "Mahidol University"]
description: "本文提出 Decom-Renorm-Merge (DRM) 方法，通过奇异值分解和重归一化构建共享表示空间，实现模型融合并显著提升多任务性能。"
---

> **Summary:** 本文提出 Decom-Renorm-Merge (DRM) 方法，通过奇异值分解和重归一化构建共享表示空间，实现模型融合并显著提升多任务性能。 

> **Keywords:** LLM, Model Merging, Multitask Learning, Representation Space, Interference Reduction

**Authors:** Yuatyong Chaichana, Thanapat Trachu, Peerat Limkonchotiwat, Konpat Preechakul, Tirasan Khandhawit, Ekapol Chuangsuwanich

**Institution(s):** Chulalongkorn University, AI Singapore, UC Berkeley, Mahidol University


## Problem Background

多任务学习旨在构建能够同时处理多个任务的模型，但传统方法需要大量数据和计算资源，成本高昂，尤其在专业领域数据获取受限的情况下；模型融合作为替代方案，通过合并独立微调的模型来创建多任务模型，但现有方法因假设权重矩阵相同位置参数具有一致语义功能而面临参数干扰问题，导致融合效果不佳。

## Method

* **核心思想**：通过奇异值分解（SVD）将不同模型的权重增量投影到一个共享表示空间，在该空间中减少参数干扰并进行融合，从而提升多任务性能。
* **具体步骤**：
  * **分解（Decompose）**：将多个任务的权重增量矩阵（即微调模型与基础模型的参数差值）水平（DRM-H）或垂直（DRM-V）拼接，使用 SVD 分解为共享基向量矩阵（U）、奇异值矩阵（Σ）和任务特定的基向量矩阵（V），从而将不同模型对齐到同一表示空间。
  * **重归一化（Renormalize）**：对任务特定的基向量进行单位长度归一化，同时调整奇异值以保留原始幅度，确保后续剪枝和融合操作的稳定性，避免因分割导致的不均匀性。
  * **剪枝（Prune）**：在重归一化后的基向量矩阵中，仅保留前 k% 高幅度条目，丢弃其余部分，以减少任务间的参数干扰。
  * **融合（Merge）**：在共享空间中应用符号选举（Sign Election）解决符号冲突，并使用不相交平均（Disjoint Averaging）合并剪枝后的基向量，最后通过共享基向量重建合并后的权重矩阵。
* **关键创新**：通过分解和重归一化解决权重矩阵直接比较时的语义不一致问题，并在共享空间中有效应用干扰减少技术，同时保留矩阵结构而非扁平化为向量。

## Experiment

* **有效性**：DRM 在多种模型架构和任务上显著优于基线方法，例如在 ViT-B/32 上比最强基线提升 5.0%，在 DeBERTa-Base 上提升 9.3%，在 LoRA 适应的 Llama3.1-8B 上提升 1.9%，表明方法在全微调和参数高效微调场景下均有效。
* **全面性**：实验覆盖了不同规模模型（从小型 ViT 到大型 Llama3.1-8B）、不同架构（编码器、编码器-解码器、解码器）和不同任务类型（图像分类、自然语言理解），并测试了不同任务数量的融合场景，验证了方法的普适性和可扩展性。
* **合理性**：消融实验证明重归一化是性能提升的关键，去除后性能下降明显（如 DeBERTa-Base 下降 8.8%）；统计显著性分析（配对 t 检验）显示 DRM 在任务数量增加时优势更显著；此外，DRM 对超参数变化表现出较强鲁棒性。
* **局限性**：方法对目标冲突的任务组合可能效果不佳，且未探索对未见任务的泛化能力。

## Further Thoughts

共享表示空间的概念可以通过分解技术扩展到模型蒸馏或联邦学习中，用于整合异构模型的知识；重归一化技术对稳定特征表示的启发可应用于神经网络解释性研究；此外，水平与垂直分解在不同架构上的性能差异提示未来可以探索自适应选择分解方向的策略，以优化融合效果。