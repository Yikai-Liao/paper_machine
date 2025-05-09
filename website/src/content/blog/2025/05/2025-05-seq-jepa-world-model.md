---
title: "seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models"
pubDatetime: 2025-05-06T04:39:11+00:00
slug: "2025-05-seq-jepa-world-model"
type: "arxiv"
id: "2505.03176"
score: 0.4295581983056516
author: "grok-3-latest"
authors: ["Hafez Ghaemi", "Eilif B. Muller", "Shahab Bakhtiari"]
tags: ["Self-Supervised Learning", "World Model", "Invariance", "Equivariance", "Sequence Processing"]
institution: ["Université de Montréal", "Mila - Quebec AI Institute", "Centre de Recherche Azrieli du CHU Sainte-Justine"]
description: "本文提出 seq-JEPA，一种基于联合嵌入预测架构的世界建模范式，通过序列化动作-观察对和架构设计同时学习不变性和等变性表征，成功缓解自监督学习中的性能权衡问题。"
---

> **Summary:** 本文提出 seq-JEPA，一种基于联合嵌入预测架构的世界建模范式，通过序列化动作-观察对和架构设计同时学习不变性和等变性表征，成功缓解自监督学习中的性能权衡问题。 

> **Keywords:** Self-Supervised Learning, World Model, Invariance, Equivariance, Sequence Processing

**Authors:** Hafez Ghaemi, Eilif B. Muller, Shahab Bakhtiari

**Institution(s):** Université de Montréal, Mila - Quebec AI Institute, Centre de Recherche Azrieli du CHU Sainte-Justine


## Problem Background

自监督学习（SSL）中的主流双视图范式通过数据增强等变换诱导不变性或等变性，但在不变性相关任务（如图像分类）和等变性相关任务（如旋转预测）之间存在性能权衡，限制了表征在下游任务中的适应性。
论文旨在通过引入序列化动作-观察对处理，模仿人类和动物通过多视角观察学习的方式，解决这一权衡问题，同时探索序列聚合在特定任务（如路径积分）中的潜力。

## Method

*   **核心思想**：基于联合嵌入预测架构（Joint-Embedding Predictive Architecture, JEPA），通过处理输入图像的一系列视图（观察）及其导致下一视图的相对变换（动作），利用架构归纳偏见同时学习对变换不变和等变的两种分离表征。
*   **具体实现**：
    *   **输入序列生成**：从输入图像生成一系列不同视图，每个视图与导致下一视图的相对变换（动作）嵌入结合。
    *   **编码与拼接**：使用骨干编码器（如 ResNet-18）对每个视图进行编码，编码结果与动作嵌入拼接。
    *   **序列聚合**：将拼接后的表征输入到一个 Transformer 编码器（类似工作记忆），通过一个可学习的 [AGG] 令牌生成聚合表征，倾向于不变性；而单个视图的编码表征倾向于等变性。
    *   **预测与优化**：基于聚合表征和下一个动作嵌入，使用 MLP 预测器预测下一视图的表征，通过余弦相似度损失优化预测结果；目标编码器采用指数移动平均（EMA）更新以避免表征坍塌。
*   **创新点**：不显式引入等变性损失或预测器，而是通过动作条件化的序列预测学习和架构设计隐式诱导不变性和等变性的分离，确保两种表征在架构上的解耦。

## Experiment

*   **有效性**：在 3DIEBench 数据集上，seq-JEPA 在旋转预测（等变性，R² 得分 0.71）上与最佳等变方法（如 ContextSSL，R² 0.74）相当，同时在分类任务（不变性，top-1 准确率 87.41%）上显著优于所有基线（最佳基线 EquiMod 为 84.29%），表明其成功缓解了不变性-等变性权衡。
*   **全面性**：实验覆盖多种变换类型（3D 旋转、手工艺增强、模拟眼动），在 3DIEBench、CIFAR100、Tiny ImageNet 和 STL-10 数据集上与多种不变性和等变性基线（如 SimCLR, BYOL, EquiMod, SIE）对比，展现出竞争力，尤其在需要序列聚合的任务（如路径积分）上表现突出。
*   **合理性**：实验设计合理，控制了架构差异（如基线也尝试了 Transformer 编码器），并通过消融实验验证了动作条件化和序列长度的作用，例如移除动作条件化后等变性性能显著下降（R² 从 0.71 降至 0.29）。
*   **局限性**：在手工艺增强任务（如 CIFAR100）上，分类性能略低于最佳不变性基线（如 BYOL 的 62.17% vs seq-JEPA 的 60.6%），可能因序列观察在人工增强数据上的优势不如自然变换明显。

## Further Thoughts

seq-JEPA 通过序列化动作-观察对和架构归纳偏见隐式诱导不变性和等变性分离，这一思想启发我们可以在其他模态（如视频、文本）或多模态数据上应用类似序列化处理，利用时间或空间序列增强表征学习；此外，其引入显著性采样和返回抑制（IoR）机制的生物启发方法，提示我们可以在 AI 模型中更多借鉴神经科学原理，设计更接近生物系统的学习机制。