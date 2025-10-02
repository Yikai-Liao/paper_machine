---
title: "AttentionViG: Cross-Attention-Based Dynamic Neighbor Aggregation in Vision GNNs"
pubDatetime: 2025-09-29T22:47:48+00:00
slug: "2025-09-attentionvig-cross-aggregation"
type: "arxiv"
id: "2509.25570"
score: 0.47379409603055905
author: "grok-3-latest"
authors: ["Hakan Emre Gedik", "Andrew Martin", "Mustafa Munir", "Oguzhan Baser", "Radu Marculescu", "Sandeep P. Chinchali", "Alan C. Bovik"]
tags: ["Vision GNN", "Cross-Attention", "Feature Aggregation", "Image Recognition", "Graph Construction"]
institution: ["The University of Texas at Austin"]
description: "本文提出基于交叉注意力的动态邻居聚合方法，并设计 AttentionViG 架构，在图像分类、检测和分割任务中实现 SOTA 性能，同时保持计算效率。"
---

> **Summary:** 本文提出基于交叉注意力的动态邻居聚合方法，并设计 AttentionViG 架构，在图像分类、检测和分割任务中实现 SOTA 性能，同时保持计算效率。 

> **Keywords:** Vision GNN, Cross-Attention, Feature Aggregation, Image Recognition, Graph Construction

**Authors:** Hakan Emre Gedik, Andrew Martin, Mustafa Munir, Oguzhan Baser, Radu Marculescu, Sandeep P. Chinchali, Alan C. Bovik

**Institution(s):** The University of Texas at Austin


## Problem Background

视觉图神经网络（Vision GNNs, ViGs）在图像识别任务中展现出潜力，但现有节点-邻居特征聚合方法缺乏动态加权机制，无法有效捕捉节点与邻居间的复杂语义关系，尤其在图构建策略不完善时性能下降。
论文旨在解决这一问题，设计一种通用聚合方法，提升 ViG 在多种视觉任务中的鲁棒性和性能，同时保持计算效率。

## Method

*   **核心思想:** 提出一种基于交叉注意力的特征聚合方法，通过动态计算节点与邻居间的注意力权重，捕捉语义相关性，而无需依赖特定图构建策略。
*   **具体步骤:** 
    *   将输入图像分割为非重叠 patches，作为图中的节点，初始特征通过卷积 stem 提取。
    *   对于每个节点，从其自身特征提取查询（Query）向量，从邻居特征提取键（Key）向量，通过余弦相似度计算相关性分数。
    *   使用指数核（Exponential Kernel）将相似度转化为注意力权重，不强制邻居间竞争，允许更灵活的加权。
    *   对邻居特征进行加权求和，与节点自身特征拼接后，通过可学习线性变换和非线性激活（如 GeLU）生成最终输出。
*   **架构设计:** 基于此方法，设计了 AttentionViG 架构，结合倒残差块（Inverted Residual Blocks）用于局部特征提取，以及 Grapher 层用于非局部消息传递，采用 SVGA 图构建策略以降低计算成本。
*   **创新点:** 交叉注意力机制弥补了固定图构建（如 SVGA）中语义无关连接的缺陷，指数核相较于 Softmax 提供了更具表达力的注意力分配。

## Experiment

*   **分类任务效果:** 在 ImageNet-1K 上，AttentionViG 取得了 SOTA 性能，例如 AttentionViG-B 达到 83.9% Top-1 准确率，优于同等参数规模的 ViG 变体（如 PViG, MobileViG）和部分 CNN/ViT 模型，同时 FLOPs 保持竞争力。
*   **下游任务表现:** 在 MS COCO 2017 目标检测和实例分割任务中，AttentionViG-B 达到 AP_box 46.4 和 AP_mask 42.3，超越多个 SOTA 模型；在 ADE20K 语义分割任务中，mIoU 达到 47.8，表现出良好的泛化能力。
*   **消融研究:** 交叉注意力方法在多种聚合函数中表现最佳（Top-1 准确率与 EdgeConv 相当，但 FLOPs 仅为其 66%）；指数核相较于 Softmax 提升了 0.5% 性能，验证了其灵活加权的优势。
*   **实验设置合理性:** 实验覆盖了分类、检测和分割任务，对比了多种模型规模和架构，数据详实且具有说服力；但未深入探讨极低资源环境下的表现及额外计算开销对硬件的适应性。

## Further Thoughts

交叉注意力动态加权邻居的机制启发了我，这种方法不仅限于 ViGs，可能推广至视频理解或点云处理等图结构任务；此外，指数核不强制竞争的设计是否能在其他注意力机制（如 ViTs）中应用，以提升表达能力？另外，AttentionViG 的 CNN-GNN 混合架构表明局部与全局特征提取的协同作用是未来视觉模型设计的重要方向，或许可以通过更灵活的模块组合或自适应图构建进一步优化性能。