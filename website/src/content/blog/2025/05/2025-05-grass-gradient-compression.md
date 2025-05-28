---
title: "GraSS: Scalable Influence Function with Sparse Gradient Compression"
pubDatetime: 2025-05-25T04:58:57+00:00
slug: "2025-05-grass-gradient-compression"
type: "arxiv"
id: "2505.18976"
score: 0.41037037943289156
author: "grok-3-latest"
authors: ["Pingbang Hu", "Joseph Melkonian", "Weijing Tang", "Han Zhao", "Jiaqi W. Ma"]
tags: ["LLM", "Gradient Compression", "Data Attribution", "Sparsity", "Influence Function"]
institution: ["University of Illinois Urbana-Champaign", "Womp Labs", "Carnegie Mellon University"]
description: "本文提出 G RA SS 和 F ACT G RA SS，通过利用梯度稀疏性和结构特性实现高效梯度压缩，显著降低数据归因的计算和内存开销，并在亿级规模模型上展现出优越的效率和归因精度。"
---

> **Summary:** 本文提出 G RA SS 和 F ACT G RA SS，通过利用梯度稀疏性和结构特性实现高效梯度压缩，显著降低数据归因的计算和内存开销，并在亿级规模模型上展现出优越的效率和归因精度。 

> **Keywords:** LLM, Gradient Compression, Data Attribution, Sparsity, Influence Function

**Authors:** Pingbang Hu, Joseph Melkonian, Weijing Tang, Han Zhao, Jiaqi W. Ma

**Institution(s):** University of Illinois Urbana-Champaign, Womp Labs, Carnegie Mellon University


## Problem Background

数据归因（Data Attribution）方法，如影响函数（Influence Functions），旨在衡量单个训练样本对机器学习模型预测的影响，但在大型语言模型（LLMs）等大规模场景中，计算和存储每个样本的梯度（Per-Sample Gradient）导致了巨大的内存和计算开销（O(np) 复杂度，其中 n 为样本数，p 为参数数），限制了其可扩展性。
本文致力于解决这一关键问题：如何在保持归因精度的同时，显著降低梯度计算和存储的资源需求，使影响函数在大规模模型中实用化。

## Method

*   **G RA SS（Gradient Sparsification and Sparse Projection）核心思想：** 针对每个样本梯度的固有稀疏性（Sparsity），提出两阶段压缩策略以实现亚线性（Sub-Linear）的时间和空间复杂度。
    *   **第一阶段 - 稀疏化（Sparsification）：** 将高维梯度向量（维度为 p）通过随机掩码（Random Mask, RM）或选择性掩码（Selective Mask, SM）方法压缩到中间维度 k'（k < k' ≪ p）。其中，RM 随机选择 k' 个维度，SM 通过优化目标（基于梯度相关性和稀疏正则化）选择更重要的维度。
    *   **第二阶段 - 稀疏投影（Sparse Projection）：** 对中间维度 k' 的子向量应用稀疏 Johnson-Lindenstrauss 变换（SJLT），进一步压缩到目标维度 k。SJLT 通过稀疏化投影矩阵减少计算量，并通过自定义 CUDA 内核优化实现，解决线程竞争和内存访问问题。
    *   **复杂度：** 整体时间和空间复杂度为 O(k')，显著低于传统随机投影方法的 O(pk)。
*   **F ACT G RA SS（Factorized G RA SS）改进：** 针对线性层的梯度结构（Kronecker Product Structure），提出三阶段压缩策略，避免显式构建完整梯度。
    *   **第一阶段 - 因子化稀疏化：** 对线性层的输入和预激活梯度分别进行稀疏化，得到维度为 k' 的子向量（k' 在 √k 到 √p 之间）。
    *   **第二阶段 - 重建（Reconstruction）：** 通过 Kronecker 乘积构建稀疏化后的梯度，维度为 (k')^2。
    *   **第三阶段 - 稀疏投影：** 对 (k')^2 维度的稀疏化梯度应用 SJLT，压缩到目标维度 k。
    *   **复杂度：** 整体时间和空间复杂度为 O((k')^2)，在实践中比现有最优方法 L O G RA（复杂度 O(√kp)）更快，尤其当 k' 选择合适时。
*   **关键创新：** 两方法均利用梯度稀疏性和结构特性，避免传统方法（如 FJLT 或 Gaussian Projection）的高计算开销，同时通过硬件优化提升效率。

## Experiment

*   **准确性（Accuracy）：** 通过线性数据建模分数（LDS）进行定量评估，G RA SS 在多个数据集（如 MNIST, CIFAR2, MAESTRO）和模型（如 MLP, ResNet9, Music Transformer）上与基线（如 FJLT, Gaussian Projection）相比，表现出相当甚至更高的归因精度；F ACT G RA SS 在 GPT2-small（WikiText 数据集）上保持了高 LDS 分数，在亿级规模模型 Llama-3.1-8B-Instruct 上，定性分析显示其识别的影响数据与查询内容高度相关，体现了良好的归因质量。
*   **效率（Efficiency）：** G RA SS 和 F ACT G RA SS 在计算时间上显著优于基线，例如 F ACT G RA SS 在 Llama-3.1-8B-Instruct 上的压缩吞吐量（Throughput）比最优基线 L O G RA 快 160%，整体缓存阶段（Cache Stage）吞吐量提升约 17%，且内存使用相当。
*   **实验设置合理性：** 实验覆盖了从小规模到亿级规模的模型和数据集（如 OpenWebText 子集），验证了方法的普适性和可扩展性；但 LDS 指标存在局限性（加性假设不完美），且亿级模型实验未使用完整预训练数据集，可能影响结果代表性。

## Further Thoughts

论文中利用梯度稀疏性和结构特性（如 Kronecker 结构）进行高效压缩的思路非常启发性，是否可以将这种稀疏性利用扩展到其他机器学习任务（如分布式训练或模型剪枝）中？此外，随机掩码（Random Mask）方法尽管简单却表现出意外的有效性，是否可以通过理论分析揭示过参数化模型中冗余信息的分布规律？另外，F ACT G RA SS 的因子化压缩是否能进一步应用于注意力层或其他复杂层结构，以提升更多模型类型的效率？