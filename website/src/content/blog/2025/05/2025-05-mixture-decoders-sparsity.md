---
title: "Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders"
pubDatetime: 2025-05-27T15:55:55+00:00
slug: "2025-05-mixture-decoders-sparsity"
type: "arxiv"
id: "2505.21364"
score: 0.699294762901087
author: "grok-3-latest"
authors: ["James Oldfield", "Shawn Im", "Yixuan Li", "Mihalis A. Nicolaou", "Ioannis Patras", "Grigorios G Chrysos"]
tags: ["LLM", "Sparse Representation", "Conditional Computation", "Interpretability", "Layer Decomposition"]
institution: ["University of Wisconsin–Madison", "Queen Mary University of London", "The Cyprus Institute"]
description: "本文提出 Mixture of Decoders (MxD) 方法，通过层级别稀疏性分解密集 MLP 层，在保持模型性能的同时显著提升可解释性。"
---

> **Summary:** 本文提出 Mixture of Decoders (MxD) 方法，通过层级别稀疏性分解密集 MLP 层，在保持模型性能的同时显著提升可解释性。 

> **Keywords:** LLM, Sparse Representation, Conditional Computation, Interpretability, Layer Decomposition

**Authors:** James Oldfield, Shawn Im, Yixuan Li, Mihalis A. Nicolaou, Ioannis Patras, Grigorios G Chrysos

**Institution(s):** University of Wisconsin–Madison, Queen Mary University of London, The Cyprus Institute


## Problem Background

大型语言模型（LLMs）中的多层感知机（MLP）层由于其密集表示，难以理解、编辑和控制，传统神经元级别稀疏性方法虽能提高可解释性，但往往以显著牺牲模型性能为代价，导致无法忠实重构原始映射；论文提出从神经元级别稀疏性转向层级别稀疏性，以解决稀疏性与准确性之间的权衡问题。

## Method

* **核心思想：** 提出 Mixture of Decoders (MxD)，通过层级别稀疏性将预训练的密集 MLP 层分解为数以万计的专门化子层，在保持原始模型性能的同时提升可解释性。
* **具体实现：** 
  * MxD 使用张量分解（tensor factorization）和 Hadamard 乘积构建参数高效的专家子层，每个子层执行全秩权重的线性变换，确保在高稀疏性下仍保留原始解码器的表达能力。
  * 通过一个门控矩阵（gating matrix）生成稀疏的专家系数（expert coefficients），并采用 Top-K 激活函数选择活跃专家子层，实现条件计算（conditional computation）。
  * 设计上，MxD 直接继承 MLP 层的功能形式，避免对隐藏单元施加稀疏性和非负性约束，同时能够泛化到现代 MLP 变体如 Gated Linear Units (GLUs)。
* **关键优势：** MxD 不需要外部后处理分析，直接将专门化特征融入模型前向传播中，且通过证明每个专家子层的权重继承原始解码器的秩，确保忠实重构能力。

## Experiment

* **有效性：** 在四个语言模型（GPT2-124M 到 Llama-3.2-3B）上的实验表明，MxD 在稀疏性-准确性前沿上显著优于现有方法（如 Transcoders），在高稀疏性下保持低交叉熵损失和更好的重构效果。
* **全面性：** 实验覆盖 108 个稀疏层，测试了不同稀疏性水平（通过调整活跃专家数量 K），并通过多种指标（如交叉熵损失、归一化 MSE）验证方法性能；此外，稀疏探测和特征引导实验表明 MxD 学习的特征与自然语言专门化概念高度相关，可解释性与基线相当。
* **局限性：** 由于资源限制，实验仅限于 3B 参数规模的模型，未能验证更大模型上的表现；同时，推理时门控函数和大型编码器仍带来额外计算成本。

## Further Thoughts

层级别稀疏性的概念为深度学习模型的可解释性研究开辟了新方向，未来可探索将其应用于其他架构（如视觉模型的卷积层或注意力机制），以在不同领域实现性能与可解释性的平衡；此外，MxD 通过张量分解实现参数高效性的大规模专家系统设计，对资源受限场景下的模型优化具有借鉴意义。