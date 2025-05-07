---
title: "Parameter-Efficient Transformer Embeddings"
pubDatetime: 2025-05-04T21:47:18+00:00
slug: "2025-05-parameter-efficient-embedding"
type: "arxiv"
id: "2505.02266"
score: 0.8385831312376152
author: "grok-3-latest"
authors: ["Henry Ndubuaku", "Mouad Talhi"]
tags: ["LLM", "Embedding Layer", "Parameter Efficiency", "Fourier Transform", "Token Representation"]
institution: ["Cactus Compute, Inc.", "Imperial College"]
description: "本文提出一种参数高效的 Transformer 嵌入方法，通过傅里叶展开和轻量 MLP 替代传统嵌入矩阵，显著减少参数量和训练时间，同时在小规模实验中保持竞争性能。"
---

> **Summary:** 本文提出一种参数高效的 Transformer 嵌入方法，通过傅里叶展开和轻量 MLP 替代传统嵌入矩阵，显著减少参数量和训练时间，同时在小规模实验中保持竞争性能。 

> **Keywords:** LLM, Embedding Layer, Parameter Efficiency, Fourier Transform, Token Representation

**Authors:** Henry Ndubuaku, Mouad Talhi

**Institution(s):** Cactus Compute, Inc., Imperial College


## Problem Background

Transformer 模型中的嵌入层通常占据大量参数（随词汇量线性增长），但性能提升与参数规模不成正比，存在稀疏性、冗余性和缺乏熵驱动压缩等问题。
作者旨在设计一种参数高效的嵌入机制，利用 token ID 的统计特性（由 Byte-Pair Encoding 赋予的频率顺序），替代传统嵌入矩阵，减少存储和计算开销，同时维持模型性能。

## Method

*   **核心思想:** 将 token 嵌入视为从归一化 token ID 到高维向量的函数映射，通过确定性数学变换（傅里叶展开）和少量可学习组件（MLP）生成嵌入向量，替代传统的大型嵌入矩阵。
*   **具体实现:**
    *   **归一化:** 将 token ID 映射到连续区间 [-1, 1]，保留其相对顺序和统计特性。
    *   **傅里叶展开:** 使用 sine 和 cosine 基函数对归一化 ID 进行展开，生成初始嵌入向量，捕捉低频（全局趋势）和高频（细节）信息。傅里叶基的选择基于其正交性和函数逼近能力，且易于硬件优化。
    *   **MLP 调整:** 引入轻量级多层感知机（MLP），对傅里叶特征进行非线性变换，通过残差连接保留原始特征，确保优化稳定并适应下游任务需求。
*   **优势:** 参数量与词汇量无关，仅依赖 MLP 规模；通过定制 CUDA 内核实现高效计算；避免存储大型嵌入矩阵，减少内存占用。

## Experiment

*   **有效性:** 在自然语言推理任务（SNLI, MNLI）训练并在句子文本相似性任务（STS-B）零样本评估中，傅里叶嵌入方法（PETE）在相同容量下性能接近传统 Transformer（例如 2 层 d_model=512 时，Spearman 相关系数 77.40 vs 77.54），但参数量大幅减少（8.9M vs 24.3M）。
*   **参数效率:** PETE 在极小参数量（58k）时仍具合理性能（Spearman 69.5），在 3.6M 参数时性能（81.9）超越多个小型 BERT 变体（如 BERT-Tiny 的 73.6）。
*   **训练速度:** 得益于定制 CUDA 内核和无嵌入矩阵设计，PETE 训练时间普遍更短（例如 2 层 d_model=512 时，2.27 小时 vs 2.982 小时）。
*   **设置合理性与局限:** 实验控制变量合理（仅嵌入层不同），但规模较小（单张 RTX 4090），任务范围有限（仅 NLI 和 STS-B），未测试超大词汇量或多样化任务，可能无法完全反映大规模场景表现。

## Further Thoughts

傅里叶基函数的使用启发了对嵌入层设计的新思考：是否可探索小波变换等其他基函数以捕捉 token ID 的局部特性？
确定性嵌入生成是否能与预训练模型结合，通过冻结部分参数进一步减少训练开销？
傅里叶嵌入的可解释性潜力是否可通过可视化频率分布揭示模型语义组织？
此外，这种参数高效方法是否可与量化或剪枝结合，适配边缘设备部署？