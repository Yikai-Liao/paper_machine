---
title: "Parameter-Efficient Transformer Embeddings"
pubDatetime: 2025-05-04T21:47:18+00:00
slug: "2025-05-parameter-efficient-embedding"
type: "arxiv"
id: "2505.02266"
score: 0.8385831312376152
author: "grok-3-latest"
authors: ["Henry Ndubuaku", "Mouad Talhi"]
tags: ["Transformer", "Embedding Layer", "Parameter Efficiency", "Fourier Expansion", "Token Representation"]
institution: ["Cactus Compute, Inc.", "Imperial College"]
description: "本文提出一种参数高效的 Transformer 嵌入方法，通过傅里叶展开和轻量级 MLP 替代传统嵌入表，显著减少参数量并保持竞争性能，同时缩短训练时间。"
---

> **Summary:** 本文提出一种参数高效的 Transformer 嵌入方法，通过傅里叶展开和轻量级 MLP 替代传统嵌入表，显著减少参数量并保持竞争性能，同时缩短训练时间。 

> **Keywords:** Transformer, Embedding Layer, Parameter Efficiency, Fourier Expansion, Token Representation

**Authors:** Henry Ndubuaku, Mouad Talhi

**Institution(s):** Cactus Compute, Inc., Imperial College


## Problem Background

Transformer 模型中的嵌入层（Embedding Layer）通常占据参数量的最大比例，但性能提升与参数规模不成正比，存在稀疏性、冗余性及缺乏熵驱动压缩等问题。
作者旨在探索一种参数高效的嵌入机制，通过减少嵌入层的参数量来降低模型存储和计算成本，同时维持或接近传统嵌入层的性能，特别是在资源受限场景下具有重要应用价值。

## Method

*   **核心思想:** 将嵌入层从传统的参数存储（大型嵌入表）转变为动态计算生成，通过确定性函数和少量可学习参数生成 token 嵌入向量，显著减少参数量。
*   **具体实现:** 
    *   **归一化处理:** 将 token ID（基于 BPE 编码的频率排序）归一化到连续区间 [-1, 1]，保留其统计特性，为后续函数逼近提供连续输入。
    *   **傅里叶展开:** 使用傅里叶基函数（正弦和余弦）对归一化后的 token ID 进行展开，生成初始嵌入向量。傅里叶基的选择基于其函数逼近能力（可逼近任意连续函数）、特征正交性（减少初始特征相关性）以及硬件优化适配性（易于 GPU/TPU 加速）。低阶项捕捉全局统计趋势，高阶项捕捉细粒度差异。
    *   **MLP 细化:** 在傅里叶嵌入基础上，引入一个轻量级的多层感知机（MLP），通过非线性变换捕捉高阶交互，并结合残差连接（Residual Connection）调整初始嵌入，使其适应下游任务需求。残差连接有助于优化，通过学习相对于基础嵌入的微调来简化训练。
*   **关键优势:** 避免存储大型嵌入表（V × d 矩阵），参数量与词汇量无关；利用 token ID 的统计结构（如频率排序）提供初始信息；通过自定义 CUDA 内核加速傅里叶计算，减少训练和推理开销。

## Experiment

*   **有效性:** 在自然语言推理任务（SNLI 和 MNLI）上训练，并在句子文本相似性任务（STS-B）上进行零样本评估，傅里叶嵌入在参数量显著减少的情况下（例如 1.1M vs 8.9M）仍接近传统嵌入性能（STS-B Spearman 相关系数 74.93 vs 77.01），特别是在模型规模增加时（2 层，512 维度）性能差距缩小至几乎可忽略（77.40 vs 77.54）。
*   **参数效率与速度:** 傅里叶嵌入的参数量仅为传统嵌入的 10%-20%，训练时间明显缩短（例如 37.88 分钟 vs 48.48 分钟），得益于自定义 CUDA 内核优化。
*   **微调表现:** 在 STS-B 微调后，极小规模的 PETE 模型（58k 参数）达到 69.0 的 Pearson 相关系数，较大规模 PETE（3.6M 参数）甚至超越 BERT-Tiny 等模型（81.7 vs 74.3），显示出高参数效率。
*   **实验设置合理性与局限:** 实验覆盖不同模型规模（1-2 层，256-512 维度）和任务类型，但由于资源限制，规模较小，未在大规模数据集或复杂任务上验证，存在一定局限性。

## Further Thoughts

嵌入层从‘存储’到‘计算’的范式转变非常具有启发性，这种通过确定性函数（如傅里叶基）结合少量可学习参数生成嵌入的思路，不仅适用于 Transformer 模型，也可能推广到图嵌入或推荐系统等领域；此外，token ID 的统计结构（频率排序）被有效利用，启发我们是否可以进一步挖掘其他潜在信息（如语义聚类）来改进嵌入生成；最后，傅里叶基在硬件优化上的优势提示模型设计应更多考虑硬件适配性，例如通过特定内核加速计算密集型操作。