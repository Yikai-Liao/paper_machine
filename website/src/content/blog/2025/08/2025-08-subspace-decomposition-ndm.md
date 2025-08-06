---
title: "Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning"
pubDatetime: 2025-08-03T20:59:29+00:00
slug: "2025-08-subspace-decomposition-ndm"
type: "arxiv"
id: "2508.01916"
score: 0.5712112000543565
author: "grok-3-latest"
authors: ["Xinting Huang", "Michael Hahn"]
tags: ["Neural Network", "Representation Space", "Subspace Decomposition", "Unsupervised Learning", "Interpretability"]
institution: ["Saarland University"]
description: "本文提出了一种无监督学习方法（NDM），通过邻居距离最小化将神经网络表示空间分解为可解释且独立的子空间，为机械解释性研究提供了新工具。"
---

> **Summary:** 本文提出了一种无监督学习方法（NDM），通过邻居距离最小化将神经网络表示空间分解为可解释且独立的子空间，为机械解释性研究提供了新工具。 

> **Keywords:** Neural Network, Representation Space, Subspace Decomposition, Unsupervised Learning, Interpretability

**Authors:** Xinting Huang, Michael Hahn

**Institution(s):** Saarland University


## Problem Background

神经网络的表示空间由于高维度特性，编码了输入的多种信息，但这些信息是否以‘自然’的方式组织在不同子空间中尚不明确；传统解释性方法（如稀疏自编码器或监督子空间方法）依赖输入或人工指定因果模型，存在局限性；本文旨在通过无监督学习分解表示空间，找到可解释且独立的子空间，以深入理解模型内部机制。

## Method

* **核心思想：** 通过无监督学习方法，将神经网络的表示空间分解为正交且可解释的子空间，使得每个子空间编码一组互斥特征，子空间间尽可能独立。
* **具体实现：** 提出邻居距离最小化（Neighbor Distance Minimization, NDM）方法，通过优化一个正交矩阵 **R**，旋转并划分表示空间；具体步骤包括：
  * 给定模型激活数据，初始化正交矩阵 **R** 和子空间维度配置。
  * 通过 **R** 变换激活数据，将其投影到子空间中，计算每个子空间内数据点到最近邻的距离。
  * 优化 **R** 以最小化子空间内最近邻距离总和，减少子空间间总相关性（Total Correlation）。
  * 使用互信息（Mutual Information, MI）估计动态调整子空间维度配置，合并互信息高的子空间，确保独立性。
* **理论依据：** NDM 基于特征组互斥性假设，即同一特征组内特征互斥，不同组间独立；通过最小化邻居距离，子空间内数据点投影集中于少数方向，间接减少子空间间依赖。
* **关键特点：** 无需监督信号，仅依赖模型激活数据；通过正交变换捕捉神经表示的分布式特性；动态调整子空间配置以适应真实模型的复杂结构。

## Experiment

* **有效性：** 在玩具模型中，NDM 成功找回预设正交子空间，验证了方法在理想条件下的正确性；在 GPT-2 Small 上，NDM 的 Gini 系数（衡量信息集中度）平均达 0.71，显著高于基线（如 PCA 的 0.38），表明目标信息（如前一词、位置）集中在少数子空间；在更大模型（如 Qwen2.5-1.5B 和 Gemma-2-2B）上，NDM 能分离上下文和参数知识路由的子空间，展现扩展性。
* **实验设置合理性：** 实验涵盖玩具模型、GPT-2 Small 和大模型，从简单到复杂全面验证方法；定量评估使用子空间激活补丁和 Gini 系数，定性分析通过 InversionView 展示子空间编码信息的一致性；对比多种基线（如随机矩阵、PCA），验证提升显著。
* **局限与开销：** 方法计算开销主要来自邻居距离计算和正交矩阵优化，论文未详细报告具体成本；子空间划分不够细粒度，可能错过小规模变量编码，需进一步优化。

## Further Thoughts

论文提出的子空间作为解释性基本单元的思路非常启发性，特别是在构建‘子空间电路’以分析跨层连接时；这让我思考是否可以进一步探索子空间的层次结构，例如将一个编码‘当前词’的子空间细分为‘语义’和‘语法角色’子子空间，以更精细地揭示模型计算逻辑；此外，NDM 的无监督特性是否可以与其他无监督方法（如聚类或生成对抗网络）结合，以增强子空间独立性和可解释性？