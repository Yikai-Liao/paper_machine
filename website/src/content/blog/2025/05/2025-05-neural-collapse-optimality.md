---
title: "Neural Collapse is Globally Optimal in Deep Regularized ResNets and Transformers"
pubDatetime: 2025-05-21T08:16:03+00:00
slug: "2025-05-neural-collapse-optimality"
type: "arxiv"
id: "2505.15239"
score: 0.5266698169527538
author: "grok-3-latest"
authors: ["Peter Súkeník", "Christoph H. Lampert", "Marco Mondelli"]
tags: ["Neural Collapse", "Deep Learning", "Feature Representation", "Regularization", "Architecture Design"]
institution: ["Institute of Science and Technology (ISTA)"]
description: "本文证明了神经崩溃在深层正则化的 ResNets 和 Transformers 中是全局最优的，并通过实验验证了深度增加对神经崩溃的促进作用。"
---

> **Summary:** 本文证明了神经崩溃在深层正则化的 ResNets 和 Transformers 中是全局最优的，并通过实验验证了深度增加对神经崩溃的促进作用。 

> **Keywords:** Neural Collapse, Deep Learning, Feature Representation, Regularization, Architecture Design

**Authors:** Peter Súkeník, Christoph H. Lampert, Marco Mondelli

**Institution(s):** Institute of Science and Technology (ISTA)


## Problem Background

本文研究了深度神经网络训练后期特征表示中出现的神经崩溃（Neural Collapse, NC）现象，即同一类样本的特征向量收敛到类均值，类均值形成等角紧框架，并与最后一层权重对齐；现有理论研究多局限于数据无关模型或简单多层感知机，而本文旨在填补这一空白，探索现代架构如 ResNets 和 Transformers 在数据感知场景下的神经崩溃是否为全局最优。

## Method

* **核心思想**：通过将端到端训练的深层 ResNets 和 Transformers 简化为等价的无约束特征模型（Unconstrained Features Model, UFM），分析神经崩溃的全局最优性。
* **具体实现**：
  - 定义了带有 LayerNorm 的 ResNets 和 Transformers 的结构，分为单线性层（L-RN1, L-Tx1）和双线性层（L-RN2, L-Tx2）两种变体。
  - 理论上证明了在深度趋于无穷大时，单线性层架构在恒定正则化强度下全局最优解趋向于神经崩溃；双线性层架构在正则化强度随深度递减的条件下也成立。
  - 通过数学推导，将深层网络的训练目标简化为广义无约束特征模型（Generalized UFM），并证明其全局最优解满足神经崩溃的三个指标（NC1, NC2, NC3 趋于 0）。
  - 对数据分布和架构设计做最小假设，如样本唯一性和上下文标签确定性，确保理论适用性。
* **创新点**：首次在现代架构中提供端到端训练下神经崩溃的理论保证，并为 UFM 的广泛使用提供理论依据。

## Experiment

* **有效性**：在计算机视觉数据集（MNIST, CIFAR10）和语言数据集（IMDB）上，实验表明随着网络深度增加（从 2 到 34 层），神经崩溃指标（NC1, NC2, NC3）显著改善，特征表示更接近理想的神经崩溃状态。
* **合理性**：实验覆盖多种架构（ResNets 和 Transformers），在不同随机种子下重复，确保结果稳健；同时，实验结果与理论预测一致，表明深度增加促进神经崩溃。
* **局限性**：实验基于梯度下降找到的解，而非严格全局最优，但仍支持理论结论，显示理论在实际训练中的指导意义。

## Further Thoughts

论文启发我们可以通过简化为无约束特征模型（UFM）来分析复杂深层网络行为，这种方法不仅适用于分类任务中的神经崩溃，还可能扩展到语言建模或不平衡数据场景；此外，网络深度作为关键参数，通过增加深度增强特征表示的对称性和可解释性，为设计更深层次网络提供了理论支持；双线性层架构在恒定正则化下的未解行为也提示未来可探索正则化与深度之间的权衡。