---
title: "One-Layer Transformers are Provably Optimal for In-context Reasoning and Distributional Association Learning in Next-Token Prediction Tasks"
pubDatetime: 2025-05-21T01:26:44+00:00
slug: "2025-05-one-layer-transformer-ntp"
type: "arxiv"
id: "2505.15009"
score: 0.6429594325984338
author: "grok-3-latest"
authors: ["Quan Nguyen", "Thanh Nguyen-Tang"]
tags: ["LLM", "Transformer", "In-context Learning", "Next-Token Prediction", "Generalization"]
institution: ["University of Victoria, Canada", "Johns Hopkins University, USA"]
description: "本文通过再参数化证明单层 Transformer 在下一词预测任务中能以线性收敛率达到贝叶斯最优，并展现出对未见样本的泛化能力。"
---

> **Summary:** 本文通过再参数化证明单层 Transformer 在下一词预测任务中能以线性收敛率达到贝叶斯最优，并展现出对未见样本的泛化能力。 

> **Keywords:** LLM, Transformer, In-context Learning, Next-Token Prediction, Generalization

**Authors:** Quan Nguyen, Thanh Nguyen-Tang

**Institution(s):** University of Victoria, Canada, Johns Hopkins University, USA


## Problem Background

大型语言模型（LLMs）中的 Transformer 架构在上下文推理（In-context Reasoning）和分布关联（Distributional Association）任务中表现出色，但对其自注意力机制的理论理解仍不充分，尤其是在下一词预测（Next-Token Prediction, NTP）任务中的训练动态和泛化能力方面存在研究空白。
现有研究多关注初始梯度步骤或无限样本场景，无法反映实际训练过程中的收敛性和对未见样本的适应能力。
本文旨在探究单层 Transformer 是否能在 NTP 任务中实现贝叶斯最优性能，并分析其训练收敛性和泛化能力。

## Method

*   **核心思想:** 通过特定的再参数化（Reparameterization）策略，设计单层 Transformer 的权重矩阵，使其逼近贝叶斯最优预测器，并在训练中实现线性收敛，同时具备泛化能力。
*   **模型设计:** 针对单层解码器 Transformer，包含一个注意力层和一个前馈层，采用线性注意力（Linear Attention）和 ReLU 注意力（ReLU Attention）两种机制。关键权重矩阵（如联合查询-键矩阵 W）被再参数化为触发词的关联记忆形式，输出嵌入矩阵 U 和值矩阵 V 也被特定初始化以支持上下文推理。
*   **训练方法:** 使用归一化梯度下降（Normalized Gradient Descent）算法训练模型，确保收敛性分析不依赖于数据分布的具体假设（Distribution-agnostic）。
*   **数据模型:** 设计合成数据模型，包含无噪声和有噪声两种设置，分别模拟纯上下文推理和带噪声的分布关联任务，确保至少有一个触发词-输出词二元组出现在句子中以提高信噪比。
*   **理论分析:** 证明在无噪声设置下，人口损失（Population Loss）可收敛至零；在有噪声设置下，损失逼近贝叶斯风险（Bayes Risk），并通过有限样本分析展示收敛率和泛化能力。

## Experiment

*   **有效性:** 实验表明，完全再参数化的单层 Transformer 在无噪声和有噪声设置下均能达到贝叶斯最优性能，人口损失以线性速率收敛，与理论预测一致。
*   **泛化能力:** 只有完全再参数化的模型能有效泛化到未见输出词，测试损失稳定；而未再参数化或部分再参数化的模型在未见输出词上的测试损失甚至发散，显示出泛化能力的显著差异。
*   **实验设置:** 数据模型设置词汇量 N=60，嵌入维度 d=128，上下文长度 H=256，噪声水平 α 取 0.2、0.5、0.8，训练样本量 M=2048，测试不同注意力类型（Linear, ReLU, Softmax）和学习率（0.01-0.5），设置较为全面。
*   **局限性:** 实验基于合成数据，数据模型较为简单，可能无法完全反映真实语言数据的复杂性，但足以验证理论结果的稳健性。
*   **功能分离:** 实验还验证了注意力层倾向于预测输出词，前馈层倾向于预测噪声词，符合理论预期。

## Further Thoughts

论文通过再参数化将 Transformer 权重矩阵设计为关联记忆的形式，分离注意力层和前馈层的学习功能，这一思想启发我们思考是否可以在多层 Transformer 中进一步设计层级功能分离机制，以提升模型对不同任务的适应性和可解释性；此外，泛化能力的关注提示未来研究可探索通过正则化或结构化约束，在训练中自然偏向泛化，而非依赖特定参数化策略。