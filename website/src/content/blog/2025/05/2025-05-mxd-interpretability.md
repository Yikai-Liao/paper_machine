---
title: "Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders"
pubDatetime: 2025-05-27T15:55:55+00:00
slug: "2025-05-mxd-interpretability"
type: "arxiv"
id: "2505.21364"
score: 0.699294762901087
author: "grok-3-latest"
authors: ["James Oldfield", "Shawn Im", "Yixuan Li", "Mihalis A. Nicolaou", "Ioannis Patras", "Grigorios G Chrysos"]
tags: ["LLM", "Sparse Representation", "Mixture of Experts", "Interpretability", "Layer Decomposition"]
institution: ["University of Wisconsin–Madison", "Queen Mary University of London", "The Cyprus Institute"]
description: "本文提出 Mixture of Decoders (MxD) 方法，通过层级稀疏性和全秩权重设计实现语言模型密集层的可解释性分解，在稀疏性-准确性权衡上显著优于现有技术。"
---

> **Summary:** 本文提出 Mixture of Decoders (MxD) 方法，通过层级稀疏性和全秩权重设计实现语言模型密集层的可解释性分解，在稀疏性-准确性权衡上显著优于现有技术。 

> **Keywords:** LLM, Sparse Representation, Mixture of Experts, Interpretability, Layer Decomposition

**Authors:** James Oldfield, Shawn Im, Yixuan Li, Mihalis A. Nicolaou, Ioannis Patras, Grigorios G Chrysos

**Institution(s):** University of Wisconsin–Madison, Queen Mary University of London, The Cyprus Institute


## Problem Background

大型语言模型（LLMs）中的多层感知机（MLP）层由于其密集表示，难以理解、编辑和控制。
现有通过神经元级别稀疏性（neuron-level sparsity）学习可解释近似的方法，在重建原始映射时显著增加了模型的下一个token交叉熵损失，牺牲了性能。
论文旨在解决如何在不牺牲模型性能的前提下，通过层级稀疏性（layer-level sparsity）实现密集层的可解释性分解。

## Method

*   **核心思想:** 提出 Mixture of Decoders (MxD)，将预训练的密集层扩展为数万个专门子层（specialized sublayers），通过层级稀疏性实现可解释性，同时保持模型性能。
*   **具体实现:** 
    *   MxD 将 MLP 层近似为多个线性变换的条件组合，通过学习门控矩阵（gating matrix）生成稀疏的专家系数（expert coefficients），决定每个输入token激活哪些子层。
    *   使用灵活的张量分解（tensor factorization）和 Hadamard 乘积（Hadamard product）参数化权重张量，确保每个稀疏激活子层具有全秩权重（full-rank weights），保留原始解码器的表达能力。
    *   参数效率设计：通过将权重张量分解为两个矩阵（C 和 D）的组合，显著减少参数量（从 NHO 降至 O·(N+H)），支持大规模专家数量（tens of thousands）。
    *   泛化能力：MxD 适用于现代 MLP 变体（如 Gated Linear Units, GLU），通过理论证明每个专家子层的权重继承原始解码器的秩属性。
*   **关键优势:** 避免对隐藏单元施加稀疏性和非负性约束，通过大规模专家子层实现特征特化（feature specialization），在高稀疏性下仍能忠实重建原始映射。

## Experiment

*   **有效性:** 在四个语言模型（GPT2-124M, Pythia-410M, Pythia-1.4B, Llama-3.2-3B，参数高达 3B）上，MxD 在稀疏性-准确性前沿（sparsity-accuracy frontier）上显著优于现有方法（如 Transcoders 和 Skip Transcoders），在所有稀疏性水平（K=16至256）下保持更低的交叉熵损失，尤其在低 K 值时重建误差比 Transcoders 低一个数量级。
*   **全面性:** 实验设置涵盖不同规模模型和多种稀疏性水平，通过下游模型损失、归一化 MSE 等指标评估忠实性；同时在稀疏探测（sparse probing）和特征引导（feature steering）任务上与基线方法竞争力相当，验证了特征特化和可解释性。
*   **局限性:** 实验限于 3B 参数规模，计算资源限制了更大模型测试；推理时的大型编码器和门控函数带来额外计算成本。

## Further Thoughts

层级稀疏性（layer-level sparsity）的概念非常具有启发性，它挑战了传统的神经元级稀疏性范式，表明通过大规模专家子层实现特化可以在不牺牲性能的情况下提升可解释性；此外，MxD 的全秩权重设计和自然浮现的共享专家（shared expert）机制为未来混合专家模型（MoE）和条件计算架构的设计提供了新思路，或许可以探索如何将这种层级特化思想应用于其他深度学习领域，如视觉模型或多模态模型。