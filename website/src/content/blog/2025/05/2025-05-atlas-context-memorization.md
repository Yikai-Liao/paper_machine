---
title: "ATLAS: Learning to Optimally Memorize the Context at Test Time"
pubDatetime: 2025-05-29T17:57:16+00:00
slug: "2025-05-atlas-context-memorization"
type: "arxiv"
id: "2505.23735"
score: 0.7114916905045041
author: "grok-3-latest"
authors: ["Ali Behrouz", "Zeman Li", "Praneeth Kacham", "Majid Daliri", "Yuan Deng", "Peilin Zhong", "Meisam Razaviyayn", "Vahab Mirrokni"]
tags: ["LLM", "Long Context", "Memory Capacity", "Recurrent Models", "Feature Mapping"]
institution: ["Google"]
description: "本文提出 Atlas 和 DeepTransformers 家族，通过高容量记忆模块、上下文感知的 Omega 更新规则和 Muon 优化器，显著提升了长上下文任务中的性能，克服了 Transformer 和现代 RNN 的局限性。"
---

> **Summary:** 本文提出 Atlas 和 DeepTransformers 家族，通过高容量记忆模块、上下文感知的 Omega 更新规则和 Muon 优化器，显著提升了长上下文任务中的性能，克服了 Transformer 和现代 RNN 的局限性。 

> **Keywords:** LLM, Long Context, Memory Capacity, Recurrent Models, Feature Mapping

**Authors:** Ali Behrouz, Zeman Li, Praneeth Kacham, Majid Daliri, Yuan Deng, Peilin Zhong, Meisam Razaviyayn, Vahab Mirrokni

**Institution(s):** Google


## Problem Background

Transformer 模型由于注意力机制的二次时间和空间复杂度，在长上下文理解和记忆任务中面临显著挑战；现代循环神经网络（RNN）虽通过固定大小记忆模块尝试解决此问题，但在长上下文理解、上下文检索和序列外推方面仍表现不佳，原因包括记忆容量有限、在线更新性质（仅基于当前输入优化记忆）和记忆管理表达不足（内部优化器依赖一阶信息，易陷入次优解）。

## Method

* **核心思想**：设计一种高容量、上下文感知的长时记忆模块 Atlas，通过提升记忆容量、克服在线更新局限性和改进记忆管理，增强长上下文任务性能，同时构建 DeepTransformers 家族以泛化 Transformer 架构。
* **高容量记忆模块**：通过高阶特征映射（如多项式核）增加记忆容量，理论上证明深层记忆模块（如多层感知机 MLP）和高阶映射如何提升可存储的键值对数量，解决记忆容量瓶颈。
* **Omega 规则**：提出一种滑动窗口更新规则，在优化记忆时考虑过去一组令牌（而非仅当前令牌），通过上下文窗口内的加权损失（如基于衰减项 γ）更新记忆，从而记忆上下文而非单个令牌，克服在线更新的局限性。
* **高级记忆管理**：在 Atlas 中引入 Muon 优化器，近似二阶信息以避免局部最优，提升长上下文任务中的记忆质量，相比传统梯度下降更有效管理固定大小记忆。
* **DeepTransformers 家族**：基于上述创新，提出一系列 Transformer 变体，如 Deep Linear Attention (DLA) 使用深层记忆模块，Deep Omega Transformer (Dot) 结合 Omega 规则和指数核，严格泛化原始 Transformer 架构。
* **并行化训练**：通过分块计算和滑动窗口掩码策略，确保训练效率，避免显著的计算开销，支持大规模应用。

## Experiment

* **有效性**：Atlas 和 OmegaNet 在语言建模、常识推理、长上下文任务（如 Needle-in-a-Haystack 和 BABILong 基准）中显著优于 Transformer 和现代 RNN（如 Titans、DeltaNet），例如在 BABILong 10M 上下文长度任务中，Atlas 达到 +80% 准确率，而 Titans 性能下降明显。
* **提升显著性**：相比基线，Atlas 的改进源于上下文记忆能力、高容量设计和优化的记忆管理，尤其在长上下文任务中表现突出；DeepTransformers 家族也显示出对 Transformer 的严格改进，特别是在语言建模和长上下文任务中。
* **实验设置合理性**：实验覆盖多种模型规模（340M 到 1.3B 参数）、训练数据量（15B 到 100B 令牌）和任务类型（语言建模、常识推理、长上下文理解、上下文内回忆），消融研究验证了各组件（如多项式映射、Muon 优化器、滑动窗口规则）的贡献。
* **局限性**：实验未充分探讨极低资源环境下的性能（如小模型或少量训练数据），长上下文任务评估集中于特定基准，可能未完全反映真实世界应用的多样性。

## Further Thoughts

上下文记忆而非单个令牌记忆的理念可能启发未来模型在长序列任务中更注重语义整体性，而非局部信息；高阶特征映射（如多项式核）提升记忆容量的方法或可扩展到图神经网络或多模态学习领域；Muon 优化器的应用表明内部优化策略对记忆管理至关重要，是否可以引入自适应优化器或元学习技术进一步提升效率？此外，是否可以将上下文记忆与强化学习结合，动态调整上下文窗口大小以适应不同任务需求？