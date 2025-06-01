---
title: "ATLAS: Learning to Optimally Memorize the Context at Test Time"
pubDatetime: 2025-05-29T17:57:16+00:00
slug: "2025-05-atlas-context-memorization"
type: "arxiv"
id: "2505.23735"
score: 0.7114916905045041
author: "grok-3-latest"
authors: ["Ali Behrouz", "Zeman Li", "Praneeth Kacham", "Majid Daliri", "Yuan Deng", "Peilin Zhong", "Meisam Razaviyayn", "Vahab Mirrokni"]
tags: ["Long Context Modeling", "Memory Capacity", "Recurrent Neural Network", "Transformer Generalization", "Test Time Memorization"]
institution: ["Google"]
description: "本文提出 Atlas，一种高容量长期内存模块，通过上下文记忆优化和先进优化器显著提升长上下文理解性能，超越 Transformer 和现代 RNN。"
---

> **Summary:** 本文提出 Atlas，一种高容量长期内存模块，通过上下文记忆优化和先进优化器显著提升长上下文理解性能，超越 Transformer 和现代 RNN。 

> **Keywords:** Long Context Modeling, Memory Capacity, Recurrent Neural Network, Transformer Generalization, Test Time Memorization

**Authors:** Ali Behrouz, Zeman Li, Praneeth Kacham, Majid Daliri, Yuan Deng, Peilin Zhong, Meisam Razaviyayn, Vahab Mirrokni

**Institution(s):** Google


## Problem Background

Transformer 模型因其二次时间和空间复杂度在长上下文理解任务中受限，而现代循环神经网络（RNN）虽在效率上有所改进，但由于内存容量有限、在线更新性质（仅基于当前输入优化内存）和内存管理表达能力不足，导致在长上下文理解和序列外推能力上表现不佳。
论文旨在解决三大关键问题：1）内存容量有限，无法存储足够多的键值对映射；2）在线更新导致无法有效捕捉上下文信息，仅记忆单个 token 而非整体上下文；3）内存管理不够强大，优化器依赖一阶信息，易陷入局部最优，影响长上下文性能。

## Method

* **核心思想**：提出 Atlas，一种高容量长期内存模块，通过在测试时学习记忆上下文（而非单个 token）克服现有 RNN 的局限性，并构建 DeepTransformers 作为 Transformer 的严格泛化。
* **内存容量提升**：对输入 token 应用高阶特征映射（如多项式核），增加内存模块的有效维度，理论上证明深层内存和多项式映射能显著提高存储独立键值对的能力。
* **上下文记忆优化**：设计 Omega 规则，一种滑动窗口更新规则，基于当前和过去 token 的上下文窗口优化内存，而非仅依赖当前输入，避免在线更新的局限性，允许模型记忆局部上下文而非单个 token。
* **内存管理改进**：在 Atlas 中引入 Muon 优化器，近似二阶信息优化内存映射，避免传统梯度下降陷入局部最优，提升长上下文任务中的性能。
* **DeepTransformers 架构**：将上述改进应用于 Transformer，提出 DeepTransformers 和 Deep Omega Transformer (Dot)，通过深层内存和 Omega 规则泛化传统注意力机制，增强长序列建模能力。
* **并行训练优化**：设计滑动窗口掩码策略和分块计算方法，确保 Omega 规则在训练时的高效性和可并行性，与在线版本相比无显著额外开销。

## Experiment

* **有效性**：Atlas 和 OmegaNet 在语言建模、常识推理、长上下文理解（如 BABILong 基准）和针式检索任务中显著优于 Transformer 和现代线性 RNN（如 Titans, DeltaNet），例如在 BABILong 10M 上下文长度任务中，Atlas 达到 80% 以上准确率，而 Titans 性能明显下降。
* **优越性**：相比基线，Atlas 的上下文记忆能力（通过 Omega 规则）和高容量内存（通过多项式核）使其在长上下文任务中表现突出；DeepTransformers 和 Dot 也显示出对传统 Transformer 的改进，尤其在长序列中。
* **实验设置合理性**：实验覆盖多种任务（语言建模、常识推理、长上下文理解）和数据集（FineWeb, Wikitext, BABILong 等），模型规模从 340M 到 1.3B 参数，训练 token 量从 15B 到 100B，设置全面；消融研究验证了各组件（如 Muon 优化器、多项式映射）的贡献。
* **开销**：通过并行训练策略，Atlas 的训练开销与在线版本（上下文窗口=1）相比无显著增加，但推理时可能因 Muon 优化器的多步计算（如 Newton-Schulz 迭代）带来额外计算负担。

## Further Thoughts

Omega 规则通过滑动窗口优化上下文记忆的思路可扩展至强化学习中的长期依赖建模或多模态任务中的跨模态上下文记忆，例如在视频理解中记忆‘事件片段’而非单个帧；此外，Atlas 使用 Muon 优化器的‘测试时计算’参数启发动态调整计算资源分配的可能性，根据任务复杂度选择优化步数，在性能与效率间找到最佳平衡；最后，多项式核提升内存容量的策略可进一步探索其他核函数（如 RBF 核）或自适应特征映射机制，根据输入特性动态调整映射方式。