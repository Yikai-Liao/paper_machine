---
title: "Intra-Layer Recurrence in Transformers for Language Modeling"
pubDatetime: 2025-05-03T16:16:55+00:00
slug: "2025-05-intra-layer-recurrence"
type: "arxiv"
id: "2505.01855"
score: 0.7335161520232741
author: "grok-3-latest"
authors: ["Anthony Nguyen", "Wenjun Lin"]
tags: ["Transformer", "Language Modeling", "Recurrence", "Compute Efficiency", "Layer Optimization"]
institution: ["Algoma University"]
description: "本文提出 Intra-Layer Recurrence (ILR) 方法，通过在 Transformer 模型中选择性循环个别层，显著降低困惑度并验证早期层循环效果最佳，为高效架构设计提供了新思路。"
---

> **Summary:** 本文提出 Intra-Layer Recurrence (ILR) 方法，通过在 Transformer 模型中选择性循环个别层，显著降低困惑度并验证早期层循环效果最佳，为高效架构设计提供了新思路。 

> **Keywords:** Transformer, Language Modeling, Recurrence, Compute Efficiency, Layer Optimization

**Authors:** Anthony Nguyen, Wenjun Lin

**Institution(s):** Algoma University


## Problem Background

Transformer 模型在自然语言处理中取得了显著成功，但其深度和参数量的增加导致了计算和内存需求的急剧上升。
现有循环 Transformer 方法通过对整个模型或层块统一应用循环来增加有效深度，但缺乏细粒度控制，无法针对不同层的特性进行优化。
本文提出 Intra-Layer Recurrence (ILR)，旨在通过在单次前向传播中选择性地对个别层进行循环，探索哪些层从循环中获益最多，以在不增加参数量的情况下提升模型性能。

## Method

*   **核心思想:** 在 Transformer 模型的单次前向传播中，选择性地对特定层进行多次处理（循环），以增加有效深度并优化计算资源分配。
*   **具体实现:** 
    *   引入一个重用映射（Reuse Map）R = [r1, r2, ..., rL]，其中 rl 表示第 l 层的循环次数。
    *   在前向传播中，若某层 l 的 rl > 1，则该层的输出会作为输入再次传入该层进行处理，重复 rl 次，形成层内循环。
    *   在反向传播中，梯度会在每个循环步骤中累积，需注意避免梯度爆炸或消失的问题。
    *   该方法允许细粒度控制每层的计算量，基于假设不同层对循环的响应不同，尤其是早期层可能因其基础表示学习作用而获益更多。
*   **优势与创新:** 相比于之前的循环方法（如对整个模型或层块统一循环），ILR 提供了更精细的控制，允许根据层级特性分配计算资源，且不增加模型参数量。

## Experiment

*   **有效性:** ILR 在不增加参数量的情况下显著降低了语言建模的困惑度（Perplexity），尤其是在早期层应用更多循环时效果最佳。例如，在小规模模型（1.2M 参数）中，重用映射 [4, 2, 1, 1] 将 NoPE 的困惑度从 16.57 降至 14.62，ALiBi 从 14.38 降至 13.63。
*   **规律性与对比:** 实验表明，早期层循环对性能提升贡献最大，优于后期层循环，且部分 ILR 配置在困惑度上优于块循环（Block Recurrence）方法。
*   **计算成本:** 循环增加了计算开销，小规模模型的 FLOPs 从基线的 4.13×10^15 增加到 8.24×10^15（双倍深度配置），表明性能提升需以计算代价为代价。
*   **实验设置合理性:** 实验覆盖了小规模和大规模模型（100M 参数），测试了多种重用映射和位置编码方法（NoPE, RoPE, Learned Absolute PE, ALiBi），设置较为全面；但大规模模型训练步数受限（仅 3B token），可能未充分发挥潜力。

## Further Thoughts

ILR 揭示了 Transformer 层级结构的异质性，启发了一种计算资源动态分配的思路。未来是否可以设计自适应循环机制，根据输入内容复杂性或任务需求动态调整每层的循环次数？此外，是否可以将循环分配与注意力机制结合，探索‘注意力驱动的循环分配’策略，以进一步优化计算效率和性能？