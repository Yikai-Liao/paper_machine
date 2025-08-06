---
title: "Trainable Dynamic Mask Sparse Attention"
pubDatetime: 2025-08-04T07:05:15+00:00
slug: "2025-08-dynamic-mask-attention"
type: "arxiv"
id: "2508.02124"
score: 0.6631676673038202
author: "grok-3-latest"
authors: ["Jingze Shi", "Yifan Wu", "Bingheng Wu", "Yiran Peng", "Liangdong Wang", "Guang Liu", "Yuyu Luo"]
tags: ["LLM", "Sparse Attention", "Long Context", "Content Aware", "Position Aware"]
institution: ["HKUST(GZ)", "BAAI", "SmallDoges"]
description: "本文提出动态掩码注意力（DMA），通过内容感知和位置感知的稀疏性设计，在保留信息完整性的同时显著降低计算复杂度，有效提升大型语言模型的长上下文建模能力。"
---

> **Summary:** 本文提出动态掩码注意力（DMA），通过内容感知和位置感知的稀疏性设计，在保留信息完整性的同时显著降低计算复杂度，有效提升大型语言模型的长上下文建模能力。 

> **Keywords:** LLM, Sparse Attention, Long Context, Content Aware, Position Aware

**Authors:** Jingze Shi, Yifan Wu, Bingheng Wu, Yiran Peng, Liangdong Wang, Guang Liu, Yuyu Luo

**Institution(s):** HKUST(GZ), BAAI, SmallDoges


## Problem Background

大型语言模型（LLMs）在处理长上下文任务时，传统自注意力机制因其二次方计算复杂度（O(n²)）而面临显著的计算和内存瓶颈。
随着对长上下文建模需求的增加（如深度推理、代码生成、多轮对话代理），现有稀疏注意力机制虽提升了效率，但仍存在静态模式或信息丢失的问题。
本文旨在设计一种高效的注意力机制，在保持信息完整性的同时显著降低计算复杂度。

## Method

*   **核心思想:** 提出动态掩码注意力（Dynamic Mask Attention, DMA），通过内容感知和位置感知的双重稀疏性设计，实现高效长上下文建模。
*   **内容感知动态掩码:** 从值表示（Value Representations）中动态生成稀疏掩码，自适应识别并聚焦于关键信息，而非依赖静态规则。
    *   通过值矩阵、采样权重矩阵和门控参数计算动态注意力权重，利用零阶保持方法确保权重稳定性，并通过非负函数强调注意力信号。
    *   结合因果掩码和top-k操作生成最终掩码，确保每个注意力头有特定的注意力模式。
*   **位置感知稀疏计算:** 基于动态掩码进行注意力权重计算，跳过被掩码位置的计算（直接置为0），从而降低计算复杂度。
    *   对每个注意力头，计算查询和键的缩放点积，并与掩码进行元素级操作，通过softmax生成注意力权重，最终与值向量加权求和。
    *   设计了块级稀疏计算优化，结合Flash Attention的流式计算策略，若整个块被掩码则跳过计算，显著减少FLOPs。
*   **关键特性:** 保留完整键-值缓存（KV Cache），避免信息压缩带来的瓶颈；设计专用计算内核，在硬件层面优化稀疏计算；通过完全可微设计确保训练和推理一致性，支持端到端学习最优稀疏模式。

## Experiment

*   **有效性:** 在预训练困惑度测试中，DMA在不同参数规模（80M到1.7B）下优于多头注意力（MHA）、滑动窗口注意力（SWA）、多头潜在注意力（MLA）和原生稀疏注意力（NSA）。
*   **长上下文能力:** 在多查询关联回忆任务中，DMA在长序列和小模型维度设置下表现出色，显示出精准定位关键信息的能力；在‘针在干草堆’任务中，DMA展现出更强的长度外推能力，超出预训练序列长度时性能下降较小。
*   **效率提升:** 推理速度上，DMA在长序列（4096及以上）逐渐赶超SWA；内核加速测试中，DMA的CUDA实现相较SDPA基准在长序列上实现高达15.5倍加速。
*   **下游任务:** 在1.7B参数模型测试中，DMA在大多数下游任务（如MMLU、TriviaQA）上优于MHA和NSA，整体性能提升显著。
*   **实验设置合理性:** 实验覆盖了模型规模、任务类型（语言建模、信息检索、下游任务）、硬件实现等多维度，设置全面，结果数据支持结论。

## Further Thoughts

DMA的可训练稀疏性设计启发我们可以在模型架构初期就嵌入稀疏性，而非依赖后处理优化；其内容与位置双重稀疏机制揭示了语言建模任务中的固有稀疏模式，提示我们可以在多模态任务中探索类似模式；此外，DMA在长度外推能力上的表现表明位置编码可能是长上下文建模的关键瓶颈，未来可以尝试将动态采样与位置信息结合，设计更具外推性的编码方案。