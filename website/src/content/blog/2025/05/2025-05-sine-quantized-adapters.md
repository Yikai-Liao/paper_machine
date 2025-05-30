---
title: "Compressing Sine-Activated Low-Rank Adapters through Post-Training Quantization"
pubDatetime: 2025-05-28T02:15:15+00:00
slug: "2025-05-sine-quantized-adapters"
type: "arxiv"
id: "2505.21895"
score: 0.7848716734700578
author: "grok-3-latest"
authors: ["Cameron Gordon", "Hemanth Saratchandran", "Yiping Ji", "Paul Albert", "Simon Lucey"]
tags: ["Low-Rank Adaptation", "Quantization", "Model Compression", "Parameter Efficiency", "Fine-Tuning"]
institution: ["Australian Institute for Machine Learning, University of Adelaide", "DATA61, CSIRO"]
description: "本文通过在量化后的低秩适配器上应用固定频率正弦非线性变换，提升稳定秩以增强表达能力，在语言、视觉和生成任务中实现高效压缩与性能保持。"
---

> **Summary:** 本文通过在量化后的低秩适配器上应用固定频率正弦非线性变换，提升稳定秩以增强表达能力，在语言、视觉和生成任务中实现高效压缩与性能保持。 

> **Keywords:** Low-Rank Adaptation, Quantization, Model Compression, Parameter Efficiency, Fine-Tuning

**Authors:** Cameron Gordon, Hemanth Saratchandran, Yiping Ji, Paul Albert, Simon Lucey

**Institution(s):** Australian Institute for Machine Learning, University of Adelaide, DATA61, CSIRO


## Problem Background

低秩适配器（LoRA）作为参数高效微调（PEFT）的重要方法，通过低秩矩阵分解显著减少了可训练参数，但其低秩约束限制了表达能力，导致性能低于全秩微调；近期研究通过固定频率正弦变换提升了低秩适配器的稳定秩（Stable Rank），而本文进一步探讨在后训练量化（Post-Training Quantization, PTQ）场景下，如何通过正弦激活技术在模型压缩后仍保持表达能力，以解决量化带来的性能损失问题。

## Method

* **核心思想**：在后训练量化后的低秩适配器上应用固定频率正弦非线性变换，以提升稳定秩（Stable Rank），从而增强表达能力，同时保留量化的内存效率。
* **理论基础**：通过数学推导（Theorem 3.1），证明量化后适配器的稳定秩受全精度版本控制，表明低秩结构在量化后依然存在限制；因此，需通过正弦激活打破这一限制。
* **具体实现**：首先对低秩适配器（由两个低秩矩阵 A 和 B 的乘积 AB 构成）进行量化，使用 k-means 量化方法将参数映射到离散值集合（如 2-bit、3-bit 等）；随后在量化结果上应用正弦变换 sin(ωQ(A)Q(B))，其中 ω 为固定频率参数，γ 为缩放因子，用于提升稳定秩；此过程不引入额外参数，仅通过非线性变换增强表达能力。
* **优势**：正弦激活作为一种无参数增强手段，可作为即插即用组件嵌入量化框架，既不增加计算复杂度，又能有效弥补量化带来的表达能力损失。

## Experiment

* **有效性**：在语言模型（LLaMA 3-8B）常识推理任务中，SineLoRA 在 5-bit 量化下性能接近或超过全精度 LoRA，同时内存占用减少约 66%（如 Rank 8 时从 27.1MB 降至 9.1MB）；在视觉-语言模型（CLIP）任务中，SineLoRA 在 3-bit 及以上量化级别下持续优于 LoRA；在文本到图像生成（Stable Diffusion）任务中，SineLoRA 在低量化级别下保持更高的目标一致性（CLIP-I 和 DINO 指标提升）。
* **实验设置合理性**：实验覆盖了多种任务（语言、视觉、生成）、不同秩（Rank 1-16）和量化级别（1-bit 到 Full），并与基线方法（如 LoRA、DoRA）对比，使用 Bjøntegaard Delta 指标评估压缩效率，设置全面且数据可信。
* **局限性**：实验聚焦后训练量化，未探索量化感知训练（QAT）的潜力；量化方案未利用 GPU 优化数据类型（如 INT-4），可能限制实际部署效率。

## Further Thoughts

论文通过正弦非线性变换提升量化后适配器的稳定秩，这一思想启发我们探索其他非线性函数或动态频率调整策略，以进一步优化量化模型的表达能力；此外，稳定秩作为衡量表达能力的指标，可推广到知识蒸馏或模型剪枝等其他压缩方法中；论文提到的适配器分发场景也提示未来可研究自适应正弦频率或动态量化的个性化适配器分发策略，以适配不同设备和任务需求。