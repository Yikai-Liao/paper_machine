---
title: "Compressing Sine-Activated Low-Rank Adapters through Post-Training Quantization"
pubDatetime: 2025-05-28T02:15:15+00:00
slug: "2025-05-sine-lora-quantization"
type: "arxiv"
id: "2505.21895"
score: 0.7848716734700578
author: "grok-3-latest"
authors: ["Cameron Gordon", "Hemanth Saratchandran", "Yiping Ji", "Paul Albert", "Simon Lucey"]
tags: ["LLM", "Low-Rank Adaptation", "Quantization", "Model Compression", "Fine-Tuning"]
institution: ["Australian Institute for Machine Learning, University of Adelaide", "DATA61, CSIRO"]
description: "本文通过后训练量化结合固定频率正弦变换，显著提升了低秩适配器的表达能力，在语言、视觉和生成任务中实现了性能与内存效率的优越平衡。"
---

> **Summary:** 本文通过后训练量化结合固定频率正弦变换，显著提升了低秩适配器的表达能力，在语言、视觉和生成任务中实现了性能与内存效率的优越平衡。 

> **Keywords:** LLM, Low-Rank Adaptation, Quantization, Model Compression, Fine-Tuning

**Authors:** Cameron Gordon, Hemanth Saratchandran, Yiping Ji, Paul Albert, Simon Lucey

**Institution(s):** Australian Institute for Machine Learning, University of Adelaide, DATA61, CSIRO


## Problem Background

低秩适配器（LoRA）作为参数高效微调（PEFT）的一种标准方法，通过低秩矩阵分解显著减少了可训练参数，但其低秩约束限制了表达能力，导致性能不如全秩微调。
近期研究通过固定频率正弦变换提升了低秩适配器的稳定秩（Stable Rank），增强了表达能力，但这种方法在后训练量化（Post-Training Quantization, PTQ）环境下的适用性尚未被探索。
论文旨在解决量化后低秩适配器表达能力下降的问题，同时保持内存效率。

## Method

*   **核心思想:** 在后训练量化环境下，通过对量化后的低秩适配器应用固定频率正弦变换，提升其稳定秩（Stable Rank），从而增强表达能力，同时保留量化的压缩优势。
*   **具体实现:** 
    *   **正弦激活:** 基于 Ji 等人（2025）的工作，对低秩适配器的矩阵乘积 AB 应用正弦变换 sin(ωAB)，其中 ω 是固定频率参数，A 和 B 是低秩矩阵，这种无参数非线性变换能显著提升稳定秩。
    *   **后训练量化（PTQ）:** 在训练完成后，对适配器参数进行量化，将其映射到低精度离散值集（如 2-bit、3-bit），采用 k-means 量化方案以优化参数分布的适应性，减少内存占用。
    *   **量化后正弦激活:** 在量化后的矩阵乘积 Q(A)Q(B) 上应用正弦变换 sin(ωQ(A)Q(B))，以恢复或提升量化带来的表达能力损失。
    *   **理论支持:** 论文通过理论分析（Theorem 3.1）证明，量化矩阵的稳定秩受未量化矩阵稳定秩的控制，量化误差影响可控，为量化后正弦激活的有效性提供了依据。
*   **关键优势:** 正弦激活作为无参数变换，不增加计算复杂度或参数量，可作为即插即用组件，适用于多种量化场景。

## Experiment

*   **有效性:** 在大型语言模型（LLM）适配任务（如 LLaMA 3-8B 常识推理）中，SineLoRA 在量化后（2-bit 到 5-bit）始终优于标准 LoRA，例如 5-bit Rank 8 SineLoRA 性能超全精度 LoRA，内存仅占 33.5%；在视觉-语言模型（VLM，如 CLIP）和文本到图像生成（如 Stable Diffusion）任务中，SineLoRA 同样在量化后展现更高准确性和生成质量。
*   **显著性:** 数据显示 SineLoRA 在中低比特量化下性能提升显著，如 2-bit 时常识推理准确率从 LoRA 的 71.0% 提升至 73.7%，VLM 任务中 3-bit SineLoRA 准确率为 74.2%（LoRA 为 74.1%）；BD-Rate 分析表明 2-bit 时内存节省达 41.6%。
*   **实验设置合理性:** 实验覆盖 LLM、VLM 和生成任务，量化级别从 1-bit 到全精度，秩从 1 到 16，数据集包括常识推理、图像分类和生成任务；对比了 k-means 与均匀量化，并用 Bjøntegaard Delta 分析压缩效率，设置全面。
*   **局限性:** 极低比特（如 1-bit）量化下提升有限，未探索量化感知训练（QAT）的潜力。

## Further Thoughts

论文中稳定秩（Stable Rank）作为表达能力指标的概念令人启发，未来可探索其他非线性变换（如余弦或动态频率函数）是否也能提升稳定秩；此外，量化后正弦激活的即插即用特性提示我们可以在训练中嵌入量化感知的非线性干预机制，以进一步优化性能与压缩的平衡；适配器压缩在边缘设备分发中的应用潜力也值得关注，是否能结合差分压缩技术优化带宽需求？