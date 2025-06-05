---
title: "MoDA: Modulation Adapter for Fine-Grained Visual Grounding in Instructional MLLMs"
pubDatetime: 2025-06-02T16:38:50+00:00
slug: "2025-06-moda-visual-grounding"
type: "arxiv"
id: "2506.01850"
score: 0.5589708242040792
author: "grok-3-latest"
authors: ["Wayner Barrios", "Andres Villa", "Juan Leon Alcazar", "SouYoung Jin", "Bernard Ghanem"]
tags: ["LLM", "Multimodal Learning", "Visual Grounding", "Cross-Attention", "Instruction Tuning"]
institution: ["Dartmouth College", "King Abdullah University of Science and Technology (KAUST)"]
description: "本文提出 MoDA（Modulation Adapter），一个轻量级模块，通过语言指令引导的跨注意力调制优化视觉特征，显著提升多模态大语言模型在细粒度视觉 grounding 上的性能并减少幻觉现象。"
---

> **Summary:** 本文提出 MoDA（Modulation Adapter），一个轻量级模块，通过语言指令引导的跨注意力调制优化视觉特征，显著提升多模态大语言模型在细粒度视觉 grounding 上的性能并减少幻觉现象。 

> **Keywords:** LLM, Multimodal Learning, Visual Grounding, Cross-Attention, Instruction Tuning

**Authors:** Wayner Barrios, Andres Villa, Juan Leon Alcazar, SouYoung Jin, Bernard Ghanem

**Institution(s):** Dartmouth College, King Abdullah University of Science and Technology (KAUST)


## Problem Background

多模态大语言模型（MLLMs）在指令跟随任务中表现出色，但由于常用视觉编码器（如 CLIP）基于 patch 的表示方式难以捕捉局部细节，导致在复杂场景中对细粒度视觉概念的 grounding 能力不足，经常产生与图像语义不一致的‘幻觉’（hallucination），影响模型在现实场景中的可靠性和安全性。
本文旨在解决如何在不改变底层架构、不增加额外监督的情况下，提升 MLLMs 对细粒度视觉信息的处理能力并减少幻觉现象。

## Method

*   **核心思想:** 提出 MoDA（Modulation Adapter），一个轻量级模块，通过语言指令引导的调制（modulation）动态优化预对齐的视觉特征，突出与任务相关的视觉信息。
*   **具体实现:** 
    *   MoDA 集成于指令调优阶段（instruction tuning），在预训练视觉编码器和适配器层对齐视觉特征后，引入一个基于 Transformer 的跨注意力机制（cross-attention）。
    *   跨注意力机制以视觉特征为查询（query），语言指令嵌入为键（key）和值（value），生成一个调制掩码（modulation mask），该掩码通过逐元素乘法（Hadamard product）作用于视觉嵌入的每个维度，重新加权特征以强调语义相关维度。
    *   调制后的视觉特征作为前缀 token 传递给大语言模型（LLM），用于自回归语言生成。
*   **训练流程:** 遵循标准两阶段训练协议（如 LLaVA），第一阶段仅训练原始适配器以对齐视觉-语言特征，第二阶段引入 MoDA 并联合微调 MoDA 和 LLM，使用自回归交叉熵损失作为目标。
*   **优势:** 不需要额外监督或大规模重新训练，与现有 MLLM 架构无缝集成，计算开销可控（尤其在浅层调制配置下）。

## Experiment

*   **有效性:** MoDA 在多个多模态基准数据集上显著提升了基线模型（如 LLaVA-1.5 和 LLaVA-MoRE）的性能，尤其在细粒度视觉理解任务 MMVP 上，LLaVA-1.5 准确率从 24.0% 提升至 36.0%（+12.0%），LLaVA-MoRE SigLIP-S2 配置从 39.3% 提升至 42.7%（+3.4%）；在 POPE（幻觉评估）和 ScienceQA（科学问答）等任务上也展现出减少幻觉和提升推理能力的优势。
*   **实验设置合理性:** 实验覆盖了不同语言骨干（Vicuna-7B、LLaMA 3.1-8B）和视觉编码器（CLIP、SigLIP-S2），并通过消融研究验证了跨注意力机制、无额外损失函数、浅层调制等设计的优越性；基准数据集涵盖视觉问答、推理和幻觉检测等多方面，评估全面。
*   **不足与开销:** 主要开销在于调制计算，尤其在深层调制配置下训练时间从约 20 小时增至 50 小时以上；论文未详细探讨 MoDA 在极端复杂场景（如多对象重叠）下的表现。

## Further Thoughts

MoDA 的调制思想启发我们，是否可以将类似动态调制机制扩展到其他模态（如音频或视频），以提升多模态任务的跨模态对齐能力？此外，MoDA 目前仅在指令调优阶段起作用，是否可以在预训练阶段引入类似机制，进一步优化视觉-语言特征的对齐质量？另一个方向是 MoDA 的调制掩码未实现显式稀疏性，未来是否可以通过改进稀疏性约束（如改进 L1 正则化）实现更强的特征选择能力，从而更精准地引导模型注意力？