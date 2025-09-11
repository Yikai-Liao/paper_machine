---
title: "Reconstruction Alignment Improves Unified Multimodal Models"
pubDatetime: 2025-09-08T23:59:32+00:00
slug: "2025-09-reconstruction-alignment-umm"
type: "arxiv"
id: "2509.07295"
score: 0.5339395436735622
author: "grok-3-latest"
authors: ["Ji Xie", "Trevor Darrell", "Luke Zettlemoyer", "XuDong Wang"]
tags: ["LLM", "Multimodal Models", "Semantic Alignment", "Image Reconstruction", "Post-Training"]
institution: ["UC Berkeley", "University of Washington"]
description: "本文提出重建对齐（RecA）方法，通过自监督图像重建利用视觉理解编码器的密集嵌入显著提升统一多模态模型的图像生成和编辑性能。"
---

> **Summary:** 本文提出重建对齐（RecA）方法，通过自监督图像重建利用视觉理解编码器的密集嵌入显著提升统一多模态模型的图像生成和编辑性能。 

> **Keywords:** LLM, Multimodal Models, Semantic Alignment, Image Reconstruction, Post-Training

**Authors:** Ji Xie, Trevor Darrell, Luke Zettlemoyer, XuDong Wang

**Institution(s):** UC Berkeley, University of Washington


## Problem Background

统一多模态模型（UMMs）旨在单一架构内实现视觉理解和生成，但传统训练依赖图像-文本对，文本描述往往稀疏，难以捕捉细粒度视觉细节（如空间布局、颜色属性），导致理解与生成之间的对齐问题，影响图像生成和编辑的保真度。

## Method

*   **核心思想:** 提出重建对齐（Reconstruction Alignment, RecA），一种资源高效的后训练方法，利用模型自身的视觉理解编码器（如 CLIP 或 DINO）提取的语义嵌入作为‘密集视觉提示’，通过自监督重建损失优化模型以重建输入图像，从而增强理解与生成的对齐。
*   **具体实现:** 
    *   从视觉理解编码器中提取输入图像的语义嵌入，将其与模板文本嵌入（如‘详细描述图像’）融合，输入到 UMM 中。
    *   使用自监督重建损失（如扩散损失或交叉熵损失）优化模型，使其生成的图像或图像潜在表示尽可能接近原始输入图像。
    *   训练时可结合图像到文本（I2T）目标以保持理解能力，而将传统文本到图像（T2I）监督的权重设为零，专注于重建对齐。
    *   推理时无需额外输入，保持模型原始可用性，与标准 UMM 操作一致。
*   **与其他方法的区别:** RecA 与分类器自由引导（CFG）正交，前者基于语义嵌入重建，后者依赖条件与无条件预测的对比；相比监督微调（SFT），RecA 不依赖标注数据，通过自监督方式提供更丰富的监督信号。
*   **适用性:** RecA 适用于多种 UMM 架构，包括自回归（AR）、掩码自回归（MAR）和自回归+扩散（AR+Diff）模型，展现出广泛的通用性。

## Experiment

*   **有效性:** RecA 显著提升了图像生成和编辑性能，以 Harmon-1.5B 模型为例，GenEval 得分从 0.73 提升至 0.86（无 GPT-4o 数据）及 0.90（有 GPT-4o 数据），DPGBench 从 80.93 提升至 87.21 及 88.15，超越更大规模的开源模型和 GPT-4o；图像编辑基准 ImgEdit 从 3.38 提升至 3.75，GEdit 从 6.94 提升至 7.25。
*   **泛化性:** RecA 在多种 UMM 架构（如 Show-o、Harmon、OpenUni、BAGEL）上均取得一致改进，尤其在 Harmon 和 OpenUni 上提升显著，显示出方法的高度通用性。
*   **资源效率:** 后训练仅需 27 个 A100 GPU 小时，相比依赖大规模数据蒸馏或强化学习的方法，计算开销极低。
*   **实验设置合理性:** 实验覆盖多种模型架构和数据集（如 MidjourneyV6、LLaVA Mix-665K），并排除 GPT-4o 图像蒸馏数据以避免评估偏差（GenEval 模板泄露），设置全面；但在计数任务和某些架构（如 Show-o）上改进有限，可能是视觉编码器容量或模型固有限制所致。
*   **局限性:** 对推理能力（如 WISE 基准）的提升较小，提示未来可能需结合更强语言模型或专门数据集。

## Further Thoughts

RecA 利用视觉理解编码器嵌入作为密集监督信号的思路令人启发，未来可探索不同类型编码器（如更强大的多模态编码器）对效果的影响；此外，是否可将 RecA 扩展至其他模态（如音频-视觉对齐），或嵌入预训练阶段作为混合目标以减少后训练开销；针对计数任务改进有限的问题，可考虑引入专门数据集或强化学习策略来增强模型能力。