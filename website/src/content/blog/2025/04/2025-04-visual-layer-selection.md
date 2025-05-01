---
title: "Rethinking Visual Layer Selection in Multimodal LLMs"
pubDatetime: 2025-05-01T15:51:21Z
slug: "2025-04-visual-layer-selection"
type: "arxiv"
id: "2504.21447"
score: 0.8727925032865116
author: "grok-3-latest"
authors: ["Haoran Chen", "Junyan Lin", "Xinhao Chen", "Yue Fan", "Xin Jin", "Hui Su", "Jianfeng Dong", "Jinlan Fu", "Xiaoyu Shen"]
tags: ["Multimodal LLM", "Visual Encoder", "Feature Fusion", "Layer Selection", "Task Performance"]
institution: ["Ningbo Key Laboratory of Spatial Intelligence and Digital Derivative, Institute of Digital Twin, EIT", "Zhejiang Gongshang University", "Genmo.ai", "Meituan Inc.", "National University of Singapore"]
description: "本文通过层级表示相似性分析系统研究 CLIP-ViT 视觉层级特征差异，并提出轻量级融合策略，显著提升多模态大语言模型在多样化任务上的性能。"
---

> **Summary:** 本文通过层级表示相似性分析系统研究 CLIP-ViT 视觉层级特征差异，并提出轻量级融合策略，显著提升多模态大语言模型在多样化任务上的性能。 

> **Keywords:** Multimodal LLM, Visual Encoder, Feature Fusion, Layer Selection, Task Performance
> **Recommendation Score:** 0.8727925032865116

**Authors:** Haoran Chen, Junyan Lin, Xinhao Chen, Yue Fan, Xin Jin, Hui Su, Jianfeng Dong, Jinlan Fu, Xiaoyu Shen
**Institution(s):** Ningbo Key Laboratory of Spatial Intelligence and Digital Derivative, Institute of Digital Twin, EIT, Zhejiang Gongshang University, Genmo.ai, Meituan Inc., National University of Singapore

## Problem Background

多模态大语言模型（MLLMs）在多种任务上表现出色，通常使用 CLIP-ViT 作为视觉编码器，但对视觉特征层级的选择多基于经验性启发，而非系统性分析。
论文指出，不同层级的视觉特征在 CLIP-ViT 中捕捉不同类型的信息——浅层关注细粒度视觉细节，深层更贴近文本语义对齐，而当前模型普遍偏向深层特征（如倒数第二层），可能忽略了浅层和中层特征的潜力。
因此，核心问题是：深层特征是否总是最优？如何系统性地选择或融合不同层级特征以提升 MLLMs 在多样化任务上的表现？

## Method

*   **层级表示相似性（Layer-wise Representation Similarity, LRS）分析：** 提出了一种方法，通过计算 CLIP-ViT 各层隐藏状态的余弦相似性矩阵，量化层与层之间的行为模式，将 24 层分为三组：浅层（1-12 层，捕捉低级视觉特征如边缘）、中层（13-20 层，过渡特征）、深层（21-24 层，高语义抽象与文本对齐）。
*   **层级特征性能评估：** 在 LLaVA 风格的模型架构上，逐层输入 CLIP-ViT 的隐藏状态到连接器（Connector），通过两阶段训练（预训练与指令微调）评估各层在多模态任务上的表现，探索浅层和中层是否在特定任务上优于深层。
*   **轻量级特征融合策略：** 基于 LRS 分析结果，设计了一种简单融合方法，从浅层、中层、深层各选取代表性层（如第 3、18、23 层），通过特征维度上的拼接（Concatenation）整合多层特征，再通过单层线性层映射到语言模型的 token 空间，旨在以最小计算开销结合各层优势。
*   **实现细节：** 使用 CLIP ViT-L/14 作为视觉编码器，语言模型包括 1.4B MobileLLaMA 等，训练采用 AdamW 优化器和余弦退火学习率调度，确保实验可控性和效率。

## Experiment

*   **实验设置全面性：** 实验基于 LLaVA 风格模型，模型规模从 1.4B 到 7B 参数，训练数据规模从 665K 到 1M 样本，评估覆盖 10 个数据集和 4 类任务（通用任务、OCR 任务、视觉中心任务、幻觉任务），确保结果的普适性和鲁棒性。
*   **层级性能差异：** 结果表明，深层（尤其是倒数第二层，即第 23 层）在 OCR 任务上表现最佳（如 OCRBench 得分 233，高于中层的 200）；浅层和中层在计数、定位等视觉推理任务上显著优于深层（如第 18 层在 CVBench 上得分 47.29，高于第 23 层的 44.26，差距约 3%）。
*   **融合策略效果：** 提出的轻量级融合策略（结合第 3、18、23 层）在 9/10 数据集上优于单一层选择和现有融合方法（如 DenseConnector 和 MMFuser），例如在 MMBench 上从基线 35.31 提升到 49.22，增幅约 39%。
*   **局限与开销：** 融合策略计算开销低（仅增加单层线性层），但对浅层特征在 OCR 任务上的负面影响需进一步优化；实验主要基于 CLIP-ViT，未广泛验证其他视觉编码器的适用性。

## Further Thoughts

论文揭示了视觉层级特征的互补性对任务表现的影响，这启发我思考是否可以设计一种动态层级选择机制，根据任务类型（如 OCR 或推理）自适应地调整层级权重或选择特定层级特征，而非静态融合；此外，是否可以将类似分析扩展到其他视觉编码器（如 DINOv2 或 SigLIP），以探索层级特征差异的通用性，并进一步提升多模态模型在复杂场景下的泛化能力？