---
title: "FASL-Seg: Anatomy and Tool Segmentation of Surgical Scenes"
pubDatetime: 2025-09-07T17:59:09+00:00
slug: "2025-09-fasl-seg-surgical-segmentation"
type: "arxiv"
id: "2509.06159"
score: 0.6412900029913484
author: "grok-3-latest"
authors: ["Muraam Abdel-Ghani", "Mahmoud Ali", "Mohamed Ali", "Fatmaelzahraa Ahmed", "Mohamed Arsalan", "Abdulaziz Al-Ali", "Shidin Balakrishnan"]
tags: ["Semantic Segmentation", "Multi-Scale Features", "Surgical Scene", "Transformer Backbone", "Feature Processing"]
institution: ["Hamad Medical Corporation", "Qatar University"]
description: "本文提出 FASL-Seg 模型，通过双流多尺度特征处理架构，在手术场景分割中显著提升工具和解剖结构的精度，超越现有 SOTA 模型。"
---

> **Summary:** 本文提出 FASL-Seg 模型，通过双流多尺度特征处理架构，在手术场景分割中显著提升工具和解剖结构的精度，超越现有 SOTA 模型。 

> **Keywords:** Semantic Segmentation, Multi-Scale Features, Surgical Scene, Transformer Backbone, Feature Processing

**Authors:** Muraam Abdel-Ghani, Mahmoud Ali, Mohamed Ali, Fatmaelzahraa Ahmed, Mohamed Arsalan, Abdulaziz Al-Ali, Shidin Balakrishnan

**Institution(s):** Hamad Medical Corporation, Qatar University


## Problem Background

随着机器人辅助微创手术的普及，精准分割手术场景中的工具和解剖结构对于手术培训和技能评估至关重要。
然而，现有研究多集中于手术工具分割，忽视了解剖结构；此外，当前最先进模型在平衡高层次上下文特征和低层次边缘特征时表现不佳，导致多尺度对象分割精度不足。

## Method

*   **核心思想:** 提出 FASL-Seg（Feature-Adaptive Spatial Localization）模型，通过多尺度特征自适应处理，同时捕捉低层次边缘细节和高层次上下文信息，实现对手术工具和解剖结构的精准分割。
*   **架构基础:** 基于 SegFormer 变换器骨干网络，利用其分层编码器块提取不同分辨率的特征图，并通过注意力机制和重叠补丁嵌入增强语义理解能力，同时对输入分辨率变化具有鲁棒性。
*   **双流处理机制:** 
    *   **Low-Level Feature Projection (LLFP) 流:** 针对早期编码器层输出的高分辨率特征图，专注于边缘细节和小对象（如工具尖端）。通过点卷积（Point-wise Convolution）层、批归一化和 Leaky ReLU 激活的组合（ConvBlock）精炼特征表示，保持空间维度；随后应用多头自注意力机制（MHSA）捕捉局部和全局依赖关系，减少噪声；最后通过多次插值上采样（Up Chain）调整特征图尺寸，确保细节不被过度平滑。
    *   **High-Level Feature Projection (HLFP) 流:** 针对后期编码器层输出的低分辨率特征图，专注于上下文信息和大对象（如解剖结构）。通过一系列 ConvBlock 压缩通道维度，保留语义信息；不使用注意力机制以避免丢失关键上下文特征；通过插值上采样调整尺寸以匹配最终输出。
*   **特征融合与解码:** 将 LLFP 和 HLFP 流处理后的多尺度特征通过通道级联融合，形成增强的特征图；随后通过浅层解码器（包含多个 ConvBlock 和双线性插值）进行加权特征选择和通道压缩，最终通过拉普拉斯卷积层生成分割结果。
*   **关键创新:** 根据特征图分辨率自适应选择处理流，确保细节和上下文信息的双重保留，同时避免过度复杂化模型结构。

## Experiment

*   **有效性:** 在 EndoVis18 部件与解剖分割任务中，FASL-Seg 的 mIoU 达到 72.71%，比 SOTA（MedT）提高 5%，Dice 系数提高 9%；在 EndoVis18 工具分割任务中 mIoU 达 85.61%，在 EndoVis17 中达 72.78%，均优于 SOTA 整体表现。
*   **一致性与优越性:** FASL-Seg 在各类别上的表现均衡，平均每类 mIoU 和 Dice 指标优于其他模型，尤其在工具和解剖结构的分割中展现出一致性；相比其他模型（如 SegFormer），其假阳性率（FPR）也显著降低。
*   **实验设置合理性:** 数据集划分明确（EndoVis18 和 EndoVis17），训练和测试集比例合理；使用 A6000 GPU 训练，超参数一致（如学习率 1E-5，批量大小 4）；采用 Tversky 和交叉熵损失组合，平衡假阳性和假阴性；数据增强仅限于随机裁剪和翻转，贴近真实手术场景。
*   **局限性:** 模型在某些类别（如覆盖的肾脏）表现较差，可能需更多增强技术；推理速度为 2.14 帧/秒，模型复杂度较高（81.99M 参数，223.42 GFLOPs），暂不适合实时应用。

## Further Thoughts

FASL-Seg 的双流多尺度特征处理机制（LLFP 和 HLFP）为处理细节与上下文的平衡提供新思路，可推广至其他视觉任务（如自动驾驶物体检测）；其选择性应用注意力机制（仅在低层次特征中使用 MHSA）的策略启发我们根据特征层级动态调整模型组件；此外，探索轻量化骨干网络和替代注意力机制的未来方向提示我们可以通过模型剪枝或知识蒸馏优化计算效率，兼顾性能与实时性。