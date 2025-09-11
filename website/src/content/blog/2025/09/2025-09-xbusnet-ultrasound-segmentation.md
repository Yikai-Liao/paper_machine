---
title: "XBusNet: Text-Guided Breast Ultrasound Segmentation via Multimodal Vision-Language Learning"
pubDatetime: 2025-09-08T20:45:55+00:00
slug: "2025-09-xbusnet-ultrasound-segmentation"
type: "arxiv"
id: "2509.07213"
score: 0.43248685273390436
author: "grok-3-latest"
authors: ["Raja Mallina", "Bryar Shareef"]
tags: ["Medical Imaging", "Segmentation", "Vision-Language Model", "Prompt Learning", "Ultrasound Analysis"]
institution: ["University of Nevada, Las Vegas"]
description: "本文提出 XBusNet，一种双分支、双提示的多模态视觉-语言模型，通过结合全局语义和局部细节显著提升乳腺超声图像分割精度，尤其在小病灶和低对比度场景下。"
---

> **Summary:** 本文提出 XBusNet，一种双分支、双提示的多模态视觉-语言模型，通过结合全局语义和局部细节显著提升乳腺超声图像分割精度，尤其在小病灶和低对比度场景下。 

> **Keywords:** Medical Imaging, Segmentation, Vision-Language Model, Prompt Learning, Ultrasound Analysis

**Authors:** Raja Mallina, Bryar Shareef

**Institution(s):** University of Nevada, Las Vegas


## Problem Background

乳腺超声（BUS）分割在乳腺癌早期诊断中至关重要，但由于小病灶、低对比度、模糊边界及斑点噪声的影响，精准分割仍面临挑战。
传统深度学习方法在复杂场景下表现有限，且临床上下文信息（如 BI-RADS 描述）未被充分利用，导致模型输出缺乏解释性和鲁棒性。
本文旨在通过结合图像特征与临床文本提示，提升 BUS 分割精度，尤其针对小病灶和低对比度病灶。

## Method

*   **核心思想:** 提出 XBusNet，一个双分支、双提示的多模态视觉-语言模型，通过结合图像特征与临床文本提示，提升乳腺超声分割精度。
*   **双分支架构:** 
    *   **全局分支（Global Feature Extractor, GFE）:** 基于 CLIP Vision Transformer，结合全局特征上下文提示（GFCP），编码病灶大小和位置等整体语义信息，帮助模型理解图像的场景级上下文。
    *   **局部分支（Local Feature Extractor, LFE）:** 基于 ResNet50 编码器和 U-Net 风格解码器，结合局部特征提示（LFP），关注形状、边界和 BI-RADS 术语等细节信息，强调边界精度。
*   **语义特征调整（Semantic Feature Adjustment, SFA）机制:** 通过文本提示嵌入计算缩放和偏移参数，动态调制全局和局部分支的特征图，使模型对临床属性更敏感，提升分割与临床描述的一致性。
*   **提示构建与融合:** 提示从结构化元数据（如数据集 CSV 文件中的病灶属性）自动生成，无需手动标注；全局和局部特征图经调制后通过融合头（Fusion Head）整合，最终输出分割掩码。
*   **训练细节:** 使用五折交叉验证，冻结 CLIP 编码器，微调 ResNet50 编码器，采用 AdamW 优化器和余弦学习率调度，确保训练稳定性。

## Experiment

*   **有效性:** 在 Breast Lesions USG (BLU) 数据集上，XBusNet 取得了平均 Dice 0.8765 和 IoU 0.8149，显著优于六个基线模型（如 U-Net 的 Dice 0.604，CLIP-TNseg 的 Dice 0.839）。
*   **小病灶提升:** 尤其在小病灶（0-110 像素）上，Dice 从基线最佳的 0.7689 提升至 0.8507，表明模型对困难案例的鲁棒性显著增强。
*   **消融实验:** 验证了全局分支、局部分支和 SFA 机制的互补作用，去掉任一组件均导致性能下降（如无 GFE 时 Dice 降至 0.8453）。
*   **实验设置合理性:** 采用五折交叉验证、大小分层分析和 Grad-CAM 可视化，评估全面；但数据集规模较小（252 张图像），可能限制泛化性测试。
*   **定性结果:** XBusNet 在高对比度病灶上生成连续轮廓，在低对比度和小病灶上减少漏检和伪影，优于基线模型。

## Further Thoughts

XBusNet 的多模态提示驱动框架为医疗图像分割提供了新思路，自动从临床元数据生成文本提示的设计增强了模型的实用性和解释性，这种方法可扩展到其他医疗成像任务（如 CT、MRI）；双分支架构平衡全局语义与局部细节的策略，以及通过文本嵌入动态调制特征图的 SFA 机制，也为自然图像分割或跨模态任务提供了启发，未来可探索更复杂的特征调制方式或跨数据集的泛化能力。