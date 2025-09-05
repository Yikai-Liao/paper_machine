---
title: "MedLiteNet: Lightweight Hybrid Medical Image Segmentation Model"
pubDatetime: 2025-09-03T05:59:13+00:00
slug: "2025-09-medlitenet-segmentation"
type: "arxiv"
id: "2509.03041"
score: 0.4583536709118831
author: "grok-3-latest"
authors: ["Pengyang Yu", "Haoquan Wang", "Gerard Marks", "Tahar Kechadi", "Laurence T. Yang", "Sahraoui Dhelim", "Nyothiri Aung"]
tags: ["Medical Segmentation", "Lightweight Model", "CNN-Transformer", "Boundary Detection", "Multi-Scale Context"]
institution: ["未明确列出具体机构，推测为学术研究团队"]
description: "MedLiteNet 提出了一种轻量级 CNN-Transformer 混合模型，通过局部-全局-边界三重耦合设计，在皮肤病变分割中以仅 3.2M 参数实现 Dice 0.91 的高精度和 1 毫秒的快速推理，显著平衡了效率与性能。"
---

> **Summary:** MedLiteNet 提出了一种轻量级 CNN-Transformer 混合模型，通过局部-全局-边界三重耦合设计，在皮肤病变分割中以仅 3.2M 参数实现 Dice 0.91 的高精度和 1 毫秒的快速推理，显著平衡了效率与性能。 

> **Keywords:** Medical Segmentation, Lightweight Model, CNN-Transformer, Boundary Detection, Multi-Scale Context

**Authors:** Pengyang Yu, Haoquan Wang, Gerard Marks, Tahar Kechadi, Laurence T. Yang, Sahraoui Dhelim, Nyothiri Aung

**Institution(s):** 未明确列出具体机构，推测为学术研究团队


## Problem Background

皮肤病变分割对皮肤癌的计算机辅助诊断至关重要，但面临病变颜色纹理多样性、低对比度、形状大小差异大及边界模糊等挑战。
传统方法依赖手动参数调整，难以适应复杂形态；现有深度学习方法如 CNN 受限于局部感受野，Vision Transformer 虽能捕捉全局上下文，但参数量大、计算复杂且在小样本医疗数据上易欠拟合；此外，现有混合架构多依赖重量级骨干网络，不适合资源受限的移动或实时临床部署。
因此，关键问题是如何在保持高精度分割的同时，显著降低模型复杂度和推理延迟。

## Method

* **整体架构**：MedLiteNet 是一种轻量级 CNN-Transformer 混合模型，采用编码器-解码器结构，旨在通过局部-全局-边界三重耦合设计实现高效高精度的皮肤病变分割。
* **轻量级卷积编码器（MBConv）**：基于 MobileNetV2 的倒残差块（Mobile Inverted Bottleneck），通过深度可分离卷积和倒残差设计减少参数量和计算成本，编码器分多个阶段提取多尺度局部纹理特征，总参数量仅约 1.8M。
* **Transformer 全局编码与局部-全局融合**：在编码器瓶颈层引入 Transformer 模块，通过多头自注意力机制（MHSA）捕捉长距离依赖，将全局特征与卷积特征通过像素级融合（沿通道轴拼接后用 1×1 卷积融合），兼顾局部细节与全局上下文。
* **边界感知注意力模块（BAA）**：专门增强病变边界检测，通过提取边界响应图并生成注意力掩码，增强边界区域特征表示，在编码器末端和解码器末端嵌入该模块以提升边界对齐精度。
* **ASPP 多尺度上下文模块**：采用空洞空间金字塔池化（ASPP），通过不同膨胀率的并行卷积提取多尺度特征，适应不同大小病变，同时在低分辨率瓶颈层操作以控制计算开销。
* **解码器与损失函数**：解码器采用类似 U-Net 的对称结构，通过跳跃连接融合多级特征，并在解码阶段嵌入边界注意力模块优化边界质量；损失函数结合 Dice 损失和二元交叉熵（BCE）损失，平衡区域重叠与像素级精度。

## Experiment

* **数据集与设置**：在 ISIC 2018 皮肤病变分割数据集（约 2594 张训练图像）上进行实验，采用官方训练/测试划分，评估指标为 Dice 系数和 IoU；数据增强包括几何变换、空间变换和像素强度调整，训练配置包括混合精度训练、梯度裁剪和余弦退火学习率调度，确保训练稳定性和收敛性。
* **结果**：单模型 MedLiteNet 取得 Dice 0.897 ± 0.010 和 IoU 0.821 ± 0.015，参数量仅 3.3M；三种变体集成后精度提升至 Dice 0.904 ± 0.012 和 IoU 0.830 ± 0.018，总参数量低于 10M，比 Vision Transformer 骨干小 90% 以上；推理速度为 256×256 图像单帧 1 毫秒（RTX A6000）。
* **定性分析**：在不规则边界、低对比度区域和多尺度病变上表现优异，但对模糊边界和外部干扰（如毛囊）仍有小幅误分割。
* **比较**：与 TransUNet（105M 参数，Dice 0.885）、FAT-Net（28M 参数，Dice 0.890）和 BACANet（7.56M 参数，Dice 0.921）相比，MedLiteNet 在精度上接近或略优，同时在参数量和推理速度上具有显著优势，体现了轻量化与高性能的平衡。

## Further Thoughts

MedLiteNet 的轻量化设计（如 MBConv 和瓶颈层 Transformer 融合）启发我们可以在其他资源受限场景中探索类似混合架构；边界感知注意力机制（BAA）提示在需要精确定位的任务中可结合传统边缘检测与深度学习特征；ASPP 在低分辨率阶段操作以控制成本的策略，启发高效模型设计时优先在低分辨率层引入复杂模块；此外，对低对比度和外部干扰的处理不足提示未来可结合预处理或多任务学习（如联合分类与分割）提升鲁棒性。