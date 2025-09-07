---
title: "Multimodal Feature Fusion Network with Text Difference Enhancement for Remote Sensing Change Detection"
pubDatetime: 2025-09-04T07:39:18+00:00
slug: "2025-09-multimodal-change-detection"
type: "arxiv"
id: "2509.03961"
score: 0.6195182083469967
author: "grok-3-latest"
authors: ["Yijun Zhou", "Yikui Zhai", "Zilu Ying", "Tingfeng Xian", "Wenlve Zhou", "Zhiheng Zhou", "Xiaolin Tian", "Xudong Jia", "Hongsheng Zhang", "C. L. Philip Chen"]
tags: ["Multimodal Learning", "Remote Sensing", "Change Detection", "Feature Fusion", "Vision-Language Model"]
institution: ["Wuyi University", "South China University of Technology", "Macau University of Science and Technology", "California State University, Northridge", "The University of Hong Kong"]
description: "本文提出 MMChange 模型，通过视觉-语言模型结合图像和文本模态，设计 IFR、TDE 和 ITFF 模块深度融合多模态特征，显著提升遥感变化检测的精度和鲁棒性。"
---

> **Summary:** 本文提出 MMChange 模型，通过视觉-语言模型结合图像和文本模态，设计 IFR、TDE 和 ITFF 模块深度融合多模态特征，显著提升遥感变化检测的精度和鲁棒性。 

> **Keywords:** Multimodal Learning, Remote Sensing, Change Detection, Feature Fusion, Vision-Language Model

**Authors:** Yijun Zhou, Yikui Zhai, Zilu Ying, Tingfeng Xian, Wenlve Zhou, Zhiheng Zhou, Xiaolin Tian, Xudong Jia, Hongsheng Zhang, C. L. Philip Chen

**Institution(s):** Wuyi University, South China University of Technology, Macau University of Science and Technology, California State University, Northridge, The University of Hong Kong


## Problem Background

遥感变化检测（Remote Sensing Change Detection, RSCD）在土地利用变化、灾害评估等领域具有重要应用，但传统单模态方法（仅依赖图像数据）在复杂场景下对噪声和光照变化的鲁棒性不足，且难以捕捉深层语义信息，导致特征表示能力和泛化性能受限。
本文旨在通过引入多模态学习，结合图像和文本模态，增强变化检测的精度和鲁棒性。

## Method

*   **核心思想:** 提出 MMChange 模型，基于视觉-语言模型（Vision-Language Model, VLM），通过图像和文本模态的深度融合，提升遥感变化检测的性能。
*   **具体实现:**
    *   **Image Feature Refinement (IFR) 模块:** 针对双时相遥感图像，利用坐标和通道信息整合、残差学习和分组卷积操作，精炼图像特征，突出形状、轮廓和纹理等关键信息，同时抑制环境噪声，为后续多模态融合提供高质量特征。
    *   **Text Difference Enhancement (TDE) 模块:** 利用 VLM（如 TinyLLaVA）生成双时相图像的语义文本描述，通过缩放点积注意力机制（Scaled Dot-Product Attention）增强文本间的差异特征，捕捉细粒度的语义变化，指导模型定位变化区域。
    *   **Image-Text Feature Fusion (ITFF) 模块:** 设计多层次注意力机制，包括通道注意力（关注关键特征通道）、空间注意力（聚焦重要区域）和像素注意力（精炼像素级细节），深度融合图像和文本特征，充分利用两种模态的互补信息。
*   **流程:** 图像通过 ResNet50 编码器提取特征并由 IFR 精炼；文本描述由 VLM 生成并通过 TDE 增强差异；ITFF 融合多模态特征后输入解码器生成变化掩码。
*   **关键点:** 不修改 VLM 基础模型，仅通过模块化设计增强多模态协作，同时通过提示设计（如‘图片中的组成部分是什么？’）优化文本生成质量。

## Experiment

*   **有效性:** MMChange 在三个公开数据集（LEVIR-CD, WHU-CD, SYSU-CD）上均显著优于 12 个最先进方法。例如，在 LEVIR-CD 上，IOU 和 F1 分别为 85.06% 和 91.93%，比 ChangeCLIP 提升 0.59% 和 0.35%；在 WHU-CD 上，IOU 和 F1 提升至 90.90% 和 95.23%，比 BiFA 高出 1.96% 和 1.08%；在 SYSU-CD 上，IOU 和 F1 分别为 72.05% 和 83.76%，展现了对复杂场景的适应性。
*   **鲁棒性:** 在噪声和光照变化的变异性实验中，MMChange 表现出较强抗干扰能力，例如在 SYSU-CD 上，IOU 和 F1 比基线分别提升 16.56% 和 12.65%。
*   **实验设置合理性:** 数据集选择涵盖不同分辨率和场景复杂度，评价指标（Precision, Recall, IOU, F1）全面，消融实验验证了各模块贡献，额外测试了不同骨干网络和提示设计的影响，设计较为严谨。
*   **不足:** 模型参数量（42.90M）和计算量（113.52G FLOPs）较高，推理速度（29.15 FPS）低于部分轻量级模型，可能限制其在资源受限环境下的应用。

## Further Thoughts

本文的多模态学习思路为遥感变化检测提供了新视角，未来可探索更多模态（如传感器数据）或更大规模 VLM 模型以提升性能；作者提到的利用 VLM 自动生成标签以减少标注依赖的设想非常有前景，可结合自监督或对比学习进一步降低数据需求；此外，提示工程的优化（如自动化提示生成）和模型轻量化设计（如通过知识蒸馏）也是值得深入研究的方向。