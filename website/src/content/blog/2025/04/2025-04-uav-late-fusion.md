---
title: "Is Intermediate Fusion All You Need for UAV-based Collaborative Perception?"
pubDatetime: 2025-05-01T15:50:16Z
slug: "2025-04-uav-late-fusion"
type: "arxiv"
id: "2504.21774"
score: 0.6949981846367574
author: "grok-3-latest"
authors: ["Jiuwu Hao", "Liguo Sun", "Yuting Wan", "Yueyang Wu", "Ti Xiang", "Haolin Song", "Pin Lv"]
tags: ["Collaborative Perception", "UAV Systems", "Feature Fusion", "Communication Efficiency", "Uncertainty Estimation"]
institution: ["School of Artificial Intelligence, University of Chinese Academy of Sciences", "Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences"]
description: "本文提出晚期中间融合（LIF）框架，通过传输紧凑检测结果并在特征层面融合，显著降低无人机协作感知的通信开销，同时实现最先进的检测性能。"
---

> **Summary:** 本文提出晚期中间融合（LIF）框架，通过传输紧凑检测结果并在特征层面融合，显著降低无人机协作感知的通信开销，同时实现最先进的检测性能。 

> **Keywords:** Collaborative Perception, UAV Systems, Feature Fusion, Communication Efficiency, Uncertainty Estimation
> **Recommendation Score:** 0.6949981846367574

**Authors:** Jiuwu Hao, Liguo Sun, Yuting Wan, Yueyang Wu, Ti Xiang, Haolin Song, Pin Lv
**Institution(s):** School of Artificial Intelligence, University of Chinese Academy of Sciences, Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences

## Problem Background

无人机（UAV）平台的协作感知通过多智能体通信增强环境感知能力，是智能交通系统的重要方向。然而，现有方法多采用中间融合策略，忽略了无人机通信带宽和计算资源的限制，以及高空视角下检测结果的高准确性，导致通信开销过高。本文旨在解决通信效率低下的问题，探索是否中间融合是无人机协作感知的唯一或最佳选择。

## Method

* **核心思想**：提出晚期中间融合（Late-Intermediate Fusion, LIF）框架，通过传输紧凑的检测结果而非高维特征，并在特征表示层面进行融合，以减少通信冗余并保持感知性能。
* **具体实现**：
  - **视觉引导的位置嵌入（VPE）**：利用其他智能体的2D检测结果，通过坐标变换生成位置嵌入，引导自我智能体在鸟瞰图（BEV）特征图上关注关键区域，提升空间注意力分配。
  - **基于边界框的虚拟增强特征（BoBEV）**：将接收到的3D边界框信息（包括几何特征和置信度）整合到自我智能体的BEV特征中，通过可学习方式增强特征表示，弥补检测结果传输导致的信息损失。
  - **不确定性驱动的通信机制**：基于不确定性评估计算需求图，筛选高质量、可信的检测结果进行共享，避免传输低质量信息，进一步优化带宽使用。
* **关键优势**：不依赖高维特征传输，支持异构模型协作，且通过特征层面的融合避免了传统晚期融合的性能下降。

## Experiment

* **有效性**：在UAV3D数据集上，LIF框架取得了最先进的性能，mAP达到72.1%，NDS达到61.4%，超越了多种中间融合方法（如When2com, V2VNet, DiscoNet）以及早期融合方法。
* **通信效率**：LIF仅传输检测结果，通信开销极低，在带宽受限场景下性能下降较小，与晚期融合接近，但感知精度显著更高。
* **实验设置**：实验对比了早期、中间、晚期融合策略，并通过消融研究验证了各模块（VPE, BoBEV, 特征图分辨率调整）的贡献，显示BoBEV对性能提升贡献最大，增大特征图分辨率对小目标检测至关重要。
* **局限性**：实验基于模拟数据集（UAV3D），缺乏真实世界场景验证，可能影响泛化性。

## Further Thoughts

LIF框架通过传输检测结果实现高效协作，启发我们在其他多智能体场景（如自动驾驶）中探索基于结果而非特征的协作方式；不确定性驱动的通信机制可扩展至分布式学习中优化数据共享；此外，特征融合阶段的灵活性提示我们可以在不同任务中尝试更早或更晚的融合策略，以适应多样化需求。