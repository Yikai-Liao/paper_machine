---
title: "Is Intermediate Fusion All You Need for UAV-based Collaborative Perception?"
pubDatetime: 2025-04-30T16:22:14+00:00
slug: "2025-04-uav-lif-collaboration"
type: "arxiv"
id: "2504.21774"
score: 0.2953562873894784
author: "grok-3-latest"
authors: ["Jiuwu Hao", "Liguo Sun", "Yuting Wan", "Yueyang Wu", "Ti Xiang", "Haolin Song", "Pin Lv"]
tags: ["Collaborative Perception", "UAV Systems", "Fusion Strategy", "Communication Efficiency", "Feature Enhancement"]
institution: ["School of Artificial Intelligence, University of Chinese Academy of Sciences", "Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences"]
description: "本文提出了一种通信高效的多无人机协作感知框架 LIF，通过后期-中间融合策略共享紧凑检测结果并整合特征，显著降低通信开销同时实现高性能3D目标检测。"
---

> **Summary:** 本文提出了一种通信高效的多无人机协作感知框架 LIF，通过后期-中间融合策略共享紧凑检测结果并整合特征，显著降低通信开销同时实现高性能3D目标检测。 

> **Keywords:** Collaborative Perception, UAV Systems, Fusion Strategy, Communication Efficiency, Feature Enhancement

**Authors:** Jiuwu Hao, Liguo Sun, Yuting Wan, Yueyang Wu, Ti Xiang, Haolin Song, Pin Lv

**Institution(s):** School of Artificial Intelligence, University of Chinese Academy of Sciences, Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences


## Problem Background

无人机（UAV）平台的协作感知通过多智能体间通信增强环境感知能力，是智能交通系统的重要方向。然而，现有方法多源自自动驾驶领域，忽略了无人机平台的通信容量和计算资源限制，以及高空视角下更准确预测的特性，导致通信开销过高。本文旨在探索一种通信高效的协作感知框架，解决带宽受限场景下的多无人机协作问题。

## Method

* **核心思想**：提出后期-中间融合（Late-Intermediate Fusion, LIF）框架，通过共享紧凑的检测结果而非高维神经特征，降低通信开销，并将融合阶段转移到特征表示级别以减少信息丢失。
* **具体实现**：
  - **视觉引导的位置嵌入（VPE）**：利用其他智能体的2D检测结果，通过坐标变换生成位置嵌入，引导自我智能体关注BEV特征图中的关键区域，提升空间注意力分配。
  - **基于框的虚拟增强特征（BoBEV）**：将接收到的3D边界框信息（包括位置、大小、方向和置信度）整合到自我智能体的BEV特征中，形成增强特征图，弥补检测结果信息不足的问题。
  - **不确定性驱动的通信机制**：通过计算不确定性地图和需求地图，筛选高质量、可信的检测结果进行共享，避免传输低质量信息，同时考虑高置信背景区域以减少误报。
* **关键优势**：不依赖高维特征传输，支持异构协作（不同智能体可使用不同模型），通过可学习融合方式提升协作效果。

## Experiment

* **有效性**：在 UAV3D 数据集上，LIF 框架达到 72.1% mAP 和 61.4% NDS，显著优于中间融合方法（如 DiscoNet 的 70.0% mAP）和早期融合方法，证明其在性能上的领先地位。
* **通信效率**：LIF 仅传输检测结果，通信开销极低，与后期融合接近，但在性能上远超后者，尤其在带宽受限场景下表现稳定。
* **实验设置**：实验对比了多种协作策略（早期、中间、后期融合），并通过消融研究验证了各模块贡献（如 BoBEV 提升显著，特征图分辨率增大改善小目标检测）。数据集为模拟环境下的 UAV3D，包含训练、验证和测试集，设置较为全面。
* **局限性**：实验基于模拟数据，作者指出需在真实世界数据集上进一步验证泛化性和鲁棒性。

## Further Thoughts

LIF 框架揭示了在资源受限场景下，紧凑检测结果结合适当融合机制即可实现高效协作的潜力，这可推广至其他领域（如物联网设备协作）。不确定性驱动的通信机制为多智能体系统提供了一种智能信息筛选思路，值得在动态通信优化中进一步探索。此外，支持异构协作的设计为实际多设备协作场景提供了灵活性，启发我们在模型兼容性上做更多尝试。