---
title: "Information transmission: Inferring change area from change moment in time series remote sensing images"
pubDatetime: 2025-09-03T08:10:30+00:00
slug: "2025-09-change-inference-tscd"
type: "arxiv"
id: "2509.03112"
score: 0.7420955845842667
author: "grok-3-latest"
authors: ["Jialu Li", "Chen Wu", "Meiqi Hu"]
tags: ["Time Series Analysis", "Change Detection", "Remote Sensing", "Feature Enhancement", "Inference Mechanism"]
institution: ["Wuhan University", "Sun Yat-sen University"]
description: "本文提出 CAIM-Net，通过从变化时刻推断变化区域，实现时间序列遥感图像变化检测中变化区域与变化时刻的协同优化，显著提升准确性和一致性。"
---

> **Summary:** 本文提出 CAIM-Net，通过从变化时刻推断变化区域，实现时间序列遥感图像变化检测中变化区域与变化时刻的协同优化，显著提升准确性和一致性。 

> **Keywords:** Time Series Analysis, Change Detection, Remote Sensing, Feature Enhancement, Inference Mechanism

**Authors:** Jialu Li, Chen Wu, Meiqi Hu

**Institution(s):** Wuhan University, Sun Yat-sen University


## Problem Background

时间序列遥感图像变化检测（Time Series Change Detection, TSCD）是生态系统动态监测的关键任务，旨在同时识别变化区域（change area）和变化时刻（change moment）。
现有深学习方法常将两者作为独立任务处理，导致结果不一致；此外，中低分辨率图像常因物体边界模糊而使变化边界不清晰。
本文旨在解决变化区域与变化时刻结果不匹配的问题，并提升模糊边界场景下的检测精度。

## Method

*   **核心思想:** 提出 CAIM-Net（Change Area Inference from Moment Network），利用变化时刻与变化区域的内在关系，从变化时刻推断变化区域，实现两者的协同检测。
*   **具体步骤:**
    *   **差异提取与增强（Difference Extraction and Enhancement）:** 设计轻量级编码器，通过批次维度堆叠和普通卷积提取空间特征，计算相邻时间步特征的绝对差异；引入边界增强卷积（Boundary Enhancement Convolution），通过中心像素与周围像素的差值计算增强变化边界区分度，解决模糊边界问题。
    *   **粗略变化时刻提取（Coarse Change Moment Extraction）:** 基于增强后的差异特征，设计时空相关模块（结合 Transformer 编码器和 LSTM）捕捉空间和时间相关性；采用两种方法初步估计变化时刻：一是通过相邻图像间的变化与无变化特征逐步推断，二是将问题转化为多类语义分割任务，增强结果鲁棒性。
    *   **精细变化时刻提取与变化区域推断（Fine Change Moment Extraction and Change Area Inference）:** 应用多尺度时间类激活映射（Temporal Class Activation Mapping, CAM），通过特征图与分类权重的组合生成热力图，精化粗略变化时刻；基于‘有变化时刻的像素必然发生变化’的原则，从精细变化时刻直接推断变化区域；使用焦点加权交叉熵损失（Focal Weighted Cross-Entropy Loss）优化结果，缓解样本不平衡问题。
*   **关键创新:** 不依赖独立任务分支，而是通过推断机制实现变化区域与时刻的协同优化，同时在特征提取阶段针对模糊边界问题设计定制化操作。

## Experiment

*   **有效性:** 在 DynamicEarthNet 和 SpaceNet7 两个全球尺度数据集上，CAIM-Net 在变化区域检测和变化时刻识别任务中均优于现有最优方法（State-of-the-Art, SOTA），Kappa 系数分别提升 1.12% 和 0.36%（DynamicEarthNet）以及 2.16% 和 0.97%（SpaceNet7）。
*   **合理性:** 实验设置全面，数据集涵盖多种地表变化场景（如土地覆盖变化和建筑物变化），训练、验证和测试数据比例为 8:1:1；消融实验验证了各模块（如边界增强卷积、时空相关模块、多尺度 CAM）的贡献，证明方法设计的合理性。
*   **局限性与分析:** 在 SpaceNet7 数据集上，变化时刻识别的 F1 分数略低于 Multi-RLD-Net，可能是由于数据极度不平衡（变化样本仅占 1%），但 Kappa 系数作为更可靠指标仍显示 CAIM-Net 的优势。
*   **效率:** CAIM-Net 计算复杂度和存储需求较低，FLOPs 和参数量仅为 MC²ABNet 的 5% 和 2%，推理时间为 19.92 秒，显著优于 RLD-Net 和 Multi-RLD-Net（约 160 秒），展现出实际应用潜力。

## Further Thoughts

从变化时刻推断变化区域的思路非常具有启发性，这种基于内在关系的推断机制可以扩展到其他时间序列任务中，如视频事件检测或医疗数据异常定位；此外，边界增强卷积的定制化设计提示我们可以在特征提取阶段针对具体问题（如边界模糊）优化模型；多尺度 CAM 的应用则启发在处理时间序列数据时，结合多尺度信息可能显著提升精度；未来是否可以通过自监督或弱监督学习减少对昂贵标注数据的依赖，尤其是在遥感领域？