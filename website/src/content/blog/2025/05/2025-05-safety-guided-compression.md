---
title: "Optimizing Deep Neural Networks using Safety-Guided Self Compression"
pubDatetime: 2025-05-01T06:50:30+00:00
slug: "2025-05-safety-guided-compression"
type: "arxiv"
id: "2505.00350"
score: 0.5038780957627446
author: "grok-3-latest"
authors: ["Mohammad Zbeeb", "Mariam Salman", "Mohammad Bazzi", "Ammar Mohanna"]
tags: ["Deep Learning", "Model Compression", "Quantization", "Generalization", "Safety-Driven"]
institution: ["American University of Beirut"]
description: "本文提出安全驱动的量化框架，通过保留集指导深度神经网络的自压缩，在显著减小模型体积的同时提升性能和泛化能力，为资源受限环境下的部署提供可靠优化策略。"
---

> **Summary:** 本文提出安全驱动的量化框架，通过保留集指导深度神经网络的自压缩，在显著减小模型体积的同时提升性能和泛化能力，为资源受限环境下的部署提供可靠优化策略。 

> **Keywords:** Deep Learning, Model Compression, Quantization, Generalization, Safety-Driven

**Authors:** Mohammad Zbeeb, Mariam Salman, Mohammad Bazzi, Ammar Mohanna

**Institution(s):** American University of Beirut


## Problem Background

深度神经网络在资源受限设备上的部署面临挑战，传统模型压缩方法（如剪枝和量化）虽能减小模型体积和提升推理速度，但往往以牺牲性能和可靠性为代价，尤其是在泛化能力和稳定性方面存在风险。
本文提出了一种安全驱动的量化框架，旨在通过保留关键模型特征，在压缩模型的同时确保性能和可靠性，解决资源受限场景下的部署难题。

## Method

*   **核心思想:** 提出安全驱动的自压缩方法，通过构建‘保留集’（Preservation Set）指导模型剪枝和量化，确保关键特征在压缩过程中不被破坏，同时提升模型泛化能力。
*   **保留集构建:** 使用 Grad-CAM 识别高激活区域、不确定性采样捕捉模型易错样本、聚类技术确保数据多样性，构建代表性强且覆盖关键特征的保留集，适用于视觉和语言模型。
*   **可微量化函数:** 设计一个动态调整权重位深度的量化函数，利用 Straight-Through Estimator (STE) 解决量化操作不可微问题，使量化过程嵌入训练中，允许模型根据数据和损失自适应调整精度。
*   **多目标损失函数:** 综合预测损失（如交叉熵）、L1 正则化项（促进稀疏性）、量化惩罚项（鼓励低位深度）和保留集损失（保护关键特征），平衡性能与压缩目标。
*   **动态反馈机制:** 在训练循环中持续监控保留集性能，若性能下降则恢复受影响组件的精度，确保压缩不损害核心功能，同时剪除冗余权重和零值组件（如 CNN 内核或 Transformer 注意力头）。
*   **适用性:** 方法在 CNN 和 Transformer 架构上均有效，展现跨领域通用性。

## Experiment

*   **压缩效果:** CNN 模型体积从 326,192 字节减至 214,730 字节（约 60% 原始大小），Transformer 模型从 831,846 字节减至 505,381 字节，显著降低资源需求。
*   **性能提升:** CNN 测试准确率从 98.6% 提升至 99.5%，Transformer 测试损失从 1.8 降至 1.6，表明方法不仅压缩模型，还通过去除参数噪声增强了泛化能力。
*   **对比优势:** 相较于无量化模型和不安全量化方法，安全驱动量化在性能与压缩之间取得更好平衡，减少训练与测试方差，提升稳定性。
*   **实验设置合理性:** 实验覆盖多种超参数（批大小、学习率、位深度 2-16 位等）和硬件环境（CPU 和 GPU），结果取平均值，确保结论可靠且不受特定配置影响。
*   **局限性:** 方法对前馈层的压缩需专用硬件支持，且训练循环需手动调整，尚未集成到标准深度学习框架中。

## Further Thoughts

‘保留集’的概念极具启发性，可扩展至强化学习中选择关键状态优化模型，或在联邦学习中为客户端定制保留集以保护个性化特征；此外，动态调整量化位深度的机制启发推理阶段根据输入复杂性实时调整模型精度，可能对边缘设备动态推理任务产生重要影响。