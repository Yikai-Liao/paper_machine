---
title: "MRD-LiNet: A Novel Lightweight Hybrid CNN with Gradient-Guided Unlearning for Improved Drought Stress Identification"
pubDatetime: 2025-09-08T06:46:35+00:00
slug: "2025-09-lightweight-cnn-unlearning"
type: "arxiv"
id: "2509.06367"
score: 0.6662587749696097
author: "grok-3-latest"
authors: ["Aswini Kumar Patra", "Lingaraj Sahoo"]
tags: ["Deep Learning", "Lightweight Model", "Convolutional Neural Network", "Machine Unlearning", "Precision Agriculture"]
institution: ["North Eastern Regional Institute of Science and Technology", "Indian Institute of Technology Guwahati"]
description: "本文提出了一种轻量化混合 CNN 框架（MRD-LiNet），结合梯度引导的机器遗忘机制，以极低计算成本实现高效干旱胁迫识别，为资源受限的精准农业提供了实用解决方案。"
---

> **Summary:** 本文提出了一种轻量化混合 CNN 框架（MRD-LiNet），结合梯度引导的机器遗忘机制，以极低计算成本实现高效干旱胁迫识别，为资源受限的精准农业提供了实用解决方案。 

> **Keywords:** Deep Learning, Lightweight Model, Convolutional Neural Network, Machine Unlearning, Precision Agriculture

**Authors:** Aswini Kumar Patra, Lingaraj Sahoo

**Institution(s):** North Eastern Regional Institute of Science and Technology, Indian Institute of Technology Guwahati


## Problem Background

干旱胁迫是全球作物生产力的主要威胁，早期精准识别对可持续农业管理至关重要。
传统方法耗时费力且不适合大规模应用，而现有深度学习模型（如 CNN 和 Vision Transformer）参数量大、计算成本高，难以在资源受限的农业环境中部署；此外，模型需适应环境变化和数据噪声，对适应性提出挑战。

## Method

*   **轻量化混合 CNN 架构 (MRD-LiNet):** 设计了一个结合 ResNet、DenseNet 和 MobileNet 优势的框架，通过以下模块实现高效特征提取和分类：
    *   **初始卷积层**：提取低级特征，使用 32 个 3x3 滤波器，结合批归一化和 ReLU6 激活。
    *   **瓶颈残差块 (Bottleneck Residual Blocks)**：基于 MobileNetV2 设计，包含扩展阶段（1x1 卷积增加通道）、深度卷积阶段（3x3 深度卷积降低计算成本）和投影阶段（1x1 卷积降维），并通过跳跃连接增强梯度流动。
    *   **稠密块 (Dense Block)**：包含 4 个卷积单元，通过特征重用和稠密连接提升表示能力，缓解梯度消失问题。
    *   **过渡层 (Transition Layer)**：通过 1x1 卷积压缩通道和 2x2 平均池化下采样，控制模型复杂性，防止过拟合。
    *   **最终处理与分类**：使用全局平均池化减少参数量，通过全连接层和 sigmoid 激活输出二分类结果。
*   **机器遗忘机制 (Machine Unlearning):** 基于梯度范数的‘影响分数’（Influence Score）评估训练样本对模型预测的影响，移除影响最低的样本（如 5% 数据），并在缩减数据集上重新训练模型，以减少噪声和冗余数据的影响，提升泛化能力。
*   **训练优化策略**：采用 Adam 优化器，初始学习率为 0.001，结合指数衰减学习率调度，并通过数据增强（如旋转、平移、翻转）增加数据多样性，缓解过拟合。

## Experiment

*   **性能表现**：在马铃薯作物航拍图像数据集上，模型在数据增强结合机器遗忘（移除 5% 低影响数据）的场景下，准确率达到 90.0%，优于无增强（88.1%）和仅增强（88.6%）场景；干旱胁迫类别的召回率从 0.84 提升至 0.87，F1 分数从 0.90 提升至 0.92，减少了假阴性错误。
*   **与现有方法对比**：相比 MobileNet（88.7% 准确率，3.5M 参数）、DenseNet121（90.7%，7.09M 参数）和 ViT-TL（91.6%，14M 参数），MRD-LiNet 以仅 0.231M 参数实现 90.0% 准确率，参数量减少 15-60 倍，显著降低计算成本。
*   **实验设置合理性**：实验通过三种场景（无增强、有增强、增强+遗忘）对比，验证了各组件作用；数据集基于真实航拍图像，贴近实际应用；学习曲线和混淆矩阵分析全面评估了模型泛化能力和分类性能。
*   **局限性**：准确率略低于 ViT-TL（91.6%），在追求极致精度场景下有改进空间；实验仅针对马铃薯作物，未涉及多作物或多模态数据，泛化性待验证。

## Further Thoughts

轻量化混合架构的设计思路启发我们在其他资源受限场景中探索更多经典模型的组合方式，以实现效率与性能的平衡；机器遗忘机制不仅可用于噪声数据移除，还可能应用于隐私保护或模型动态更新，为农业领域的动态环境适应提供了新方向；轻量化模型在无人机或边缘设备上的实时部署潜力值得进一步探索，尤其结合多模态数据（如光谱图像）可能进一步提升检测精度。