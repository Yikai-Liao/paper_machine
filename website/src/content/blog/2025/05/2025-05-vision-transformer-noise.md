---
title: "Balancing Accuracy, Calibration, and Efficiency in Active Learning with Vision Transformers Under Label Noise"
pubDatetime: 2025-05-07T12:53:13+00:00
slug: "2025-05-vision-transformer-noise"
type: "arxiv"
id: "2505.04375"
score: 0.49886477444502625
author: "grok-3-latest"
authors: ["Moseli Mots’oehli", "Hope Mogale", "Kyungim Baek"]
tags: ["Vision Transformer", "Active Learning", "Label Noise", "Model Calibration", "Computational Efficiency"]
institution: ["University of Hawai‘i at Manoa", "University of Pretoria"]
description: "本文系统分析了视觉变换器在主动学习和标签噪声环境下的表现，揭示模型规模和补丁大小对准确率、校准和效率的影响，为资源受限场景下的模型选择提供实用指导。"
---

> **Summary:** 本文系统分析了视觉变换器在主动学习和标签噪声环境下的表现，揭示模型规模和补丁大小对准确率、校准和效率的影响，为资源受限场景下的模型选择提供实用指导。 

> **Keywords:** Vision Transformer, Active Learning, Label Noise, Model Calibration, Computational Efficiency

**Authors:** Moseli Mots’oehli, Hope Mogale, Kyungim Baek

**Institution(s):** University of Hawai‘i at Manoa, University of Pretoria


## Problem Background

视觉变换器（Vision Transformers, ViT）在主动学习（Active Learning）场景下的应用研究较少，尤其是在标签噪声（Label Noise）普遍存在的现实环境中，如何平衡分类准确率、模型校准和计算效率仍是一个未充分探索的问题。
论文旨在解决在有限标注预算和资源约束下，标签噪声对不同规模和配置的 ViT 模型性能的影响，以及如何选择合适的模型配置以优化性能与效率的权衡。

## Method

*   **模型配置与对比**：研究了两种变换器架构：原始 ViT 和 Swin Transformer V2。ViT 包括 Base 和 Large 两种规模，结合 16x16 和 32x32 补丁大小（即 ViTb16, ViTb32, ViTl16, ViTl32）；SwinV2 包括 Tiny、Small 和 Base 三种规模，均使用 4x4 补丁大小，旨在分析模型容量和补丁大小对性能的影响。
*   **标签噪声模拟**：采用对称标签噪声，噪声率从 0% 到 90% 逐步增加（步长 10%），噪声仅在训练阶段注入，测试集保持干净，以评估模型在噪声环境下的泛化能力。
*   **主动学习策略**：实现了三种查询策略：随机查询（Random Query）作为基线，基于熵的查询（Entropy-based Selection）选择不确定性最高的样本，以及专为 ViT 设计的 GCI_ViTAL 策略，通过结合熵和注意力向量的 Frobenius 范数选择语义挑战性样本。
*   **实验设置**：在 CIFAR10 和 CIFAR100 数据集上进行实验，图像调整为 224x224 以适应变换器输入，模型基于 ImageNet-1k 预训练后微调，训练 20 个 epoch 并使用早停机制。
*   **评估指标**：使用 Top-1 准确率评估分类性能，Brier Score 评估模型校准（值越低越好），并记录训练时间以衡量计算效率。

## Experiment

*   **准确率表现**：实验结果表明，随着标签噪声率增加，所有模型准确率均下降，但较大规模的 ViT 模型（如 ViTl32）在高噪声下表现更优，例如在 CIFAR10 上 70% 噪声率下准确率为 88.28%，显著高于 ViTb32 的 81.11%；SwinV2 模型普遍逊于 ViT；主动学习策略中，GCI_ViTAL 在中度噪声（30%-60%）下优于随机和熵策略，但在极高噪声下优势消失。
*   **校准效果**：Brier Score 显示噪声增加导致校准变差，但 ViTl32 保持较好校准（CIFAR10 上 70% 噪声率下 Brier Score 为 48.28%，优于 ViTl16 的 51.07%）；随机查询在高噪声下校准优于信息驱动策略。
*   **效率分析**：训练时间与标注数据比例呈线性关系，补丁大小显著影响效率，ViTl16 训练时间远高于 ViTl32（228s vs 90s），但性能提升不明显，性价比低；SwinV2 训练时间与 ViT 相当，但性能较差。
*   **实验设置合理性**：实验覆盖多种模型配置、噪声率和主动学习策略，数据集选择具有代表性，设置全面合理，但对补丁大小影响的机制解释略显不足。

## Further Thoughts

论文揭示了较大规模 ViT 模型在噪声环境下的鲁棒性，启发我们思考是否可以通过模型剪枝或知识蒸馏在小模型上模拟大模型的鲁棒性；补丁大小的非直觉性影响（ViTl32 优于 ViTl16）提示是否可以通过动态调整补丁大小适应不同任务需求；主动学习策略在高噪声下校准不佳，是否可以设计联合优化准确率和校准的查询策略；此外，如何构建一个根据任务需求自动推荐最优模型配置的决策框架也值得探索。