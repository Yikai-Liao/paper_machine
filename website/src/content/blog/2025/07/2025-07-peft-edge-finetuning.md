---
title: "From LLMs to Edge: Parameter-Efficient Fine-Tuning on Edge Devices"
pubDatetime: 2025-07-31T13:23:21+00:00
slug: "2025-07-peft-edge-finetuning"
type: "arxiv"
id: "2507.23536"
score: 0.6125513805123286
author: "grok-3-latest"
authors: ["Georg Slamanig", "Francesco Corti", "Olga Saukh"]
tags: ["LLM", "Edge Devices", "Fine-Tuning", "Low-Rank Adaptation", "Resource Efficiency"]
institution: ["Graz University of Technology", "CSH Vienna"]
description: "本文系统评估了参数高效微调（PEFT）方法在边缘设备卷积神经网络上的表现，揭示了精度与资源效率的权衡，为资源受限环境下的方法选择提供了指导。"
---

> **Summary:** 本文系统评估了参数高效微调（PEFT）方法在边缘设备卷积神经网络上的表现，揭示了精度与资源效率的权衡，为资源受限环境下的方法选择提供了指导。 

> **Keywords:** LLM, Edge Devices, Fine-Tuning, Low-Rank Adaptation, Resource Efficiency

**Authors:** Georg Slamanig, Francesco Corti, Olga Saukh

**Institution(s):** Graz University of Technology, CSH Vienna


## Problem Background

边缘设备上部署深度神经网络（DNN）面临硬件资源限制（如内存和计算能力）和数据分布偏移的问题，导致传统全模型微调（FFT）不可行。
论文旨在探索参数高效微调（PEFT）方法在资源受限环境下的卷积神经网络（CNNs）中的应用，以实现高效的模型更新，适应新任务或分布偏移。

## Method

*   **核心思想:** 通过减少更新参数数量或限制更新到低秩子空间，实现计算和内存成本的降低，适用于边缘设备上的模型更新。
*   **具体方法:**
    *   **LoRA (Low-Rank Adaptation):** 将权重更新分解为两个低秩矩阵的乘积（∆W = AB^T），仅对低秩因子A和B计算梯度和更新，显著减少参数量和计算复杂度。其简单性和兼容性使其易于应用，但低秩可能限制对复杂任务的适应。
    *   **DoRA (Weight-Decomposed Low-Rank Adaptation):** 在LoRA基础上，将预训练权重分解为大小向量和方向矩阵，仅对方向矩阵应用低秩更新，试图更接近全模型微调的行为，但计算图复杂性增加，可能导致内存开销。
    *   **GaLore (Gradient Low-Rank Projection):** 通过对梯度矩阵进行奇异值分解（SVD）并基于阈值截断秩，动态调整参数更新，以平衡效率和精度，但SVD操作在高维梯度下可能带来额外计算负担。
    *   **BN+H (Head-only Fine-Tuning with Batch-Normalization):** 仅更新分类头层和批归一化统计信息，参数量极少，资源效率高，但适应能力受限于分布偏移程度。
*   **对比基准:** 与全模型微调（FFT）相比，上述方法均旨在降低资源需求，同时尽量维持性能。

## Experiment

*   **有效性:** GaLore在不同任务和模型架构上的精度最稳定，与全模型微调（FFT）接近；LoRA和DoRA在MobileNet架构上的精度波动较大（最高差距20%），但LoRA在ResNet-18上展现了最佳精度-效率权衡；BN+H在轻量任务上表现尚可，但在复杂任务上精度差距高达40%。
*   **资源效率:** 在深度可分离卷积（DSCs）架构（如MobileNet）上，PEFT方法的内存效率仅为大型语言模型（LLMs）上的一半，LoRA在ResNet-18上峰值内存减少67%，但在MobileNetV2上仅减少22%；LoRA和DoRA在DSCs上FLOPs减少高达80%，而在标准卷积上减少57%；GaLore因SVD操作FLOPs开销比FFT高10-30%。
*   **实验设置合理性:** 实验涵盖多种模型（MobileNetV2、V3、ResNet-18）、数据集（ImageNet、CIFAR-10-C、VWW）和任务类型（分布偏移、新类别适应），设置较为全面，但未进行实际设备部署测试和量化效果评估，可能限制结果的现实适用性。

## Further Thoughts

论文揭示了模型架构（如深度可分离卷积 vs. 标准卷积）对PEFT资源效率的影响，可能比方法本身更大，启发我们是否可以设计特定于架构的PEFT策略；GaLore的动态秩调整思路提示是否可以结合任务特性或设备资源实现自适应微调；此外，LoRA在训练迭代次数上比GaLore多2倍但资源效率更高，是否意味着边缘设备上时间成本与资源成本的权衡需要更精细建模？