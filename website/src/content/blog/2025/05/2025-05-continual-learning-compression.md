---
title: "Low-Complexity Inference in Continual Learning via Compressed Knowledge Transfer"
pubDatetime: 2025-05-13T08:07:40+00:00
slug: "2025-05-continual-learning-compression"
type: "arxiv"
id: "2505.08327"
score: 0.7076811484579548
author: "grok-3-latest"
authors: ["Zhenrong Liu", "Janne M. J. Huttunen", "Mikko Honkala"]
tags: ["Continual Learning", "Class-Incremental Learning", "Pre-Trained Model", "Model Compression", "Knowledge Distillation", "Pruning", "Inference Efficiency"]
institution: ["Nokia Bell Labs, Espoo, Finland"]
description: "本文提出剪枝和知识蒸馏两种框架，在类增量学习中通过压缩预训练模型实现性能与推理效率的平衡，显著降低计算成本并提升准确率。"
---

> **Summary:** 本文提出剪枝和知识蒸馏两种框架，在类增量学习中通过压缩预训练模型实现性能与推理效率的平衡，显著降低计算成本并提升准确率。 

> **Keywords:** Continual Learning, Class-Incremental Learning, Pre-Trained Model, Model Compression, Knowledge Distillation, Pruning, Inference Efficiency

**Authors:** Zhenrong Liu, Janne M. J. Huttunen, Mikko Honkala

**Institution(s):** Nokia Bell Labs, Espoo, Finland


## Problem Background

持续学习（Continual Learning, CL）旨在让模型顺序学习多个任务而不遗忘旧知识，但大型预训练模型在 CL 中虽性能强大，其高推理成本限制了实际应用，尤其在低延迟或低能耗场景中。
本文聚焦类增量学习（Class-Incremental Learning, CIL），解决如何在保持性能（准确率和抗遗忘能力）的同时显著降低推理复杂性的关键问题。

## Method

*   **核心思想:** 利用预训练模型的强大性能，通过模型压缩技术（剪枝和知识蒸馏）提取与下游任务相关的知识，降低推理成本，同时在 CIL 中平衡新任务学习（可塑性）和旧任务保留（稳定性）。
*   **剪枝框架 (Pruning-Based Framework):**
    *   分为预剪枝（Pre-Pruning）和后剪枝（Post-Pruning）两种策略。
    *   预剪枝：在初始任务上对预训练模型进行微调以适应下游数据分布，然后基于初始任务数据估计参数重要性进行剪枝，移除冗余参数，之后用压缩模型进行后续 CIL 训练。
    *   后剪枝：在每个任务训练后对完整模型进行剪枝，仅用于推理，保留更多下游相关知识但训练复杂度较高。
    *   技术细节：采用结构化剪枝，通过对批量归一化规模参数施加 L1 正则化促进稀疏性，根据参数重要性全局移除低影响通道，剪枝后微调以恢复性能。
*   **知识蒸馏框架 (Knowledge Distillation-Based Framework, KD):**
    *   采用教师-学生架构，教师为大型预训练模型（仅用于训练），学生为紧凑模型（用于训练和推理）。
    *   训练过程：初始任务时，教师在任务数据上微调，学生通过分类损失和蒸馏损失（KL 散度）从教师学习下游相关知识；后续任务时，学生先从前一任务的教师模型蒸馏以缓解遗忘，再结合新任务数据训练。
    *   技术细节：蒸馏损失基于温度缩放的 softmax 输出，针对先前任务类别进行知识转移，平衡可塑性和稳定性；架构灵活性允许教师和学生采用不同网络设计（如 ResNet 教师和 MobileNet 学生）。
*   **关键点:** 两种方法均不改变预训练模型的原始知识，仅通过压缩或转移实现高效推理，且可与现有 CIL 方法无缝集成。

## Experiment

*   **有效性:** KD 框架在所有数据集（CIFAR-100、FGVC Aircraft、Cars）上显著提升准确率（ACC 通常比基线高 20 个百分点，如 CIFAR-100 上结合 SS-IL 达到 49.77%），并改善后向转移（BWT），表明抗遗忘能力增强；剪枝框架在低剪枝率（如 40%）下 ACC 仅小幅下降（如 CIFAR-100 上 44.33% vs 47.33%），参数量和 FLOPs 减半。
*   **效率提升:** KD 框架推理成本极低（如 MobileNetV2 学生模型仅 0.013 GFLOPs，相比 ResNet-34 的 2.32 GFLOPs 降低超 50 倍）；剪枝框架 FLOPs 降低幅度较小（40% 剪枝率下为 1.07 GFLOPs）。
*   **对比分析:** KD 框架因架构灵活性在效率上更优，剪枝框架在低压缩率下 ACC 略高；预剪枝优于后剪枝，训练复杂度更低。
*   **实验设置合理性:** 实验覆盖分布内（CIFAR-100）和分布外（Aircraft、Cars）数据集，结合多种 CIL 方法（LwF、iCaRL、SS-IL），结果鲁棒；但在数据稀疏的分布外数据集上 KD 的 BWT 较低，分析表明这是因初始准确率高导致下降幅度大，而非遗忘加剧。

## Further Thoughts

KD 框架的教师-学生架构灵活性启发跨架构知识转移的潜力，如 Vision Transformer 教师与 CNN 学生的组合可能带来额外收益；剪枝框架中预剪枝基于初始任务数据压缩，未来可探索动态或任务自适应剪枝以保留后续任务新特征；此外，结合硬件特性（如边缘设备）优化压缩策略或能进一步提升实际应用价值。