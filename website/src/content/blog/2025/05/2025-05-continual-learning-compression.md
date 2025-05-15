---
title: "Low-Complexity Inference in Continual Learning via Compressed Knowledge Transfer"
pubDatetime: 2025-05-13T08:07:40+00:00
slug: "2025-05-continual-learning-compression"
type: "arxiv"
id: "2505.08327"
score: 0.7076811484579548
author: "grok-3-latest"
authors: ["Zhenrong Liu", "Janne M. J. Huttunen", "Mikko Honkala"]
tags: ["Continual Learning", "Knowledge Distillation", "Model Compression", "Pre-Training", "Class-Incremental Learning"]
institution: ["Nokia Bell Labs, Espoo, Finland"]
description: "本文提出剪枝和知识蒸馏两种框架，成功在持续学习（特别是类增量学习）中利用预训练模型的强大性能，通过模型压缩显著降低推理成本，同时保持高准确性和抗遗忘能力。"
---

> **Summary:** 本文提出剪枝和知识蒸馏两种框架，成功在持续学习（特别是类增量学习）中利用预训练模型的强大性能，通过模型压缩显著降低推理成本，同时保持高准确性和抗遗忘能力。 

> **Keywords:** Continual Learning, Knowledge Distillation, Model Compression, Pre-Training, Class-Incremental Learning

**Authors:** Zhenrong Liu, Janne M. J. Huttunen, Mikko Honkala

**Institution(s):** Nokia Bell Labs, Espoo, Finland


## Problem Background

持续学习（Continual Learning, CL）旨在让模型顺序学习多个任务而不遗忘旧知识，但神经网络常遭受灾难性遗忘（Catastrophic Forgetting），即新任务学习后旧任务性能下降。
近年来，大型预训练模型因其强大的泛化能力和抗遗忘能力被广泛用于 CL，然而其高昂的推理计算成本限制了在低延迟或低能耗场景下的应用。
论文聚焦于类增量学习（Class-Incremental Learning, CIL），一种更具挑战性的设置（推理时无法获取任务身份），试图解决如何在保持模型性能的同时显著降低推理成本的问题。

## Method

*   **剪枝框架（Pruning-Based Framework）:**
    *   提出两种策略：预剪枝（Pre-Pruning）和后剪枝（Post-Pruning）。
    *   预剪枝：在初始任务上对预训练模型进行微调以适应下游数据分布，然后基于初始任务数据估计参数重要性进行剪枝，得到压缩模型用于后续 CIL 训练；此策略假设初始任务知识可近似代表下游相关知识。
    *   后剪枝：在每个任务训练后进行剪枝，保留完整预训练模型进行训练，仅在推理时使用压缩模型；此策略能利用更多任务积累的知识，但训练复杂度较高。
    *   剪枝技术：采用结构化剪枝，通过对批量归一化（Batch Normalization）缩放参数施加 L1 正则化来实现通道剪枝，依据参数重要性移除冗余神经元，并在剪枝后微调以恢复性能。
*   **知识蒸馏框架（Knowledge Distillation-Based Framework）:**
    *   采用教师-学生架构，教师为大型预训练模型（仅用于训练），学生为小型模型（用于训练和推理）。
    *   训练过程分阶段：初始任务时，教师在任务数据上微调后通过知识蒸馏（KD）指导学生，增强学生的适应性（Plasticity）；后续任务时，学生先从前一任务的教师模型蒸馏旧任务知识以缓解遗忘，再在当前任务上训练。
    *   蒸馏方法：通过 KL 散度对齐教师和学生的 logits，尤其针对旧任务类别进行蒸馏以提高稳定性（Stability），并可灵活选择不同架构的教师和学生模型（如 ResNet-34 作为教师，MobileNetV2 作为学生）。
*   **共同目标:** 两种框架均旨在利用预训练模型的强大性能，通过压缩提取下游相关知识，降低推理成本，同时平衡新任务适应性和旧任务知识保留。

## Experiment

*   **数据集与设置:** 实验在 CIFAR-100（分布内，ID）、FGVC Aircraft 和 Cars（分布外，OoD）三个 CIL 基准数据集上进行，评估指标包括平均准确率（ACC）、后向转移（BWT，衡量遗忘程度）和推理成本（FLOPs）。测试了多种基线方法（如 LwF, iCaRL, SS-IL）及其增强版本，涵盖不同压缩比例和学生模型变体。
*   **KD 框架效果:** 在所有数据集上，KD 框架显著提升 ACC（相较基线通常提高约 20 个百分点，例如在 CIFAR-100 上从 LwF 的 11.81% 提升至 40.19%），BWT 也有改善（遗忘减少，如在 CIFAR-100 上从 -43.60 提升至 -0.14）。推理成本极低（MobileNetV2 仅 0.013 GFLOPs），尤其在 OoD 数据集上展现了强泛化能力。
*   **剪枝框架效果:** 预剪枝优于后剪枝，在 40% 剪枝比例下 ACC 仅下降 3 个百分点（从 47.33% 至 44.33%），参数量和 FLOPs 减少约一半（从 21.28M 和 2.32G 降至 11.31M 和 1.07G），显示出较好的性能-效率权衡。
*   **对比与合理性:** KD 框架在 FLOPs 和参数量减少上更优（得益于架构灵活性），而剪枝框架在低剪枝比例下 ACC 略高。实验设置全面，涵盖 ID 和 OoD 场景，数据稀疏和丰富场景，多种方法对比，数据可信度高，充分验证了方法的有效性。

## Further Thoughts

论文中关于‘下游相关知识’（Downstream-Relevant Knowledge）的提取是一个重要启发，提示我们可以通过有针对性的压缩技术（如剪枝或 KD）从大型预训练模型中提炼出对特定任务最有用的部分，这不仅适用于持续学习，也可能推广到其他领域如迁移学习。
此外，KD 框架中教师和学生模型架构灵活性的设计启发我们，在模型压缩时可以更多探索架构创新，而非仅关注参数减少，例如结合深度可分离卷积等高效设计。
最后，论文揭示了在数据稀疏场景下预训练知识的重要性，这提示在资源受限或数据有限的边缘设备应用中，预训练模型可能是性能提升的关键驱动力。