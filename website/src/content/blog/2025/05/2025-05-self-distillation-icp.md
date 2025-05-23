---
title: "Self Distillation via Iterative Constructive Perturbations"
pubDatetime: 2025-05-20T13:15:27+00:00
slug: "2025-05-self-distillation-icp"
type: "arxiv"
id: "2505.14751"
score: 0.6044248951504156
author: "grok-3-latest"
authors: ["Maheak Dave", "Aniket Kumar Singh", "Aryan Pareek", "Harshita Jha", "Debasis Chaudhuri", "Manish Pratap Singh"]
tags: ["Deep Learning", "Self-Distillation", "Input Optimization", "Feature Alignment", "Perturbation"]
institution: ["Techno India University", "DRDO Young Scientist Laboratory - Cognitive Technologies"]
description: "本文提出一种结合迭代构造性扰动和自蒸馏的框架，通过输入优化和特征对齐显著提升深度神经网络的性能和泛化能力。"
---

> **Summary:** 本文提出一种结合迭代构造性扰动和自蒸馏的框架，通过输入优化和特征对齐显著提升深度神经网络的性能和泛化能力。 

> **Keywords:** Deep Learning, Self-Distillation, Input Optimization, Feature Alignment, Perturbation

**Authors:** Maheak Dave, Aniket Kumar Singh, Aryan Pareek, Harshita Jha, Debasis Chaudhuri, Manish Pratap Singh

**Institution(s):** Techno India University, DRDO Young Scientist Laboratory - Cognitive Technologies


## Problem Background

深度神经网络在多个领域取得了显著成就，但训练过程中平衡性能和泛化能力仍是一个挑战。
传统方法多集中于模型架构、训练方法或超参数优化，计算复杂性较高，而现有自蒸馏方法在计算开销或特征表示有效性上存在局限，缺乏对输入数据的系统性优化。
论文旨在通过同时优化模型参数和输入表示，弥合性能与泛化之间的差距。

## Method

*   **核心思想:** 提出一种结合迭代构造性扰动（Iterative Constructive Perturbation, ICP）和自蒸馏的框架，通过迭代优化输入数据和对齐中间特征表示来提升模型性能。
*   **ICP 实现细节:** 
    *   ICP 借鉴快速梯度符号方法（FGSM）的梯度思想，但方向相反，通过迭代梯度下降调整输入数据以最小化模型损失，使输入更贴近模型的特征空间。
    *   每次迭代中，根据当前输入计算任务损失的梯度，沿负梯度方向更新输入，迭代次数由参数 T 控制。
    *   提出多种 ICP 变体，包括基于随机梯度下降的 SGD-ICP、基于 Adam 优化器的 Adam-ICP 以及引入额外动量项的 AdEMAMix-ICP，以提升扰动生成的稳定性和效率。
*   **自蒸馏实现细节:** 
    *   将原始输入和 ICP 扰动后的输入分别通过模型，获取中间特征图，并通过均方误差（MSE）计算层级蒸馏损失以对齐特征表示。
    *   采用余弦衰减策略动态调整任务损失和蒸馏损失的权重，初期注重任务性能，后期逐渐关注特征对齐。
*   **关键点:** 该方法强调输入优化的前瞻性，而非单纯依赖模型权重优化，同时通过层级权重分配捕捉从基础到抽象的特征。

## Experiment

*   **有效性:** 实验在图像分类（CIFAR-100 数据集）和图像生成（CUB 数据集）任务上展开，结果显示 ICP 结合自蒸馏显著提升性能。例如，AdEMAMix-ICP 在分类任务中准确率比基线提高 19.06%，F1 分数也有改善；在生成任务中，SSIM 和 FID 指标优于基线。
*   **实验设置:** 实验设置较为全面，涵盖不同任务、不同 ICP 变体（SGD-ICP、Adam-ICP、AdEMAMix-ICP）以及超参数调优（如 k=25, T=5），所有模型训练 100 轮次以确保一致性。
*   **局限与开销:** 尽管方法提升明显，但计算开销增加，训练时间较基线更长，尤其在资源受限环境下生成任务质量仍较低。

## Further Thoughts

输入优化的前瞻性策略是一个亮点，通过迭代扰动输入数据提升特征表示质量的思路可扩展至其他领域，如自然语言处理中对文本嵌入的动态扰动以增强鲁棒性；此外，ICP 结合不同优化算法的思路启发我们探索优化算法在输入处理中的更广泛应用，例如是否可以通过更先进的优化器进一步提升扰动效率或稳定性。