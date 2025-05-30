---
title: "TuneComp: Joint Fine-tuning and Compression for Large Foundation Models"
pubDatetime: 2025-05-27T23:49:35+00:00
slug: "2025-05-tunecomp-compression"
type: "arxiv"
id: "2505.21835"
score: 0.8756177586252445
author: "grok-3-latest"
authors: ["Xiangyu Chen", "Jing Liu", "Ye Wang", "Matthew Brand", "Pu Wang", "Toshiaki Koike-Akino"]
tags: ["LLM", "Distillation", "Pre-Training", "Post-Training", "Compression"]
institution: ["Mitsubishi Electric Research Laboratories (MERL)"]
description: "本文提出 TuneComp 框架，通过联合微调、知识蒸馏、低秩近似和剪枝，直接生成小型高效模型，在高压缩率下显著提升性能。"
---

> **Summary:** 本文提出 TuneComp 框架，通过联合微调、知识蒸馏、低秩近似和剪枝，直接生成小型高效模型，在高压缩率下显著提升性能。 

> **Keywords:** LLM, Distillation, Pre-Training, Post-Training, Compression

**Authors:** Xiangyu Chen, Jing Liu, Ye Wang, Matthew Brand, Pu Wang, Toshiaki Koike-Akino

**Institution(s):** Mitsubishi Electric Research Laboratories (MERL)


## Problem Background

大型基础模型（如 Transformer）在自然语言处理和计算机视觉任务中表现出色，但其巨大的参数量和计算需求使得在资源受限设备上的微调和部署变得困难。
传统的顺序微调和压缩方法（先微调后压缩或先压缩后微调）会导致性能损失，并产生不必要的中间计算开销。
本文旨在解决这一问题，通过联合微调和压缩直接生成小型高效模型，避免性能下降和资源浪费。

## Method

* **核心思想**：提出 TuneComp 框架，通过渐进式压缩和知识蒸馏，将预训练模型（教师分支）逐步过渡到压缩后的学生分支，同时对学生分支进行低秩近似和剪枝，确保性能与效率的平衡。
* **双分支结构**：每个线性层分为两个并行分支，教师分支为冻结的预训练权重，学生分支为可训练的低秩结构；通过衰减因子 *α* 逐步减少教师分支影响，最终完全依赖学生分支。
* **低秩近似**：采用奇异值分解（SVD）对权重矩阵进行低秩分解，并提出激活感知的初始化方法 RootCorDA，根据激活统计优化低秩矩阵初始化，提升压缩后性能。
* **剪枝策略**：在低秩结构上应用硬收缩（Hard Shrink）进行剪枝，动态计算阈值去除不重要元素，进一步减少参数量。
* **层级正则化**：引入层级特征损失，指导学生分支学习教师分支中间表示，并采用动态衰减的正则化权重策略，在训练初期提供更多指导，后期逐步放松约束。
* **整体流程**：将微调、知识蒸馏、低秩近似和剪枝整合到一个统一框架，避免传统顺序方法的性能损失。

## Experiment

* **有效性**：在 Vision Transformer (ViT) 模型上，从 ImageNet1K 迁移到 CIFAR100 任务，TuneComp 在高压缩率下显著优于基线方法（如先微调后蒸馏、先蒸馏后微调、PC-LoRA），在准确率-参数量权衡上表现最佳（Pareto 前沿图）。
* **初始化影响**：激活感知的 RootCorDA 初始化方法在低秩（如 r=32）时准确率提升明显，优于其他初始化策略（如 Gaussian、CorDA）。
* **剪枝效果**：适度剪枝（20%-40%）进一步提升准确率-效率权衡，但过高剪枝率（60%以上）导致性能下降。
* **正则化策略**：动态衰减正则化权重优于常数正则化，尤其在低秩下提升显著。
* **实验局限**：实验设置较为全面，涵盖不同压缩率和策略的影响，但仅在 ViT 和图像分类任务上验证，缺乏对其他领域（如 NLP）或模型架构的泛化性测试；未详细讨论计算开销（如训练时间、内存占用）。

## Further Thoughts

联合优化（微调与压缩同步）的潜力巨大，是否可以将其他目标（如量化、硬件适配）纳入框架，进一步减少后处理步骤？
激活感知初始化表明输入数据统计特性对压缩至关重要，是否可以根据输入难度动态调整压缩策略？
渐进式过渡避免性能突变，是否可应用于模型迁移或增量学习，逐步引入新数据或新模型的影响？