---
title: "Basis Vector Metric: A Method for Robust Open-Ended State Change Detection"
pubDatetime: 2025-09-09T00:58:43+00:00
slug: "2025-09-basis-vector-detection"
type: "arxiv"
id: "2509.07308"
score: 0.6594690684605352
author: "grok-3-latest"
authors: ["David Oprea", "Sam Powers"]
tags: ["Image Classification", "State Detection", "Embedding", "Feature Weighting", "Supervised Learning"]
institution: ["Lumiere Foundation"]
description: "本文提出了一种基于基向量的新方法 BVM，用于图像状态变化检测，在名词-形容词对分类任务中取得最佳性能，同时具备低计算开销和易实现的特点。"
---

> **Summary:** 本文提出了一种基于基向量的新方法 BVM，用于图像状态变化检测，在名词-形容词对分类任务中取得最佳性能，同时具备低计算开销和易实现的特点。 

> **Keywords:** Image Classification, State Detection, Embedding, Feature Weighting, Supervised Learning

**Authors:** David Oprea, Sam Powers

**Institution(s):** Lumiere Foundation


## Problem Background

图像分类领域主要聚焦于静态图像分类，而对动态图像变化（如对象状态变化）的检测研究较少，导致相关任务实现困难且资源消耗大；本文旨在解决图像状态检测问题，通过设计一种高效、鲁棒的方法来识别图像中对象的细微状态变化。

## Method

* **核心思想**：提出一种名为 Basis Vector Metric (BVM) 的方法，通过训练基向量（Basis Vectors）来放大图像嵌入中区分状态的关键特征，同时抑制不重要特征，以实现精准的状态分类。
* **实现步骤**：
  1. 使用 CLIP-ViT-Large-Patch14 模型生成图像嵌入（维度为 768），作为输入数据。
  2. 定义数据集嵌入矩阵 D、基向量矩阵 B 和目标矩阵 T（包含 0 和 1，用于指示重要特征）。
  3. 初始基向量通过对每个状态（形容词）的嵌入取平均值得到，随后基于损失函数（D 与 B 点积与 T 的差异）进行训练，目标是使损失趋于 0。
  4. 训练过程中使用 Adam 优化器进行反向传播，迭代若干轮（epochs）。
  5. 在推理阶段，通过查询图像嵌入与训练后的基向量计算匹配分数（Match Scores），判断图像状态。
* **特点**：BVM 不需要大量预训练，计算开销低，易于实现和调试，且通过可视化工具（如 TSNE 图）可直观观察基向量的学习过程。

## Experiment

* **实验设置**：基于 MIT-States 数据集（约 53,000 张图像，245 个名词，115 个形容词），设计了两个任务：名词-形容词对测试（评估状态分类准确率）和形容词区分测试（评估形容词区分能力）；对比方法包括余弦相似度、点积、二进制索引、产品量化、朴素贝叶斯、自定义神经网络和逻辑回归。
* **结果**：在名词-形容词对测试中，BVM 平均准确率达 66.14%，优于朴素贝叶斯（65.23%）和其他方法，表现出明显的提升；在形容词区分测试中，BVM 准确率（40.46%）低于逻辑回归（45.13%），但通过更换嵌入模型（如 VGG19）可略有提升。
* **评估**：BVM 在状态分类任务中表现鲁棒，实验设置全面，但对嵌入模型依赖性较强，且未充分控制数据集图像一致性问题带来的干扰；计算开销低，适合资源受限场景。

## Further Thoughts

BVM 对嵌入模型的依赖性启发了我，是否可以通过联合优化嵌入生成和基向量训练来提升性能，例如微调 CLIP 模型与 BVM 联合训练以捕捉状态相关特征；此外，BVM 的基向量训练思路可扩展到文本或音频状态检测领域，并可结合注意力机制或异常检测技术进一步提升鲁棒性。