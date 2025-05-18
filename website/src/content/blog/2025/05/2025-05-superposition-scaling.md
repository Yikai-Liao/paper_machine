---
title: "Superposition Yields Robust Neural Scaling"
pubDatetime: 2025-05-15T16:18:13+00:00
slug: "2025-05-superposition-scaling"
type: "arxiv"
id: "2505.10465"
score: 0.875758750274718
author: "grok-3-latest"
authors: ["Yizhou Liu", "Ziming Liu", "Jeff Gore"]
tags: ["LLM", "Neural Scaling", "Superposition", "Representation Learning", "Model Dimension"]
institution: ["Massachusetts Institute of Technology"]
description: "本文通过玩具模型和实际 LLMs 分析，揭示超位置作为神经缩放法则核心机制的作用，解释了损失随模型维度反比缩放的鲁棒行为。"
---

> **Summary:** 本文通过玩具模型和实际 LLMs 分析，揭示超位置作为神经缩放法则核心机制的作用，解释了损失随模型维度反比缩放的鲁棒行为。 

> **Keywords:** LLM, Neural Scaling, Superposition, Representation Learning, Model Dimension

**Authors:** Yizhou Liu, Ziming Liu, Jeff Gore

**Institution(s):** Massachusetts Institute of Technology


## Problem Background

大型语言模型（LLMs）的性能随模型规模增加而提升，这一现象被称为神经缩放法则（neural scaling laws），即损失（loss）随模型规模呈幂律下降；然而，其根本原因尚未明确，作者旨在探究为何会出现这种缩放行为，并提出‘超位置’（superposition）——模型在有限维度中表示远超维度数量的特征——可能是关键机制，解决这一问题有助于理解模型性能提升的本质并指导高效模型设计。

## Method

* **核心思想**：通过一个简化的玩具模型（toy model）研究超位置对损失缩放的影响，揭示其在神经缩放法则中的作用。
* **模型设计**：玩具模型模拟特征表示学习任务，通过数据恢复（data recovery）学习特征向量；数据维度（n，表示特征数量）远大于模型维度（m，表示隐藏空间维度），特征频率分布不同（幂律、指数、线性），以反映特征重要性；模型通过权重矩阵（W）和偏置向量（b）将数据嵌入隐藏空间并重建，损失定义为重建数据与原始数据之间的均方误差（MSE）。
* **超位置控制**：通过权重衰减（weight decay）参数控制超位置程度，负值权重衰减鼓励强超位置（表示更多特征，特征向量重叠），正值权重衰减鼓励弱超位置（仅表示最频繁特征，无重叠）；训练使用改进的 AdamW 优化器，结合学习率调度和批量数据采样。
* **理论分析**：在弱超位置下，损失由未被表示的特征频率决定；在强超位置下，损失主要由特征向量间的干扰（squared overlaps）决定，几何分析表明干扰随模型维度（m）反比缩放。
* **实际验证**：分析四类开源 LLMs（OPT, GPT-2, Qwen, Pythia）的语言模型头权重矩阵，计算向量重叠和损失缩放，与玩具模型预测对比。

## Experiment

* **玩具模型结果**：在强超位置下，损失随模型维度（m）呈反比（幂律指数接近 1），对特征频率分布不敏感，表现出鲁棒性；在弱超位置下，损失缩放依赖特征频率分布，仅在幂律分布时呈幂律缩放；实验设置全面，涵盖多种数据分布、模型规模和权重衰减参数，参数扫描细致，结果可信。
* **实际 LLMs 分析**：四类开源 LLMs 的向量重叠随模型维度呈 1/m 缩放，损失幂律指数为 0.91 ± 0.04，接近玩具模型预测的 1；Chinchilla 模型推算指数为 0.88 ± 0.06，也支持强超位置机制；验证覆盖多种模型和数据集，结论普适性强。
* **局限性**：玩具模型忽略 LLMs 复杂架构和训练动态，实际损失未降至零，反映数据固有不确定性，实验未完全解释所有缩放现象。

## Further Thoughts

超位置的概念启发我们思考如何通过鼓励强超位置来优化模型设计，例如调整优化策略（如避免过度权重衰减）或架构（如限制隐藏状态到单位球面），以在较小规模下实现大模型性能；此外，超位置程度可能影响模型涌现能力（如推理）和可训练性，是否可以通过设计特定损失函数或正则化项，在预训练或后训练阶段平衡超位置与任务需求，从而提升特定任务表现？