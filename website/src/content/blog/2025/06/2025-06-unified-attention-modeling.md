---
title: "Unified Attention Modeling for Efficient Free-Viewing and Visual Search via Shared Representations"
pubDatetime: 2025-06-03T11:29:11+00:00
slug: "2025-06-unified-attention-modeling"
type: "arxiv"
id: "2506.02764"
score: 0.45916916919004563
author: "grok-3-latest"
authors: ["Fatma Youssef Mohammed", "Kostas Alexis"]
tags: ["Visual Attention", "Shared Representation", "Multi-Task Learning", "Transformer Architecture", "Computational Efficiency"]
institution: ["Norwegian University of Science and Technology (NTNU)"]
description: "本文通过改进 HAT 模型，证明了自由观看和视觉搜索任务之间存在共享表示，并实现了高效的注意力建模，显著降低计算成本同时保持性能。"
---

> **Summary:** 本文通过改进 HAT 模型，证明了自由观看和视觉搜索任务之间存在共享表示，并实现了高效的注意力建模，显著降低计算成本同时保持性能。 

> **Keywords:** Visual Attention, Shared Representation, Multi-Task Learning, Transformer Architecture, Computational Efficiency

**Authors:** Fatma Youssef Mohammed, Kostas Alexis

**Institution(s):** Norwegian University of Science and Technology (NTNU)


## Problem Background

人类视觉注意力建模通常将自由观看（Free-Viewing，Bottom-Up）和任务驱动的视觉搜索（Visual Search，Top-Down）分开研究，缺乏对两者是否共享共同表示（Shared Representation）的探索。
这种分离导致模型在不同任务间需要重新训练，增加了计算成本和资源需求。
本文旨在研究这两种注意力机制之间是否存在共享表示，以提高计算效率并加深对注意力机制的理解。

## Method

*   **核心思想:** 基于 Human Attention Transformer (HAT) 模型，提出一种改进的神经网络架构，通过在特征提取模块中设置共享层和任务特定层，探索自由观看和视觉搜索任务之间的共享表示。
*   **具体实现:**
    *   **特征提取模块:** 使用 ResNet-50 作为像素编码器（Pixel Encoder），从 COCO 数据集预训练权重初始化，提取通用语义特征；像素解码器（Pixel Decoder）采用 MSDeformAttn 结构，包含 6 个 Transformer 层和 8 个注意力头，输出多尺度特征图。
    *   **共享层配置:** 提出了 Late-Split (LS) 和 Early-Split (ES) 多种变体，控制像素解码器中共享层和任务特定层的数量（从全部共享到仅 1 层共享），以测试不同共享程度对性能的影响。
    *   **训练策略:** 采用两阶段训练，首先训练自由观看分支，冻结共享层后训练视觉搜索分支，假设自由观看学到的特征可以为视觉搜索提供通用基础。
    *   **后续模块:** 包括注视模块（Foveation Module）提取注视相关 token，聚合模块（Aggregation Module）整合信息，注视预测模块（Fixation Prediction Module）生成注视热图和终止概率。
*   **关键点:** 通过复用自由观看任务的特征，减少视觉搜索任务的训练参数和计算成本，同时尽量维持预测性能。

## Experiment

*   **有效性:** Late-Split (LS) 配置（全部共享层）在视觉搜索任务上的性能仅比端到端训练的 HAT 模型下降 3.86%（以 SemSS 指标计），但计算成本（GFLOPs）降低了 92.29%，训练参数减少了 31.23%。
*   **优越性:** Early-Split (ES) 配置（部分共享层）在某些指标（如 ES 2,4 配置的 cIG 和 cNSS）上甚至优于 HAT，表明适度任务特定训练可提升性能，但计算成本略增。
*   **泛化能力:** 在新收集的未见数据集（办公室场景）上，共享层模型表现出较好的泛化能力，部分配置（如 ES 4,2）在 cIG 和 cAUC 指标上优于 HAT。
*   **实验设置合理性:** 实验在 COCO-FreeView 和 COCO-Search18 数据集上进行，涵盖多种共享层配置对比，采用多指标（SemSS, SS, cIG, cNSS, cAUC）评估注视路径相似性和条件显著性，同时测试了未见数据，设置全面且合理。

## Further Thoughts

共享表示的概念可以扩展到其他多任务学习领域，如自然语言处理中不同任务的语义表示复用；此外，是否可以反向探索任务驱动特征对自由观看任务的增强作用？另外，共享层配置的优化可能通过自动化方法（如神经架构搜索）实现动态调整，以进一步平衡性能和效率。