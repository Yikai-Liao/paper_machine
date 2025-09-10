---
title: "UNO: Unifying One-stage Video Scene Graph Generation via Object-Centric Visual Representation Learning"
pubDatetime: 2025-09-07T18:30:41+00:00
slug: "2025-09-unified-vid-sgg"
type: "arxiv"
id: "2509.06165"
score: 0.7465454096490755
author: "grok-3-latest"
authors: ["Huy Le", "Nhat Chung", "Tung Kieu", "Jingkang Yang", "Ngan Le"]
tags: ["Video Understanding", "Scene Graph", "Object-Centric Learning", "Temporal Consistency", "Slot Attention"]
institution: ["FPT Software AI Center, Vietnam", "Aalborg University, Denmark", "Pioneer Centre for AI, Denmark", "Nanyang Technological University, Singapore", "AICV Lab, University of Arkansas, USA"]
description: "本文提出 UNO，一个单阶段、统一的视频场景图生成框架，通过对象中心表示学习同时处理框级和像素级任务，显著提升性能和效率。"
---

> **Summary:** 本文提出 UNO，一个单阶段、统一的视频场景图生成框架，通过对象中心表示学习同时处理框级和像素级任务，显著提升性能和效率。 

> **Keywords:** Video Understanding, Scene Graph, Object-Centric Learning, Temporal Consistency, Slot Attention

**Authors:** Huy Le, Nhat Chung, Tung Kieu, Jingkang Yang, Ngan Le

**Institution(s):** FPT Software AI Center, Vietnam, Aalborg University, Denmark, Pioneer Centre for AI, Denmark, Nanyang Technological University, Singapore, AICV Lab, University of Arkansas, USA


## Problem Background

视频场景图生成（VidSGG）旨在从视频中提取结构化动态表示，通过将对象和交互建模为时空图，支持视频理解和推理等任务。
现有方法分为框级（DSGG）和像素级（PVSG）两种粒度，通常需要任务特定的架构和多阶段训练管道，导致计算开销大、参数共享少、泛化能力受限。
本文致力于解决如何设计一个统一的单阶段框架，同时处理两种粒度的 VidSGG 任务，减少任务特定修改并提升效率和性能。

## Method

*   **核心思想:** 提出 UNO（UNified Object-centric VidSGG），一个单阶段、端到端的框架，通过对象中心表示学习统一框级和像素级 VidSGG 任务，最大化参数共享，减少任务特定设计。
*   **视觉编码:** 使用预训练的视觉编码器（如 Vision Transformer 或 ResNet-50）从视频帧中提取特征图，作为后续分解的基础，特征图包含丰富的对象信息但语义纠缠。
*   **对象分解:** 采用扩展的 Slot Attention 机制，将特征图分解为对象槽（object slots），每个槽代表一个独立对象区域，通过竞争机制和 GRU 迭代更新，确保槽捕获模块化语义，支持跨帧一致性。
*   **对象时序一致性学习:** 引入对比损失，强制对象槽在不同帧间保持一致性，无需显式跟踪模块，通过正样本（相同对象）和负样本（不同对象）距离优化，提升时空表示稳定性。
*   **关系分解:** 使用 Slot Attention 将特征图分解为关系槽（relation slots），捕获对象间潜在交互区域，与对象槽并行预测，避免传统顺序预测的低效。
*   **动态三元组预测:** 提出动态三元组预测模块，通过关系槽生成主体和客体引用嵌入，与对象槽进行相似性匹配，形成三元组（subject-relation-object），避免构建冗余对象对矩阵，减少计算复杂度和预测冗余。
*   **训练目标:** 针对 DSGG 和 PVSG 任务设计损失函数，包括分类损失、边界框/掩码损失和关系匹配损失，支持端到端训练。

## Experiment

*   **有效性:** 在 Action Genome 数据集（DSGG 任务）上，UNO 在 SGDET 任务中 R@20 达到 45.2%（比第二名高 4.3%），在 PredCLS 任务中 R@20 达到 80.3%；在 PVSG 数据集（PVSG 任务）上，R@20 达到 9.44%（vIOU=0.5，远超基线 4.51%），在 vIOU=0.1 时达到 17.54%，表明在两种粒度任务上均显著优于现有方法。
*   **效率:** 作为单阶段框架，UNO 避免了多阶段管道的计算开销，通过参数共享和并行预测进一步提升效率。
*   **实验设置合理性:** 实验针对两个任务独立训练和评估，避免数据混淆，使用多种预训练骨干网络（如 ViT 和 ResNet）验证鲁棒性，消融实验分析多任务学习和时序一致性学习效果，设置全面合理。
*   **不足:** 论文指出在对象定位和复杂动态场景（如对象消失/重现、拥挤场景）中仍有改进空间，高召回率指标（R@50, R@100）提升幅度不如 R@20 显著。

## Further Thoughts

对象中心表示通过 Slot Attention 机制统一潜在表示空间，不仅适用于 VidSGG，还可能推广到其他多模态任务（如图像-文本场景图生成），为跨任务学习提供新思路；
时序一致性学习的无跟踪设计通过对比损失实现对象跨帧一致性，这种轻量级方法在资源受限场景中具有潜力，可探索其在视频目标跟踪等任务中的应用；
动态三元组预测利用稀疏性优化计算复杂度，启发我们在其他图结构预测任务（如社交网络分析、知识图谱构建）中设计高效匹配机制。