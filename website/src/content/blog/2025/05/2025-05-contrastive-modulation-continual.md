---
title: "Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning"
pubDatetime: 2025-05-20T09:31:57+00:00
slug: "2025-05-contrastive-modulation-continual"
type: "arxiv"
id: "2505.14125"
score: 0.6699488116308125
author: "grok-3-latest"
authors: ["Viet Anh Khoa Tran", "Emre Neftci", "Willem A. M. Wybo"]
tags: ["Continual Learning", "Contrastive Learning", "Modulation Invariance", "Sparse Supervision", "Class-Incremental Learning"]
institution: ["Peter Grünberg Institute 15 – Neuromorphic Software Ecosystems, Forschungszentrum Jülich", "Faculty of Electrical Engineering and Information Technology, RWTH Aachen"]
description: "本文提出任务调制对比学习（TMCL），通过生物启发的顶向下调制和对比学习整合稀疏监督信号，实现类增量持续学习中稳定性与可塑性的平衡，并在标签稀疏场景下显著优于现有方法。"
---

> **Summary:** 本文提出任务调制对比学习（TMCL），通过生物启发的顶向下调制和对比学习整合稀疏监督信号，实现类增量持续学习中稳定性与可塑性的平衡，并在标签稀疏场景下显著优于现有方法。 

> **Keywords:** Continual Learning, Contrastive Learning, Modulation Invariance, Sparse Supervision, Class-Incremental Learning

**Authors:** Viet Anh Khoa Tran, Emre Neftci, Willem A. M. Wybo

**Institution(s):** Peter Grünberg Institute 15 – Neuromorphic Software Ecosystems, Forschungszentrum Jülich, Faculty of Electrical Engineering and Information Technology, RWTH Aachen


## Problem Background

在持续学习（Continual Learning）场景下，特别是在类增量学习（Class-Incremental Learning）中，传统机器学习方法容易遭受灾难性遗忘（Catastrophic Forgetting），即在新任务学习时丢失对旧任务的记忆。
论文从生物大脑的学习机制中汲取灵感，试图解决如何在数据流中以无监督方式持续学习，同时利用稀疏监督信号（如1%的标签）增强特定类别的表示，而不破坏已学的通用表示的问题。
这一问题在自然学习环境中尤为重要，因为生物体通常在无监督数据流中学习，偶尔接收稀疏的监督信息。

## Method

*   **核心思想:** 提出任务调制对比学习（Task-Modulated Contrastive Learning, TMCL），模仿生物大脑中顶向下（Top-Down）和自下向上（Bottom-Up）信息流的交互，通过学习任务特定的调制参数（Modulations）来增强新类别的表示，同时利用自监督对比学习将这些调制整合到共享表示空间中，避免灾难性遗忘。
*   **具体实现:** 方法分为两个阶段：
    *   **正交化阶段（Orthogonalization）:** 当观察到新类别的标签时，在冻结前馈权重（Feedforward Weights）的基础上，学习一组调制参数（包括增益和偏置），通过正交投影损失（Orthogonal Projection Loss, OPL）使新类别的表示与已有类别的表示正交化，从而增强类别区分度。这一阶段仅使用当前可用的稀疏标签样本作为正样本，其他样本作为负样本，避免了传统全对全分类的需求。
    *   **巩固阶段（Consolidation）:** 使用自监督对比学习（如Barlow Twins）在无调制表示空间中训练前馈权重，将调制后的表示整合到无调制表示空间中，形成调制不变性（Modulation Invariance）。通过将不同调制视图和无调制视图作为正样本对，促使模型学习对调制不变的表示，同时利用历史调制参数稳定表示空间，减少遗忘。
*   **关键点:** 调制参数不影响前馈权重更新，仅在正交化阶段学习并冻结，巩固阶段通过对比学习实现知识整合；这种分离机制模仿了生物大脑中近端（Perisomatic）和远端（Apical）区域的功能区分，提供了稳定性与可塑性的平衡。

## Experiment

*   **有效性:** 在CIFAR-100类增量学习基准测试（5个会话，每个会话20个类别）中，TMCL在标签稀疏场景（1%或10%标签）下显著优于纯自监督方法（如Barlow Twins）和纯监督方法（如SupCon），例如在1%标签条件下，结合调制不变性（MI）的kNN准确率和线性探针准确率均表现出色。
*   **优越性:** 相比纯自监督方法，TMCL通过稀疏监督信号显著提升了类别区分能力；相比纯监督方法，其在标签稀疏场景下表现更稳定，尤其在转移学习任务中（如Aircraft、CUBirds等数据集），学到的表示展现出更好的泛化能力。
*   **实验设置合理性:** 实验设置较为全面，涵盖了不同标签比例（1%、10%、100%）和多种评估方式（kNN和线性探针），同时对比了多种基线方法（如VI、SI、SupCon、CE）；此外，还测试了不同架构（ConViT和ResNet-18）的影响，验证了方法的鲁棒性。
*   **局限性:** 在完全标签场景下，TMCL的表现略逊于SupCon，表明其设计更适合稀疏监督场景；此外，调制参数随类别数量增加而线性增长，可能导致内存问题，实验未探讨数据增量或领域增量场景。

## Further Thoughts

调制不变性（Modulation Invariance）的概念非常具有启发性，不仅可以用于持续学习，还可能扩展到多任务学习或跨模态学习中，通过学习不同任务或模态的调制参数，动态调整模型的表示空间，实现更灵活的知识整合。
此外，生物启发的顶向下调制机制是否可以与强化学习结合，用于动态调整学习目标或策略？例如，通过调制参数模拟奖励信号的上下文依赖性，可能为智能体在动态环境中的学习提供新的思路。
另一个值得探索的方向是调制参数的生成机制，是否可以通过一个小型生成网络动态生成调制参数，从而解决内存随类别数量增加的问题，同时保持表示的灵活性？