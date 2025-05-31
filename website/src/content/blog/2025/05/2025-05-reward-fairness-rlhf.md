---
title: "Towards Reward Fairness in RLHF: From a Resource Allocation Perspective"
pubDatetime: 2025-05-29T11:12:00+00:00
slug: "2025-05-reward-fairness-rlhf"
type: "arxiv"
id: "2505.23349"
score: 0.5232279612496933
author: "grok-3-latest"
authors: ["Sheng Ouyang", "Yulan Hu", "Ge Chen", "Qingyang Li", "Fuzheng Zhang", "Yong Liu"]
tags: ["LLM", "Reward Model", "Fairness", "Resource Allocation", "RLHF"]
institution: ["Renmin University of China", "Kuaishou Technology", "University of Chinese Academy of Sciences"]
description: "本文从资源分配视角提出奖励公平框架，通过 Fairness Regularization 和 Fairness Coefficient 方法有效提升 RLHF 中奖励分布的公平性，同时保持模型性能，为解决多种偏差提供了通用解决方案。"
---

> **Summary:** 本文从资源分配视角提出奖励公平框架，通过 Fairness Regularization 和 Fairness Coefficient 方法有效提升 RLHF 中奖励分布的公平性，同时保持模型性能，为解决多种偏差提供了通用解决方案。 

> **Keywords:** LLM, Reward Model, Fairness, Resource Allocation, RLHF

**Authors:** Sheng Ouyang, Yulan Hu, Ge Chen, Qingyang Li, Fuzheng Zhang, Yong Liu

**Institution(s):** Renmin University of China, Kuaishou Technology, University of Chinese Academy of Sciences


## Problem Background

在强化学习从人类反馈（RLHF）中，奖励模型作为人类偏好的代理，指导大型语言模型（LLM）对齐人类偏好。然而，奖励模型常存在多种偏差（如长度偏差、类别偏差、社会偏差），导致奖励分布不公平（Reward Unfairness），进而影响模型输出与人类偏好的对齐效果。本文将这些偏差统一归纳为奖励不公平问题，试图从资源分配视角提出通用解决方案。

## Method

* **核心思想**：将偏好学习建模为资源分配问题，把奖励视为需要分配的资源，目标是在效用（Utility，即奖励与人类偏好的对齐程度）和公平性（Fairness，即奖励在不同数据类型上的分布一致性）之间取得平衡。
* **具体实现**：
  - **Fairness Regularization (FR)**：在优化目标中加入公平性正则化项，通过超参数 α 控制公平性对整体目标的影响，以加权方式平衡效用和公平性。
  - **Fairness Coefficient (FC)**：将公平性作为系数与效用相乘，通过超参数 γ 调整公平性的权重，强调公平性对整体目标的贡献。
* **公平性度量**：引入统一公平性函数（基于 Lan et al., 2010），满足连续性、齐次性和单调性，确保公平性度量的合理性。
* **应用场景**：方法被应用于验证场景（训练公平奖励模型）和强化学习场景（结合 DPO 训练公平策略模型），无需针对特定偏差设计专门方法。
* **关键点**：方法不依赖特定偏差类型，具有通用性，且通过超参数灵活调整公平性与效用的权衡。

## Experiment

* **验证场景效果**：在 HH-RLHF 和 Reward Bench 数据集上，FR 和 FC 方法显著提升了奖励分布的公平性（如 Helpful 和 Harmless 数据间的奖励分布差异减小），同时准确率与基线 Bradley-Terry 模型相当，表明在不牺牲效用的情况下实现了公平性提升；在数据选择任务中，FR 和 FC 模型展现出更高的采样效率。
* **强化学习场景效果**：在 AlpacaEval2 和 MT-Bench 数据集上，结合 DPO 的 FR 和 FC 方法在长度控制胜率（LC WR）和整体得分上优于基线，且输出长度较短，表明缓解了长度偏差。
* **实验设置合理性**：实验覆盖分布内（ID）和分布外（OOD）数据，验证了方法的泛化能力；消融实验分析了公平性函数参数 τ 和贡献参数 α/γ 的影响，证明方法对参数变化具有鲁棒性，但过高公平性权重会导致效用下降。
* **局限性**：实验主要聚焦类别偏差和长度偏差，对社会偏差的验证较为简单，未能全面覆盖奖励不公平的所有表现形式（如奖励操控）。

## Further Thoughts

资源分配视角为解决机器学习中的偏见问题提供了新思路，启发我们思考是否可以将数据不平衡或模型偏见问题也建模为资源分配问题；此外，公平性与效用的权衡机制是否可以动态调整，例如根据任务需求或用户偏好实时优化参数；还可以探索结合多目标优化或博弈论方法，进一步改进权衡策略。