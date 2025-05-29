---
title: "DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning"
pubDatetime: 2025-05-26T17:20:17+00:00
slug: "2025-05-dream-prm-reweighting"
type: "arxiv"
id: "2505.20241"
score: 0.6078870672630271
author: "grok-3-latest"
authors: ["Qi Cao", "Ruiyi Wang", "Ruiyi Zhang", "Sai Ashish Somayajula", "Pengtao Xie"]
tags: ["LLM", "Multimodal Reasoning", "Process Reward Model", "Domain Reweighting", "Bi-level Optimization"]
institution: ["University of California, San Diego"]
description: "DreamPRM 通过双层优化和领域重加权策略，缓解多模态推理数据集质量不平衡问题，显著提升过程奖励模型的泛化能力和多模态大型语言模型的推理性能。"
---

> **Summary:** DreamPRM 通过双层优化和领域重加权策略，缓解多模态推理数据集质量不平衡问题，显著提升过程奖励模型的泛化能力和多模态大型语言模型的推理性能。 

> **Keywords:** LLM, Multimodal Reasoning, Process Reward Model, Domain Reweighting, Bi-level Optimization

**Authors:** Qi Cao, Ruiyi Wang, Ruiyi Zhang, Sai Ashish Somayajula, Pengtao Xie

**Institution(s):** University of California, San Diego


## Problem Background

多模态大型语言模型（MLLMs）由于需要融合视觉和语言信号，面临训练到测试分布的严重偏移（distribution shift），导致泛化能力不足；同时，现有推理数据集质量不平衡（quality imbalance），如包含噪声输入或过于简单的问题，影响过程奖励模型（PRMs）的训练效果。本文旨在通过有效的数据选择策略，解决数据集质量不平衡问题，提升多模态 PRM 的泛化能力和推理性能。

## Method

* **核心思想**：提出 DreamPRM，一个基于领域重加权（domain-reweighting）的多模态 PRM 训练框架，通过动态调整不同数据集的权重，优先关注高质量数据，减少噪声数据对模型训练的影响。
* **具体实现**：
  * **双层优化框架（Bi-level Optimization, BLO）**：
    - **下层优化**：在多个训练数据集上以领域权重进行 PRM 参数微调，使用蒙特卡洛（Monte Carlo）方法生成过程监督信号，通过均方误差（MSE）损失优化 PRM 对推理步骤的评分能力，权重参数在此阶段固定。
    - **上层优化**：在独立的元数据集（meta dataset）上评估 PRM 性能，计算基于聚合函数的损失（aggregation function loss），并据此更新领域权重，使高质量数据集获得更高权重，增强 PRM 的泛化能力。
  * **结构化思维提示（Structural Thinking Prompt）**：设计包含五个推理步骤的提示模板（重述问题、收集图像证据、识别背景知识、基于证据推理、总结结论），引导 MLLMs 进行系统化推理，便于 PRM 的步骤评估。
  * **聚合函数设计**：在推理阶段，PRM 对每个推理步骤生成概率评分，通过聚合函数计算整体轨迹得分，选择最高质量的推理轨迹作为最终输出。
* **关键特点**：相比手动设计的数据选择规则，DreamPRM 通过数据驱动的双层优化自适应调整领域权重，避免人为偏见，同时通过蒙特卡洛信号和聚合损失模拟真实推理过程，提升训练与推理的一致性。

## Experiment

* **有效性**：DreamPRM 在五个多模态推理基准（WE MATH, MATH VISTA, MATH VISION, MMVET, MMS TAR）上均优于基线 PRM（Vanilla PRM）和其他数据选择方法（如 s1-PRM 和 CaR-PRM），平均性能提升 2%-3%，在基模型 InternVL-2.5-8B-MPO 上平均提升 4%（如 MATH VISTA 从 65.4% 提升至 68.9%）。
* **优越性**：相比其他测试时扩展方法（如 Self-consistency 和 Self-correction），DreamPRM 表现出更显著的性能提升，表明高质量 PRM 对测试时扩展至关重要。
* **泛化性**：DreamPRM 在不同模型（如 GPT-4.1-mini）上也展现出性能提升（如 MATH VISTA 从 71.5% 提升至 74.4%），且随着推理轨迹（CoT）数量增加，性能持续提高，证明其跨模型泛化能力和在复杂候选池中选择高质量轨迹的能力。
* **实验设置合理性**：数据集覆盖科学、图表、几何、常识等多领域，任务类型多样，元数据集（MMMU）质量较高，适合上层优化；消融实验验证了双层优化、聚合函数损失和结构化思维提示的重要性；但蒙特卡洛采样的计算开销较高，论文也指出未来优化方向。

## Further Thoughts

DreamPRM 的领域重加权策略启发了我，是否可以将这种动态权重调整扩展到其他多源数据训练场景，如多任务学习或跨领域迁移学习，以提升模型对复杂分布的适应能力？此外，针对蒙特卡洛采样的高计算成本，是否可以通过自适应采样或强化学习方法，让模型在推理中自学习高质量轨迹特征？另外，是否可以进一步探索不同模态（视觉 vs. 文本）对推理贡献的权重分配，设计模态感知的奖励模型，深化多模态交互的理解？