---
title: "Learning Dynamics in Continual Pre-Training for Large Language Models"
pubDatetime: 2025-05-12T17:47:32+00:00
slug: "2025-05-cpt-scaling-law"
type: "arxiv"
id: "2505.07796"
score: 0.7719812832393543
author: "grok-3-latest"
authors: ["Xingjin Wang", "Howe Tissue", "Lu Wang", "Linjing Li", "Daniel Dajun Zeng"]
tags: ["LLM", "Continual Pre-Training", "Scaling Law", "Distribution Shift", "Learning Rate Annealing"]
institution: ["School of Artificial Intelligence, University of Chinese Academy of Sciences", "State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences", "Ritzz-AI"]
description: "本文提出了一种CPT缩放定律，通过解耦数据分布偏移和学习率退火的影响，量化描述大型语言模型持续预训练过程中的学习动态，并指导超参数优化以平衡通用和下游性能。"
---

> **Summary:** 本文提出了一种CPT缩放定律，通过解耦数据分布偏移和学习率退火的影响，量化描述大型语言模型持续预训练过程中的学习动态，并指导超参数优化以平衡通用和下游性能。 

> **Keywords:** LLM, Continual Pre-Training, Scaling Law, Distribution Shift, Learning Rate Annealing

**Authors:** Xingjin Wang, Howe Tissue, Lu Wang, Linjing Li, Daniel Dajun Zeng

**Institution(s):** School of Artificial Intelligence, University of Chinese Academy of Sciences, State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences, Ritzz-AI


## Problem Background

持续预训练（Continual Pre-Training, CPT）是大型语言模型（LLMs）适应特定下游任务或领域（如代码、金融、数学等）的重要方法。
然而，CPT过程中通用领域和下游领域的性能如何随训练步骤演变仍缺乏量化描述，尤其是在面对灾难性遗忘（catastrophic forgetting）问题时，即下游性能提升可能导致通用性能下降。
本文旨在研究CPT的学习动态（learning dynamics），解决如何综合建模影响性能的多种因素（如数据分布偏移、学习率退火等），并预测任意训练步骤的验证损失。

## Method

*   **核心思想:** 提出一种CPT缩放定律（CPT scaling law），通过解耦数据分布偏移（distribution shift）和学习率退火（learning rate annealing）的影响，量化描述CPT过程中的损失曲线（loss curve），以预测任意训练步骤和学习率调度下的验证损失。
*   **具体实现:**
    *   **分布偏移建模:** 观察到CPT损失曲线是从通用领域预训练（PT）数据集的隐藏曲线向CPT数据集隐藏曲线的转移曲线，分布偏移项描述了这一偏差。实验验证该项符合幂律形式（power-law form），且与转移起点和模型大小无关，但受学习率调度影响，通过引入前向面积（forward area）调整其表达式。
    *   **学习率退火建模:** 基于前人工作，将学习率退火的影响建模为损失下降的局部效应，结合前向面积（S1）和退火面积（S2）进行量化，分别对应PT和CPT阶段的不同系数。
    *   **综合公式:** 将上述两项结合，形成统一的CPT缩放定律，能够描述PT和CPT阶段的损失动态，并扩展到模型大小（N）和数据回放比例（replay ratio）等变量，增强适用性。
    *   **超参数优化:** 利用该定律分析关键因素（如损失潜力、峰值学习率、训练步骤、回放比例）对性能的影响，指导CPT过程中超参数的选择，以平衡通用和下游性能。
*   **关键特点:** 不依赖特定模型或数据集，方法具有普适性；支持对开源模型（未知PT信息）和域外数据集（out-of-domain）损失的预测，通过代理数据集和线性组合等策略实现。

## Experiment

*   **有效性:** 提出的CPT缩放定律在多种数据集（如FineWeb、Knowledge-Pile、Pile-of-Law）、模型规模（106M到1.7B参数）和学习率调度（常量、余弦、WSD）下均能准确拟合和预测损失曲线，验证了其描述学习动态的能力。
*   **性能提升:** 通过定律预测的超参数优化策略（如选择较高损失潜力的PT模型、调整峰值学习率）显著改善下游领域性能，同时缓解通用领域损失上升，例如在Knowledge-Pile数据集上，较高损失潜力的模型在下游验证损失上表现更优。
*   **实验设置合理性:** 实验覆盖了多种实际场景，包括不同批大小、序列长度、回放比例的变化，以及开源模型的未知信息处理，增强了方法的实用性；数据集中FineWeb作为通用领域代表，Knowledge-Pile和Pile-of-Law作为领域特定数据集，具有代表性。
*   **局限性:** 实验模型规模较小（最大1.7B），未涉及主流大模型（如7B、70B），对更大模型的适用性需进一步验证；此外，方法基于经验分析，缺乏严格理论推导，可能影响某些场景下的解释力。

## Further Thoughts

论文提出的损失潜力（loss potential）概念非常具有启发性，提示我们可以在预训练阶段设计特定的学习率调度策略，预留更多损失下降空间以适应下游任务；此外，分布偏移与模型大小无关的发现启发我们未来可以通过度量数据集间的分布距离来预测CPT效果，而线性组合预测域外损失的方法则提示我们可以探索基于领域相似性的多域损失建模，或许结合深度学习方法（如神经网络）实现更精确的非线性映射。