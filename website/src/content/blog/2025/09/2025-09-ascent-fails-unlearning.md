---
title: "Ascent Fails to Forget"
pubDatetime: 2025-09-30T15:48:49+00:00
slug: "2025-09-ascent-fails-unlearning"
type: "arxiv"
id: "2509.26427"
score: 0.705999812126616
author: "grok-3-latest"
authors: ["Ioannis Mavrothalassitis", "Pol Puigdemont", "Noam Itzhak Levi", "Volkan Cevher"]
tags: ["Machine Unlearning", "Data Dependency", "Gradient Ascent", "Model Performance", "Privacy Protection"]
institution: ["École Polytechnique Fédérale de Lausanne (EPFL)"]
description: "本文通过理论和实验揭示基于梯度上升的机器遗忘方法因忽视遗忘集与保留集的统计依赖性而失效，为未来遗忘算法设计提供了关键指导。"
---

> **Summary:** 本文通过理论和实验揭示基于梯度上升的机器遗忘方法因忽视遗忘集与保留集的统计依赖性而失效，为未来遗忘算法设计提供了关键指导。 

> **Keywords:** Machine Unlearning, Data Dependency, Gradient Ascent, Model Performance, Privacy Protection

**Authors:** Ioannis Mavrothalassitis, Pol Puigdemont, Noam Itzhak Levi, Volkan Cevher

**Institution(s):** École Polytechnique Fédérale de Lausanne (EPFL)


## Problem Background

机器遗忘（Machine Unlearning）是机器学习领域的一个关键问题，旨在从已训练模型中移除特定训练数据的影响，以应对数据隐私、模型维护及有毒数据等问题。
论文指出，当前基于梯度上升的无约束优化方法（如 Descent-Ascent, DA）常因忽视遗忘集（Forget Set）和保留集（Retain Set）之间的统计依赖性（甚至是简单相关性）而失败，导致遗忘过程不仅无效，还可能损害模型整体性能。

## Method

*   **核心思想:** 揭示基于梯度上升的遗忘方法（如 Gradient Ascent, GA 和 Gradient Descent/Ascent, GDA）在存在数据依赖性时的失效机制。
*   **理论分析:** 
    *   在随机遗忘集场景下，通过概率论证明遗忘集指标与测试集指标高度相关，表明通过梯度上升降低遗忘集性能必然损害整体性能。
    *   在逻辑回归场景下，推导出闭式解，证明 DA 方法的更新方向与理想解（Oracle）背道而驰，尤其在遗忘集与保留集相关性高时，模型会偏离最优解甚至比预训练模型更差。
    *   在低维非线性示例中，展示 DA 方法如何陷入次优局部极小值，且后续微调难以挽救。
*   **实验验证:** 
    *   使用复杂神经网络（如 ResNet-9）在 CIFAR-10 数据集上验证理论结论，评估遗忘效果（KLoM 指标）。
    *   对比不同遗忘集（如随机集与主成分集）的影响，揭示数据依赖性对遗忘性能的破坏作用。
*   **关键点:** 不修改模型架构或训练过程，而是通过理论和实验揭示现有方法的根本缺陷，强调数据依赖性是遗忘失败的核心原因。

## Experiment

*   **有效性:** 实验表明 GA 和 GDA 方法在大多数情况下未能实现有效遗忘，尤其在遗忘集与保留集高度相关时（如选择主成分方向的遗忘集），要么无法显著偏离预训练模型，要么严重损害模型性能（KLoM 指标显示遗忘集与测试集分布差异大）。
*   **对比分析:** 相比随机遗忘集，结构化相关遗忘集（如主成分集）导致的性能下降更严重，验证了数据依赖性对遗忘的影响。
*   **实验设置合理性:** 实验覆盖了多种遗忘集大小（0.02% 到 10%）和类型（随机 vs 结构化），并在多个数据集（CIFAR-10, MNIST, FashionMNIST）上测试，设置较为全面；但对多层网络复杂依赖性的探讨不足。
*   **局限性:** 方法对超参数（如学习率、停止准则）极为敏感，且缺乏明确的优化目标，实际应用中易产生误导性结果（如‘Ascent Forgets Illusion’）。

## Further Thoughts

论文揭示的数据依赖性对遗忘算法的影响是一个重要启发，未来可以探索通过正则化或噪声注入（如随机梯度噪声）缓解遗忘集与保留集的相关性问题；此外，是否可以通过设计依赖性感知的遗忘目标函数（如基于影响函数的跨集相关性矩阵）来改进遗忘效果，也是一个值得深入研究的方向。