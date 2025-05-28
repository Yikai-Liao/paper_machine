---
title: "Graceful Forgetting in Generative Language Models"
pubDatetime: 2025-05-26T09:03:57+00:00
slug: "2025-05-graceful-forgetting-lwf"
type: "arxiv"
id: "2505.19715"
score: 0.8360227711922664
author: "grok-3-latest"
authors: ["Chunyang Jiang", "Chi-min Chan", "Yiyang Cai", "Yulong Liu", "Wei Xue", "Yike Guo"]
tags: ["Generative Models", "Negative Transfer", "Fine-Tuning", "Forgetting", "Knowledge Selection"]
institution: ["Hong Kong University of Science and Technology (HKUST)"]
description: "本文提出 LWF 框架，通过自生成数据、遗忘置信度评估和周期性遗忘，在生成式语言模型微调中实现优雅遗忘，有效缓解负迁移并提升性能。"
---

> **Summary:** 本文提出 LWF 框架，通过自生成数据、遗忘置信度评估和周期性遗忘，在生成式语言模型微调中实现优雅遗忘，有效缓解负迁移并提升性能。 

> **Keywords:** Generative Models, Negative Transfer, Fine-Tuning, Forgetting, Knowledge Selection

**Authors:** Chunyang Jiang, Chi-min Chan, Yiyang Cai, Yulong Liu, Wei Xue, Yike Guo

**Institution(s):** Hong Kong University of Science and Technology (HKUST)


## Problem Background

生成式语言模型在预训练-微调范式下常面临负迁移（Negative Transfer）问题，即预训练中获得的某些知识对下游微调任务产生不利影响，导致性能下降。
论文指出，传统微调方法无法区分预训练知识的有益与有害部分，而现有优雅遗忘（Graceful Forgetting）方法多针对视觉任务或非自回归模型，难以直接应用于生成式语言模型，因其知识边界模糊，任务间相关性难以量化。
因此，本研究旨在探索如何通过选择性遗忘无关或有害知识，增强生成式语言模型在微调任务中的学习塑性（Learning Plasticity）。

## Method

*   **核心思想：** 提出 Learning With Forgetting (LWF) 框架，通过优雅遗忘机制在微调过程中选择性地丢弃与目标任务冲突的预训练知识，以缓解负迁移问题并提升性能。
*   **具体步骤：**
    *   **自知识提取（Eliciting Self-Knowledge）：** 针对预训练数据不可访问的问题，利用生成式模型自身的生成能力，将遗忘任务（Forgetting Task）的提示输入基础模型，收集其生成的响应文本，形成自生成数据集（D_self），作为遗忘知识的代理表示。
    *   **遗忘置信度评估（Evaluating Forgetting Confidence）：** 为每个自生成数据点计算遗忘置信度（FC），以判断其与学习任务（Learning Task）的冲突程度；具体方法是基于 Fisher 信息矩阵（FIM）对参数更新的重要性进行加权，通过近似计算数据点诱导的参数更新与目标任务最优参数的偏差，量化冲突大小，置信度越高表示越应遗忘。
    *   **周期性遗忘（Periodically Unlearning）：** 根据遗忘置信度选择高置信度数据点组成遗忘子集（D_U），在微调过程中周期性地对这些数据执行机器遗忘，采用梯度上升（Gradient Ascent）方法反向优化损失函数，削弱模型对这些知识的记忆；遗忘与学习交替进行，以固定间隔（N_u）执行，确保训练过程稳定。
*   **特点：** LWF 不修改模型架构，仅在微调阶段调整训练策略，适用于现有生成式语言模型，且通过细粒度的数据点级遗忘置信度评估，适应语言模型语义复杂性。

## Experiment

*   **有效性：** 在领域特定问答任务（QA）上，LWF 相较于 vanilla 微调显著提升性能，例如在 gsm8k 任务上遗忘 dental 数据时提升 10.4%，混合遗忘设置（Mixed Setting）表现最稳定，普遍带来 2%-7% 的性能提升。
*   **对比分析：** 相较传统结构化调控方法（如 BSS 和 SRS），LWF 在生成式语言模型上的表现更优，表明其针对性设计更适合语言模型的特性。
*   **遗忘置信度作用：** 对比实验显示，遗忘高置信度数据带来更高的平均性能提升和更小的方差，验证了置信度计算的有效性。
*   **周期性遗忘策略：** 对比提前遗忘和随机遗忘，周期性遗忘显著减少性能波动，避免了提前遗忘对基础知识的破坏，证明其在稳定性上的优势。
*   **扩展性：** 在更大模型（Llama3-8B）上，LWF 仍能提升性能，尽管提升幅度因基线更高而减小，表明方法具有一定的模型规模适应性。
*   **任务泛化性：** 在机器翻译和多语言 QA 任务上，LWF 效果不如 QA 任务明显，性能提升幅度较小，部分结果因评价指标限制而不一致，提示方法可能需要任务特定调整。
*   **实验设置合理性：** 实验覆盖多个数据集（gsm8k, qasc, sst5, dental, psychol）、不同模型规模（Llama3.2-1B 和 Llama3-8B）及任务类型，设置较为全面，但遗忘置信度的启发式计算和任务泛化性不足是潜在局限。

## Further Thoughts

自生成数据作为知识表示的思路启发我们在数据受限场景中利用模型生成能力构建代理数据集；遗忘置信度的细粒度评估为量化知识冲突提供了新视角，未来可结合注意力机制或语义相似度进一步优化；周期性遗忘策略平衡学习与遗忘的稳定性，其交替优化思想可推广至多任务学习或持续学习等需要权衡多目标的场景。