---
title: "Graceful Forgetting in Generative Language Models"
pubDatetime: 2025-05-26T09:03:57+00:00
slug: "2025-05-graceful-forgetting-llm"
type: "arxiv"
id: "2505.19715"
score: 0.8360227711922664
author: "grok-3-latest"
authors: ["Chunyang Jiang", "Chi-min Chan", "Yiyang Cai", "Yulong Liu", "Wei Xue", "Yike Guo"]
tags: ["LLM", "Graceful Forgetting", "Negative Transfer", "Fine-Tuning", "Learning Plasticity"]
institution: ["Hong Kong University of Science and Technology (HKUST)"]
description: "本文提出 Learning With Forgetting (LWF) 框架，通过自生成数据、遗忘置信度评估和周期性遗忘策略，首次系统性实现生成式语言模型中的优雅遗忘，显著提升微调性能。"
---

> **Summary:** 本文提出 Learning With Forgetting (LWF) 框架，通过自生成数据、遗忘置信度评估和周期性遗忘策略，首次系统性实现生成式语言模型中的优雅遗忘，显著提升微调性能。 

> **Keywords:** LLM, Graceful Forgetting, Negative Transfer, Fine-Tuning, Learning Plasticity

**Authors:** Chunyang Jiang, Chi-min Chan, Yiyang Cai, Yulong Liu, Wei Xue, Yike Guo

**Institution(s):** Hong Kong University of Science and Technology (HKUST)


## Problem Background

在生成式语言模型的预训练-微调范式中，部分预训练知识可能对下游任务产生负面影响（即负迁移，Negative Transfer），导致微调性能下降。
论文旨在解决这一问题，探索通过优雅遗忘（Graceful Forgetting）选择性地遗忘无关或有害知识，以提升微调效果。

## Method

*   **核心思想:** 提出 Learning With Forgetting (LWF) 框架，通过优雅遗忘机制在微调过程中选择性地移除与学习任务冲突的预训练知识，同时保持模型的学习可塑性。
*   **具体实现:**
    *   **自知识提取（Eliciting Self-Knowledge）:** 由于预训练数据不可访问，LWF 利用生成式模型自身的生成能力，通过输入遗忘任务的提示（Prompts），生成相关文本作为遗忘知识的表示（自生成数据集 D_self）。
    *   **遗忘置信度评估（Evaluating Forgetting Confidence）:** 针对每个自生成数据点，基于 Fisher 信息矩阵（FIM）对参数更新的加权，计算遗忘置信度（Forgetting Confidence），以衡量该数据点与学习任务的冲突程度；置信度越高，数据点越可能被遗忘。
    *   **周期性遗忘（Periodically Unlearning）:** 从自生成数据中选择高置信度数据点组成遗忘子集（D_U），在微调过程中以固定间隔（N_u）周期性地执行遗忘操作；遗忘采用梯度上升（Gradient Ascent）方法，通过对遗忘数据损失取负值来削弱相关知识影响，同时与学习任务训练交替进行以稳定训练过程。
*   **关键点:** LWF 不直接修改模型架构，而是通过数据点级别的细粒度遗忘和周期性策略，平衡遗忘与学习的关系，适用于生成式语言模型的特性。

## Experiment

*   **有效性:** LWF 在多个领域特定问答任务（如 GSM8K, QASC, SST5, Dental, Psychol）上显著提升了微调性能，例如在 GSM8K 任务遗忘 Dental 数据时性能提升 10.4%；尤其在混合遗忘（Mixed Forgetting）设置下，所有任务均获稳定提升。
*   **合理性:** 实验设置全面，涵盖不同领域任务，并验证了方法在小模型（Llama3.2-1B）和大模型（Llama3-8B）上的适用性；大模型提升幅度略减，但仍有效。
*   **对比分析:** 相较传统结构化调控方法（如 BSS, SRS），LWF 表现更优，证明其更适合生成式语言模型。
*   **消融实验:** 验证了遗忘置信度和周期性遗忘策略的重要性；遗忘低置信度数据或采用提前/随机遗忘策略会导致性能波动或下降。
*   **局限性与副作用:** 遗忘任务性能通常下降，但复杂任务（如 Dental, Psychol）下降较小；可能出现表面遗忘问题（如格式差异导致非目标任务性能下降）。

## Further Thoughts

LWF 的自生成数据作为知识表示的思路启发我们，可以利用生成式模型的特性来表征和操作知识，未来可应用于隐私保护或模型对齐中移除敏感信息；此外，数据点级别的遗忘置信度评估为细粒度知识管理提供了新视角，或许可以通过结合因果分析或注意力机制进一步提升评估精度；周期性遗忘策略也提示遗忘与学习可以协同优化，可能在持续学习或多任务学习中动态调整知识保留与遗忘的平衡。