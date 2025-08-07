---
title: "RCP-Merging: Merging Long Chain-of-Thought Models with Domain-Specific Models by Considering Reasoning Capability as Prior"
pubDatetime: 2025-08-05T06:38:18+00:00
slug: "2025-08-rcp-merging-reasoning"
type: "arxiv"
id: "2508.03140"
score: 0.8771758876992852
author: "grok-3-latest"
authors: ["Junyao Yang", "Jianwei Wang", "Huiping Zhuang", "Cen Chen", "Ziqian Zeng"]
tags: ["LLM", "Model Merging", "Reasoning", "Domain Knowledge", "Task Balancing"]
institution: ["South China University of Technology, China"]
description: "本文提出 RCP-Merging 框架，通过将推理能力作为先验并结合 Fisher 信息矩阵保护机制，成功融合长链式推理模型与领域特定模型，在提升领域任务表现的同时保持推理能力。"
---

> **Summary:** 本文提出 RCP-Merging 框架，通过将推理能力作为先验并结合 Fisher 信息矩阵保护机制，成功融合长链式推理模型与领域特定模型，在提升领域任务表现的同时保持推理能力。 

> **Keywords:** LLM, Model Merging, Reasoning, Domain Knowledge, Task Balancing

**Authors:** Junyao Yang, Jianwei Wang, Huiping Zhuang, Cen Chen, Ziqian Zeng

**Institution(s):** South China University of Technology, China


## Problem Background

大型语言模型（LLMs）在长链式推理（Long Chain-of-Thought, CoT）任务中表现出色，但往往在特定领域（如生物医学和金融）表现较弱，而领域特定模型虽在专业任务上表现优异，却缺乏复杂的多步推理能力；现有的模型融合方法在尝试整合推理模型与领域特定模型时，常导致推理能力退化、输出崩溃或生成无意义内容，因此亟需一种方法在提升领域任务表现的同时保护推理能力。

## Method

* **核心思想**：提出 RCP-Merging 框架，通过将推理能力作为先验（Prior），在模型融合过程中保护长链式推理能力，同时有选择地融入领域特定知识，确保融合模型在两个方面的性能平衡。
* **具体实现**：方法分为三个阶段：
  * **Domain Knowledge Sensitivity**：通过计算每个参数对领域特定任务损失的影响（使用一阶泰勒展开近似），量化其对领域任务的重要性，识别关键的领域特定权重。
  * **Reasoning Preservation Indicator**：基于贝叶斯规则和 Fisher 信息矩阵（FIM），构建推理能力保护指标，衡量参数偏离推理最优值的影响，防止融合过程中推理能力的灾难性遗忘。
  * **Reasoning-preserved Merging**：结合领域敏感度和推理保护指标，定义约束度量（Constraint Metric），通过多数投票机制决定是否接受参数更新，最终生成融合模型，确保领域知识和推理能力的平衡。
* **特点**：方法无需额外训练，仅需少量校准数据即可完成融合，资源效率高；通过超参数（如推理保护系数 λ）调节领域性能和推理保护的权衡。

## Experiment

* **有效性**：RCP-Merging 在生物医学和金融领域任务上的表现分别比现有最优方法提升了 9.5% 和 9.2%，在推理任务（如 GSM8K 和 HumanEval）上也保持了较高性能，展现出显著的性能提升。
* **稳定性**：相比其他融合方法，RCP-Merging 的无意义输出率（gibberish rate）最低，仅为 14.3%，表明其输出质量更可靠，避免了输出崩溃问题。
* **全面性**：实验覆盖了多种模型架构（如 Qwen2.5-7B, Llama3.1-8B）和规模（如 1.5B 参数），以及多个领域（生物医学、金融）和数据集（数学、代码生成等），验证了方法的通用性和可扩展性。
* **合理性**：实验设置合理，通过对比多种基线方法（如 Task Arithmetic, TIES-Merging）并使用多种评估指标（如准确率、Pass@1、gibberish rate），全面验证了方法的优越性；但超参数选择对结果影响较大，需进一步优化。

## Further Thoughts

RCP-Merging 将推理能力作为先验的思路非常新颖，基于 Fisher 信息矩阵的保护机制可推广至其他多能力融合场景，如安全性和领域适应性的平衡；此外，资源高效的校准数据使用方式启发我们探索更小数据集或无监督方法降低成本；未来可考虑动态调整推理保护强度，或扩展至多模型融合以整合更广泛能力。