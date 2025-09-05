---
title: "mFARM: Towards Multi-Faceted Fairness Assessment based on HARMs in Clinical Decision Support"
pubDatetime: 2025-09-02T06:47:57+00:00
slug: "2025-09-mfarm-fairness-clinical"
type: "arxiv"
id: "2509.02007"
score: 0.49414337220967186
author: "grok-3-latest"
authors: ["Shreyash Adappanavar", "Krithi Shailya", "Gokul S Krishnan", "Sriraam Natarajan", "Balaraman Ravindran"]
tags: ["LLM", "Fairness Assessment", "Clinical Decision", "Bias Detection", "Model Alignment"]
institution: ["Indian Institute of Technology Madras", "University of Texas at Dallas"]
description: "本文提出 mFARM 框架，通过五个独立指标多维度评估临床 LLMs 的公平性，并结合 FAB Score 平衡公平性和准确性，为医疗 AI 对齐提供实用工具。"
---

> **Summary:** 本文提出 mFARM 框架，通过五个独立指标多维度评估临床 LLMs 的公平性，并结合 FAB Score 平衡公平性和准确性，为医疗 AI 对齐提供实用工具。 

> **Keywords:** LLM, Fairness Assessment, Clinical Decision, Bias Detection, Model Alignment

**Authors:** Shreyash Adappanavar, Krithi Shailya, Gokul S Krishnan, Sriraam Natarajan, Balaraman Ravindran

**Institution(s):** Indian Institute of Technology Madras, University of Texas at Dallas


## Problem Background

大型语言模型（LLMs）在高风险医疗场景中的应用面临严重的 AI 对齐挑战，因为模型可能继承并放大社会偏见，导致医疗结果的不平等，尤其是在种族和性别等维度上。
现有公平性评估方法依赖单一指标，忽视了医疗伤害的多维度特性，且模型可能通过输出‘安全’但不准确的建议掩盖偏见，造成公平性虚高但临床实用性低下的问题。

## Method

*   **核心框架：mFARM（Multi-faceted Fairness Assessment based on HARMs）**：提出一个多维度公平性评估框架，通过五个独立指标评估三种伤害类型：
    *   **分配伤害（Allocational Harm）**：通过均值差异（Mean Difference）检测模型预测概率的系统性偏移，确保资源分配不偏向特定群体。
    *   **稳定性伤害（Stability Harm）**：通过方差异质性（Variance Heterogeneity）和绝对偏差（Absolute Deviation）评估模型预测的一致性和稳定性，避免某些群体获得不稳定输出。
    *   **潜在伤害（Latent Harm）**：通过 Kolmogorov-Smirnov 分布公平性（KS Distributional Fairness）和相关性差异（Correlation Difference）检测分布形状差异和信心相关偏见，揭示隐藏的刻板印象。
*   **统计方法**：每个指标采用三阶段方法，包括全组检验（如 Friedman 检验、Levene 检验）、后验分析（如 Wilcoxon 检验）和公平性分数计算，确保统计显著性和效果量评估。
*   **公平性-准确性平衡（FAB Score）**：通过谐波均值结合准确性和 mFARM 分数，确保模型在公平性和临床实用性之间取得平衡。
*   **基准数据集**：基于 MIMIC-IV 数据库构建两个大规模任务数据集（ED-Triage 和 Opioid Analgesic Recommendation），包含超过 5 万个提示，覆盖 12 种种族×性别组合和三种上下文层次（高、中、低），以隔离人口统计学因素的影响。
*   **模型测试与优化**：评估四个开源 LLMs（Mistral-7B、BioMistral-7B、Qwen-2.5-7B、BioLlama3-8B），并通过 LoRA 微调提升性能，同时测试量化（16-bit、8-bit、4-bit）和上下文变化的影响。

## Experiment

*   **有效性**：mFARM 框架在捕捉细微偏见方面优于传统指标（如统计均等和机会均等），例如在 ED-Triage 任务中，Qwen-2.5 的相关性差异分数为 0.36，揭示了隐藏的分布不均，而传统指标未能发现。
*   **性能提升**：LoRA 微调显著提升准确性和 FAB 分数，例如 BioLlama 在 ED-Triage 任务上的准确性从 0.492 提升至 0.738，FAB 分数从 0.623 提升至 0.804，表明微调增强了临床实用性。
*   **上下文敏感性**：上下文减少时公平性和 FAB 分数显著下降，例如 Qwen 在低上下文下的 mFARM 分数降至 0，表明信息量对公平性至关重要。
*   **量化鲁棒性**：降低量化精度（如 4-bit）通常不损害公平性，有时甚至提升，例如 BioLlama 在 Opioid 任务上的 mFARM 分数从 16-bit 的 0.674 提升至 4-bit 的 0.956，可能是量化扰动减少了刻板印象。
*   **实验全面性**：实验覆盖多个模型、任务、上下文层次和量化级别，确保结果鲁棒，但任务范围有限（仅两个任务），可能无法完全代表所有医疗场景。

## Further Thoughts

mFARM 的多维度评估思路可推广至其他高风险领域（如金融、司法），通过定制伤害类型和指标揭示隐藏偏见；上下文敏感性实验启发动态上下文补充机制的研究，以提升低上下文场景下的公平性；量化对公平性的正向影响提示可通过受控噪声或正则化减少模型对敏感属性的依赖；FAB Score 可作为训练损失函数，直接优化公平性-准确性权衡。