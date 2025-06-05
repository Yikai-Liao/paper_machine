---
title: "Understanding Overadaptation in Supervised Fine-Tuning: The Role of Ensemble Methods"
pubDatetime: 2025-06-02T17:23:16+00:00
slug: "2025-06-overadaptation-ensemble"
type: "arxiv"
id: "2506.01901"
score: 0.6961584853992734
author: "grok-3-latest"
authors: ["Yifan Hao", "Xingyuan Pan", "Hanning Zhang", "Chenlu Ye", "Rui Pan", "Tong Zhang"]
tags: ["LLM", "Ensemble Methods", "Fine-Tuning", "Overadaptation", "Bias-Variance Tradeoff"]
institution: ["University of Illinois Urbana-Champaign"]
description: "本文通过理论分析和实验验证，揭示了模型集成通过平衡‘偏差-方差’权衡，有效缓解监督微调中的过适应问题，提升下游任务性能并减少预训练知识遗忘。"
---

> **Summary:** 本文通过理论分析和实验验证，揭示了模型集成通过平衡‘偏差-方差’权衡，有效缓解监督微调中的过适应问题，提升下游任务性能并减少预训练知识遗忘。 

> **Keywords:** LLM, Ensemble Methods, Fine-Tuning, Overadaptation, Bias-Variance Tradeoff

**Authors:** Yifan Hao, Xingyuan Pan, Hanning Zhang, Chenlu Ye, Rui Pan, Tong Zhang

**Institution(s):** University of Illinois Urbana-Champaign


## Problem Background

监督微调（Supervised Fine-Tuning, SFT）是大型语言模型（LLMs）适应特定下游任务的主流方法，但常导致模型过适应（Overadaptation），即过度专注于微调任务而遗忘预训练阶段的通用知识，损害泛化能力和上游任务表现。
论文旨在解决这一关键问题：探索为何模型集成（Ensembling）能有效缓解过适应，同时提升下游任务性能并减少遗忘，尤其是在过参数化模型背景下缺乏理论解释的情况下。

## Method

*   **核心思想:** 通过对预训练模型和微调模型的权重进行加权平均（Weighted Averaging），构建集成模型，以平衡微调任务性能和预训练知识保留。
*   **理论分析:** 在过参数化的线性回归框架下，作者证明集成方法通过优化‘偏差（Bias）’和‘方差（Variance）’的权衡，减少测试误差。预训练模型偏向高偏差（缺乏任务特定信息），而无正则化的微调模型偏向高方差（过拟合噪声数据）；集成方法通过加权参数 *τ*（0 ≤ τ ≤ 1）在两者间找到更优平衡点。
*   **实验实现:** 在多个开源大型语言模型（如 Llama-3-8B, Qwen2-7B, Gemma-2-9B）上，结合不同正则化策略（如 Norm-Penalty, DiffNorm-Penalty）进行微调，并通过加权平均生成集成模型，测试其在指令跟随任务和通用能力上的表现。
*   **关键细节:** 集成方法不修改模型结构，仅在权重层面操作，计算开销低；同时，作者探索了不同 *τ* 值的影响，确保方法在多种设置下的鲁棒性。

## Experiment

*   **有效性:** 实验结果表明，集成方法在微调任务（如 MT-Bench）上显著优于单独微调模型，例如 Llama-3-8B 的 MT-Bench 评分从 5.68 提升至 5.96，同时在预训练任务（如 MMLU, Commonsense-QA）上减少了遗忘，表现出更好的性能权衡。
*   **全面性:** 实验覆盖多个数据集（Dolly, MT-Bench, MMLU, Commonsense-QA）和模型，测试了不同正则化与集成组合的效果，并扩展至 LoRA 微调方法，验证了方法的普适性。
*   **合理性:** 实验设计合理，包含超参数搜索（如学习率、惩罚系数、集成权重 *τ*）和多轮试验（标准差仅 0.06），结果稳定且可信；此外，作者通过模拟数据进一步验证了理论分析的预测。
*   **开销:** 集成方法仅涉及权重平均，计算成本低，易于实际应用。

## Further Thoughts

模型集成不仅限于预训练与微调模型的平均，可以探索多个微调模型的集成以进一步提升任务特定性能和泛化能力；此外，结合参数高效微调（如 LoRA），设计基于任务特性的动态加权策略（自适应调整 *τ*）可能带来更大收益；最后，如何将线性框架的‘偏差-方差’分析推广至非线性神经网络的复杂动态（如注意力机制）是一个值得深入研究的方向。