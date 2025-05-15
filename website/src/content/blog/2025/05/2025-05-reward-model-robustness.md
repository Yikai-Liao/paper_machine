---
title: "On the Robustness of Reward Models for Language Model Alignment"
pubDatetime: 2025-05-12T06:48:26+00:00
slug: "2025-05-reward-model-robustness"
type: "arxiv"
id: "2505.07271"
score: 0.6338849657486068
author: "grok-3-latest"
authors: ["Jiwoo Hong", "Noah Lee", "Eunki Kim", "Guijin Son", "Woojin Chung", "Aman Gupta", "Shao Tang", "James Thorne"]
tags: ["LLM", "Reward Model", "RLHF", "Regularization", "Over-Optimization"]
institution: ["KAIST AI", "OneLineAI", "LinkedIn Corporation"]
description: "本文提出批次和为零正则化（BSR）方法，通过约束奖励总和为零缓解隐藏状态范数分散导致的过优化问题，显著提升奖励模型在 RLHF 中的鲁棒性和下游任务表现。"
---

> **Summary:** 本文提出批次和为零正则化（BSR）方法，通过约束奖励总和为零缓解隐藏状态范数分散导致的过优化问题，显著提升奖励模型在 RLHF 中的鲁棒性和下游任务表现。 

> **Keywords:** LLM, Reward Model, RLHF, Regularization, Over-Optimization

**Authors:** Jiwoo Hong, Noah Lee, Eunki Kim, Guijin Son, Woojin Chung, Aman Gupta, Shao Tang, James Thorne

**Institution(s):** KAIST AI, OneLineAI, LinkedIn Corporation


## Problem Background

奖励模型（Reward Models, RMs）在通过人类反馈进行强化学习（RLHF）对大型语言模型（LLMs）进行对齐时，扮演着关键角色，但其训练过程中常出现过优化（over-optimization）问题，导致在未见数据分布上的泛化能力下降，无法准确反映真实人类偏好，进而影响下游 RLHF 效果。
论文指出，过优化的主要原因是隐藏状态范数（hidden state norms）的过度分散，导致奖励分数异常极端，削弱了模型对未见输入的适应性。

## Method

*   **核心思想:** 提出批次和为零正则化（Batch-wise Sum-to-Zero Regularization, BSR），通过约束每个批次内奖励总和为零，控制隐藏状态范数的过度分散，从而缓解过优化问题，提升奖励模型的鲁棒性。
*   **具体实现:** 
    *   BSR 作为一种附加正则化项，添加到传统的 Bradley-Terry (BT) 模型损失函数中，形成新的目标函数 L_BT-BSR。
    *   在训练过程中，BSR 惩罚奖励分数的极端值（即异常大的正值或负值），通过梯度调整限制隐藏状态范数的增长，防止奖励模型对训练数据过拟合。
    *   实现上，BSR 不需要修改模型架构，仅通过调整损失函数即可完成，计算开销小，易于集成到现有训练流程中。
*   **参数调整:** BSR 的正则化强度通过超参数 λ 控制，实验中测试了不同 λ 值（如 10^-2, 10^-3, 10^-4），以平衡鲁棒性和模型性能。
*   **优势:** BSR 是一种轻量级方法，理论上基于隐藏状态范数与过优化的关系，实践上无需额外复杂计算即可显著提升泛化能力。

## Experiment

*   **有效性:** 实验分为三部分，全面验证了 BSR 的效果：
    *   在奖励模型过优化实验中，BSR 在四种泛化场景（In-Domain, Prompt-Disjoint, Response-Disjoint, Mutual-Disjoint）中均优于基准方法（如 BT, BT-Hinge），尤其在未见响应风格（Response OOD 和 Mutual OOD）上，Kendall’s τ 指标显著提升（如 Qwen2.5-3B 模型上 Mutual OOD 从 0.587 提升到 0.6106）。
    *   在 RLHF 鲁棒性传播实验中，使用 BSR 训练的奖励模型在 RLOO 训练中表现出更稳定的奖励最大化和更好的策略对齐，与金标准偏好模型的一致性更高（图 5）。
    *   在现实世界影响实验中，BSR 在 8B 规模模型上提升了复杂偏好预测任务的准确率（Hard Acc 提升超过 5%），并在 AlpacaEval 2.0 上减少生成长度 40%（从 2247 到 1337），同时胜率提升 7%（从 2.59% 到 9.02%）。
*   **实验设置合理性:** 实验覆盖了多种模型规模（1B 到 8B）、不同模型家族（Llama-3, Qwen2.5）以及多样化数据集（UltraFeedback, Skywork-Reward-Preference），并通过四种泛化场景细致评估了鲁棒性，设置较为全面。
*   **局限性:** 实验主要聚焦于英语数据集和通用任务，未涉及多语言或特定领域（如数学推理）的泛化性测试；此外，BSR 效果在更大模型上更显著，可能对小模型的收益有限。

## Further Thoughts

BSR 通过正则化隐藏状态范数来提升奖励模型鲁棒性的思路具有启发性，提示我们可以在其他代理模型训练中探索类似约束机制，例如在图像生成或推荐系统的评分模型中引入类似‘和为零’的正则化；此外，奖励模型鲁棒性对 RLHF 下游任务的影响显著，启发我们在设计对齐算法时应优先优化代理模型的泛化能力，而不仅仅是训练集准确率；另一个值得探索的方向是，是否可以通过直接对隐藏状态进行归一化或引入对抗性训练来替代 BSR，进一步提升效果或降低计算成本。