---
title: "On the Robustness of Reward Models for Language Model Alignment"
pubDatetime: 2025-05-12T06:48:26+00:00
slug: "2025-05-reward-model-robustness"
type: "arxiv"
id: "2505.07271"
score: 0.6328226353324459
author: "grok-3-latest"
authors: ["Jiwoo Hong", "Noah Lee", "Eunki Kim", "Woojin Chung", "Guijin Son", "Aman Gupta", "Shao Tang", "James Thorne"]
tags: ["LLM", "Reward Model", "RLHF", "Over-Optimization", "Regularization"]
institution: ["KAIST AI", "OneLineAI", "LinkedIn Corporation"]
description: "本文揭示了奖励模型过优化的根源在于隐藏状态范数分散，并提出批次和为零正则化（BSR）方法，显著提升了奖励模型的分布鲁棒性和 RLHF 对齐效果。"
---

> **Summary:** 本文揭示了奖励模型过优化的根源在于隐藏状态范数分散，并提出批次和为零正则化（BSR）方法，显著提升了奖励模型的分布鲁棒性和 RLHF 对齐效果。 

> **Keywords:** LLM, Reward Model, RLHF, Over-Optimization, Regularization

**Authors:** Jiwoo Hong, Noah Lee, Eunki Kim, Woojin Chung, Guijin Son, Aman Gupta, Shao Tang, James Thorne

**Institution(s):** KAIST AI, OneLineAI, LinkedIn Corporation


## Problem Background

奖励模型（Reward Models, RMs）是强化学习与人类反馈（RLHF）中用于对齐大型语言模型（LLMs）与人类偏好的关键组件，通常基于 Bradley-Terry (BT) 模型训练。然而，BT 模型训练的奖励模型容易过拟合训练数据，导致在未见输入分布上的泛化能力下降，即过优化问题。这种过优化不仅影响奖励模型的准确性，还会向下游 RLHF 过程传播，削弱语言模型的对齐效果。论文指出，过优化的主要原因是隐藏状态范数（hidden state norms）的过度分散，导致奖励分数异常放大，引发过自信问题。

## Method

*   **核心思想:** 提出批次和为零正则化（Batch-wise Sum-to-Zero Regularization, BSR），通过在每个批次内强制奖励总和接近零，约束奖励分数的极端值，减少隐藏状态范数的分散，从而缓解过优化问题。
*   **具体实现:** BSR 作为 BT 损失函数的一个附加正则化项，通过惩罚奖励分数的异常大值（即偏离零的程度），防止模型在训练过程中过度放大隐藏状态差异。BSR 的梯度设计对称地作用于正负奖励，限制隐藏状态差异的过度增长。
*   **评估框架:** 设计了四种泛化场景（In-Domain, Prompt-Disjoint, Response-Disjoint, Mutual-Disjoint），分别测试奖励模型在相同/不同提示和响应空间下的鲁棒性。
*   **对比基线:** 与标准 BT 模型及三种改进方法（BT-Hinge, BT-Norm, BT-DR）进行对比，验证 BSR 的优越性。
*   **优势:** BSR 实现简单，仅需一个超参数（正则化权重 λ），且不需修改模型架构或训练流程即可显著提升鲁棒性。

## Experiment

*   **有效性:** 在四种泛化场景中，BSR 训练的奖励模型（RM BT-BSR）在未见数据上的表现（以 Kendall’s τ 和准确率衡量）均优于标准 BT 模型和其他基线，尤其在 Response-Disjoint 和 Mutual-Disjoint 场景中，表明其对未见响应风格的鲁棒性更强。
*   **RLHF 传播效果:** 使用 RM BT-BSR 进行 RLOO 训练时，策略与金标准偏好模型（ArmoRM）的对齐效果更好，奖励最大化过程更稳定，KL 散度变化平滑。
*   **真实世界影响:** 在 8B 规模模型和高质量数据集（如 Skywork-Reward-Preference-80K）上的实验显示，RM BT-BSR 在 RM-Bench 的‘Hard Acc’任务中提升了约 5-7%，在 AlpacaEval 2.0 上减少了 40% 的生成长度，同时提升了 7% 的胜率，缓解了冗长偏见问题。
*   **实验设置合理性:** 实验涵盖多种模型规模（1B 到 8B）、不同模型家族（Llama-3, Qwen2.5）及多种数据集，验证了方法的普适性；同时通过有效秩（effective rank）分析进一步确认了 BSR 对过优化的缓解作用。

## Further Thoughts

BSR 的正则化思路启发我们关注模型内部表示的稳定性，而不仅仅是预测准确性。未来可以探索动态调整正则化强度（例如根据训练阶段或数据分布特性调整 λ）以进一步优化效果；此外，是否可以对隐藏状态直接施加正则化，或设计跨批次的约束机制？BSR 与其他过优化缓解方法（如奖励模型集成或数据增强）的结合也可能带来更强的鲁棒性。