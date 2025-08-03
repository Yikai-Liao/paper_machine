---
title: "LENS: Learning Ensemble Confidence from Neural States for Multi-LLM Answer Integration"
pubDatetime: 2025-07-31T00:35:45+00:00
slug: "2025-07-lens-ensemble-confidence"
type: "arxiv"
id: "2507.23167"
score: 0.6761361480838388
author: "grok-3-latest"
authors: ["Jizhou Guo"]
tags: ["LLM", "Ensemble Learning", "Confidence Estimation", "Internal Representation", "Model Integration"]
institution: ["Shanghai Jiao Tong University"]
description: "本文提出 LENS 方法，通过学习大型语言模型内部表示的置信度，显著提升多模型集成在问答任务中的性能，同时保持高效性和通用性。"
---

> **Summary:** 本文提出 LENS 方法，通过学习大型语言模型内部表示的置信度，显著提升多模型集成在问答任务中的性能，同时保持高效性和通用性。 

> **Keywords:** LLM, Ensemble Learning, Confidence Estimation, Internal Representation, Model Integration

**Authors:** Jizhou Guo

**Institution(s):** Shanghai Jiao Tong University


## Problem Background

大型语言模型（LLMs）在不同任务和领域中表现出不同的优势和局限性，单一模型难以应对复杂任务的多样化需求。
传统的集成方法（如多数投票或概率平均）未能考虑模型在不同上下文中的置信度和可靠性差异，导致性能提升有限。
本文旨在通过学习模型的上下文依赖置信度，解决多模型预测集成中的信息损失问题，提升系统鲁棒性和准确性。

## Method

*   **核心思想:** 提出 LENS（Learning Ensemble Confidence from Neural States），通过分析每个 LLM 的内部表示（hidden states）来学习模型置信度，并基于置信度加权集成预测结果。
*   **具体步骤:**
    *   **内部表示提取:** 利用 logit lens 技术，从每个模型的各层隐藏状态中提取特征，计算每一层的归一化概率（softmax），并将所有层的概率特征拼接为一个特征向量。
    *   **置信度预测器训练:** 为每个模型训练一个轻量级线性置信度预测器，输入为提取的特征向量，输出为置信度分数（通过 sigmoid 函数），使用二元交叉熵损失在验证集上优化，判断模型预测是否正确。
    *   **集成决策:** 在推理阶段，采用 Max Confidence 策略，选择置信度最高的模型预测作为最终结果。
*   **优势:** 不需要修改原始模型参数，计算开销小（仅增加线性预测器的训练和推理成本），方法通用，可应用于任何预训练模型组合。

## Experiment

*   **有效性:** LENS 的 Max Confidence 策略在 6 个问答数据集中的 5 个上显著优于传统集成方法（如多数投票和概率最大），例如在 BoolQ 数据集上准确率从 80.9% 提升至 84.1%，在 SWAG 上从 57.3% 提升至 58.8%。
*   **实验设置:** 实验涵盖多选题和布尔问答任务，使用 5 个不同架构的 LLM（如 LLaMA-2-7B、Mistral-7B），数据集包括 CoinFlip、BoolQ 等，每个数据集随机抽取 500 个样本，分为训练和测试集，设置较为全面合理。
*   **局限性:** 样本量较小（每个数据集仅 500 实例），可能影响置信度预测器的泛化能力；未提供详细计算开销数据，仅声称开销可忽略；未探讨不同模型规模对方法效果的影响。

## Further Thoughts

LENS 启发我们，模型的内部表示不仅用于预测，还蕴含了关于置信度和可靠性的宝贵信息，这种思路可扩展至模型解释性研究或动态模型选择。
此外，置信度预测器的跨任务迁移性值得探索，例如是否能将在一个任务上训练的置信度预测器应用于其他任务，实现零样本集成。
另一个有趣方向是将置信度估计与多智能体系统结合，动态分配任务权重，或与强化学习结合优化集成策略。