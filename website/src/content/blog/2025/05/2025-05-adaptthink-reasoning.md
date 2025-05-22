---
title: "AdaptThink: Reasoning Models Can Learn When to Think"
pubDatetime: 2025-05-19T17:50:52+00:00
slug: "2025-05-adaptthink-reasoning"
type: "arxiv"
id: "2505.13417"
score: 0.8689223945654835
author: "grok-3-latest"
authors: ["Jiajie Zhang", "Nianyi Lin", "Lei Hou", "Ling Feng", "Juanzi Li"]
tags: ["LLM", "Reasoning", "Sampling", "RLHF", "Efficiency"]
institution: ["Tsinghua University"]
description: "本文提出 *AdaptThink*，一种基于强化学习的算法，使推理模型根据问题难度自适应选择思考模式，显著降低推理成本并提升性能。"
---

> **Summary:** 本文提出 *AdaptThink*，一种基于强化学习的算法，使推理模型根据问题难度自适应选择思考模式，显著降低推理成本并提升性能。 

> **Keywords:** LLM, Reasoning, Sampling, RLHF, Efficiency

**Authors:** Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, Juanzi Li

**Institution(s):** Tsinghua University


## Problem Background

大型推理模型（Large Reasoning Models, LRMs）通过生成详细的思维链（Chain of Thought）在复杂任务上表现出色，但冗长的思考过程显著增加了推理成本，尤其在简单任务上显得低效。
作者发现，对于相对简单的任务，跳过思考过程（NoThinking）直接生成答案不仅更高效，甚至在性能上可能优于传统思考模式（Thinking），因此提出一个核心问题：能否让模型根据问题难度自适应地选择思考模式，以在性能和效率之间取得平衡。

## Method

*   **核心思想:** 通过强化学习（RL）训练推理模型，使其根据问题难度自适应地选择 *Thinking*（深入思考）或 *NoThinking*（直接生成答案）模式，以优化推理效率和性能。
*   **具体实现:** 
    *   **约束优化目标:** 设计一个优化目标，鼓励模型优先选择 *NoThinking* 模式以提升效率，同时通过奖励函数（基于准确率）确保整体性能不下降。引入一个可调参数 *δ* 控制效率与准确性的权衡，并使用 PPO（Proximal Policy Optimization）算法通过策略梯度方法优化目标。
    *   **重要性采样策略:** 针对初始模型倾向于总是选择 *Thinking* 模式的冷启动问题，设计一种重要性采样策略，在训练时强制平衡 *Thinking* 和 *NoThinking* 样本的比例，确保模型从两种模式中学习，并持续探索和利用两种模式的优势。
*   **关键特点:** 不修改模型架构或预训练数据，仅通过训练策略调整实现自适应选择，具有较强的通用性，可应用于现有推理模型。

## Experiment

*   **有效性:** 在 GSM8K、MATH500 和 AIME 2024 数据集上，*AdaptThink* 显著降低了响应长度（1.5B 模型平均减少 53%，7B 模型减少 40.1%），同时提升了准确率（1.5B 模型提升 2.4%，7B 模型提升 2.3%），证明了方法在效率和性能上的双重优势。
*   **自适应性:** 模型在简单任务（如 GSM8K）上更多选择 *NoThinking* 模式，在困难任务（如 AIME 2024）上更多选择 *Thinking* 模式，体现了根据问题难度自适应选择的能力。
*   **对比优越性:** 与多种基线方法（如 DPO、OverThink、ModelMerging 等）相比，*AdaptThink* 在准确率和响应长度上均表现最佳，验证了自适应选择思考模式的优越性。
*   **实验设置合理性:** 实验覆盖了不同模型规模（1.5B 和 7B）和难度级别的数据集，评估指标（准确率和响应长度）直接对应研究目标；参数 *δ* 的影响分析展示了方法的鲁棒性；MMLU 数据集上的测试表明方法具有一定的泛化能力，但训练数据限于数学领域可能限制更广泛的适用性。

## Further Thoughts

论文通过强化学习实现推理策略自适应选择的思路非常具有启发性，是否可以进一步扩展到更多推理模式（如发散性思维或多步推理）或根据任务类型动态调整？此外，重要性采样策略在解决冷启动问题上的应用是否可以推广到其他强化学习场景中平衡探索和利用？另一个思考方向是，是否可以结合轻量级模型预测问题难度，辅助主模型决定推理深度，进一步降低计算成本。