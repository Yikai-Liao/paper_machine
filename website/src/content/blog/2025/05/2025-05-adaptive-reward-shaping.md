---
title: "Learn to Reason Efficiently with Adaptive Length-based Reward Shaping"
pubDatetime: 2025-05-21T15:03:26+00:00
slug: "2025-05-adaptive-reward-shaping"
type: "arxiv"
id: "2505.15612"
score: 0.7722057388282769
author: "grok-3-latest"
authors: ["Wei Liu", "Ruochen Zhou", "Yiyun Deng", "Yuzhen Huang", "Junteng Liu", "Yuntian Deng", "Yizhe Zhang", "Junxian He"]
tags: ["LLM", "Reasoning", "Reinforcement Learning", "Efficiency", "Reward Shaping"]
institution: ["The Hong Kong University of Science and Technology", "City University of Hong Kong", "University of Waterloo", "Apple"]
description: "本文提出 LASER 系列方法，通过动态、难度感知的长度奖励塑造，显著提升大型推理模型的性能与效率，并在多个基准测试中实现 Pareto 最优。"
---

> **Summary:** 本文提出 LASER 系列方法，通过动态、难度感知的长度奖励塑造，显著提升大型推理模型的性能与效率，并在多个基准测试中实现 Pareto 最优。 

> **Keywords:** LLM, Reasoning, Reinforcement Learning, Efficiency, Reward Shaping

**Authors:** Wei Liu, Ruochen Zhou, Yiyun Deng, Yuzhen Huang, Junteng Liu, Yuntian Deng, Yizhe Zhang, Junxian He

**Institution(s):** The Hong Kong University of Science and Technology, City University of Hong Kong, University of Waterloo, Apple


## Problem Background

大型推理模型（Large Reasoning Models, LRMs）通过生成长推理轨迹（Chains of Thought, CoT）展现出强大的问题解决能力，但这种长输出往往伴随冗余和‘过度思考’（over-thinking）问题，导致 token 使用量激增和潜在错误累积。
论文旨在解决如何在保持推理性能的同时显著提高效率（减少 token 使用量）这一关键挑战。

## Method

*   **统一框架：** 提出基于强化学习（Reinforcement Learning, RL）的长度基础奖励塑造（Length-based Reward Shaping）框架，将多种高效推理方法统一为奖励设计问题，通过结合正确性奖励和长度奖励优化模型输出。
*   **LASER（Length-based Step Reward）：** 设计一个阶梯奖励函数，对正确且短于目标长度的回答给予额外奖励，避免硬性截断导致的性能下降，同时设置较大的上下文窗口以减少截断发生。
*   **LASER-D（Dynamic and Difficulty-aware）：** 扩展 LASER，通过动态调整目标长度并根据问题难度分配不同 token 限制（简单问题少 token，复杂问题多 token），利用自动适应机制（Automatic Adapting Mechanism）基于实时训练数据评估难度并调整奖励。
*   **LASER-DE（Exploration-enhanced）：** LASER-D 的变体，对错误回答鼓励进一步探索，允许更长的推理以寻找正确模式，通过对错误回答施加较轻的长度惩罚实现。
*   **实现细节：** 使用 GRPO 优化目标，结合 KL 约束确保模型稳定性，奖励设计中平衡正确性与长度因素（如设置奖励系数 α=0.5）。

## Experiment

*   **有效性：** 在 DeepSeek-R1-Distill-Qwen 系列模型（1.5B、7B、32B）上，LASER 系列方法显著提升性能与效率，例如在 AIME2024 数据集上，LASER-D 和 LASER-DE 准确率提升 +6.1%，token 使用量减少 63%。
*   **对比性：** 与基线方法（如截断、组奖励、预算奖励）相比，LASER-D 和 LASER-DE 在 Pareto 边界上表现最佳，尤其在 token 预算较紧或问题较难时优势明显。
*   **泛化性：** 在域外数据集（GPQA, LSAT, MMLU）上，方法同样展现出准确率和效率的双重提升，证明跨领域适用性。
*   **合理性与局限：** 实验设置全面，覆盖多种模型规模和任务难度，参数调整充分探索；但计算成本较高，尤其在较大模型上的实验受限，可能影响结果的全面性。

## Further Thoughts

动态与难度感知的奖励设计是一个亮点，可以启发其他 RL 任务中自适应策略的开发，例如根据任务复杂度调整代码生成或多模态推理的资源分配；此外，LASER-DE 平衡探索与效率的思想提示我们可以在推理策略中结合‘快思考’和‘慢思考’，未来或许可以通过用户交互或上下文信息进一步优化难度评估机制，甚至在多任务场景下设计跨任务的奖励共享机制。