---
title: "J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning"
pubDatetime: 2025-05-15T14:05:15+00:00
slug: "2025-05-thinking-llm-judge"
type: "arxiv"
id: "2505.10320"
score: 0.676100124779818
author: "grok-3-latest"
authors: ["Not Specified"]
tags: ["LLM", "Reinforcement Learning", "Reasoning", "Reward Model", "Synthetic Data", "Test Time Scaling"]
institution: ["Not Specified"]
description: "本文提出 J1 方法，通过强化学习和合成数据训练大型语言模型作为通用评判者，在多个基准上显著提升评判准确性和位置一致性。"
---

> **Summary:** 本文提出 J1 方法，通过强化学习和合成数据训练大型语言模型作为通用评判者，在多个基准上显著提升评判准确性和位置一致性。 

> **Keywords:** LLM, Reinforcement Learning, Reasoning, Reward Model, Synthetic Data, Test Time Scaling

**Authors:** Not Specified

**Institution(s):** Not Specified


## Problem Background

大型语言模型（LLM）作为评判者（LLM-as-a-Judge）的应用受到关注，传统奖励模型直接输出分数，缺乏显式推理步骤，导致解释性和泛化能力不足。
本文旨在解决如何训练 LLM 成为通用的评判者，能够同时处理可验证任务（如数学问题）和不可验证任务（如主观用户提示），并缓解位置偏见（Position Bias）问题，同时提升评判的准确性和透明度。

## Method

*   **核心思想:** 通过强化学习（Reinforcement Learning, RL）训练 LLM，使其在评判前生成推理过程（Chain-of-Thought, CoT），从而提高评判的准确性和可解释性。
*   **数据构建:** 使用合成数据（Synthetic Data），包括 WildChat（17K 条）和 MATH（5K 条）数据集，生成高质量和低质量的响应对，将评判任务转化为可验证任务。
*   **训练模式:** 提出两种评判模式：成对评判（Pairwise-J1），输入指令和一对响应，输出优选响应或分数；逐点评判（Pointwise-J1），输入单个响应，输出质量分数，天然避免位置偏见。
*   **奖励机制:** 设计两种奖励：正确性奖励（Verdict Correctness Reward），基于最终判断的准确性；一致性奖励（Verdict Consistency Reward），确保成对评判在响应顺序交换时结果一致，缓解位置偏见。
*   **优化方法:** 采用 Group Relative Policy Optimization（GRPO）算法，联合优化推理过程和最终判断，避免依赖单独的批评模型（Critic Model），提高训练效率。
*   **种子提示:** 使用种子思考提示（Seed Thinking Prompt）引导模型生成详细推理轨迹，确保评判过程透明且逻辑清晰。
*   **模型实现:** 基于 Llama-3.1-8B-Instruct 和 Llama-3.3-70B-Instruct 训练 J1-Llama-8B 和 J1-Llama-70B 模型，验证方法在不同规模下的效果。

## Experiment

*   **有效性:** J1-Llama-70B 在 PPE 数据集上达到 69.6% 准确率，超越同规模基线（如 EvalPlanner 67.9%，DeepSeek-GRM-27B 62.2%），在非可验证任务上甚至优于更大的 DeepSeek-R1 模型；J1-Llama-8B 也显著优于同规模基线（如 EvalPlanner-8B 54.1% 提升至 59.8%）。
*   **位置偏见缓解:** 通过一致性奖励和 Pointwise-J1 设计，位置一致性准确率显著提升，例如 J1-Llama-70B 在 JudgeBench 上达到 60.0%，Pointwise-J1 的平局率低至 9.4%。
*   **测试时扩展:** 采用自一致性（Self-Consistency）和多生成采样（Sampling with N=32），性能进一步提升，例如 J1-Llama-70B 在 PPE Correctness 上从 72.9% 提升至 79.9%。
*   **实验设置合理性:** 实验覆盖多种基准数据集（PPE, RewardBench, JudgeBench, RM-Bench, FollowBenchEval），任务类型包括可验证和不可验证，与多种基线（零样本 LLM、标量奖励模型、生成式奖励模型）对比，设置较为全面；但训练数据量较小（仅 22K 合成数据对），可能限制泛化能力，且未探讨跨语言或文化背景的表现。

## Further Thoughts

将评判任务转化为可验证任务的思路非常具有启发性，可以推广到其他需要透明解释的领域，如代码审查或医疗诊断辅助；Pointwise-J1 通过成对监督训练逐点评判模型的设计，或许能应用于其他需要一致性决策的场景；测试时扩展（Test-Time Scaling）通过多生成采样提升性能的策略，启发我们在高风险决策场景中投入更多推理计算资源以换取更高精度。