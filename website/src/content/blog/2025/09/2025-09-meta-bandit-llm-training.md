---
title: "When Greedy Wins: Emergent Exploitation Bias in Meta-Bandit LLM Training"
pubDatetime: 2025-09-29T15:25:42+00:00
slug: "2025-09-meta-bandit-llm-training"
type: "arxiv"
id: "2509.24923"
score: 0.8622425665243745
author: "grok-3-latest"
authors: ["Sanxing Chen", "Xiaoyin Chen", "Yukun Huang", "Roy Xie", "Bhuwan Dhingra"]
tags: ["LLM", "Reinforcement Learning", "Supervised Fine-Tuning", "Exploration-Exploitation", "Bandit Task"]
institution: ["Duke University", "Mila - Québec AI Institute", "Université de Montréal"]
description: "本文通过监督微调和强化学习提升大型语言模型在多臂老虎机任务中的探索能力，并揭示训练诱导的贪婪偏见，为设计鲁棒探索策略提供重要见解。"
---

> **Summary:** 本文通过监督微调和强化学习提升大型语言模型在多臂老虎机任务中的探索能力，并揭示训练诱导的贪婪偏见，为设计鲁棒探索策略提供重要见解。 

> **Keywords:** LLM, Reinforcement Learning, Supervised Fine-Tuning, Exploration-Exploitation, Bandit Task

**Authors:** Sanxing Chen, Xiaoyin Chen, Yukun Huang, Roy Xie, Bhuwan Dhingra

**Institution(s):** Duke University, Mila - Québec AI Institute, Université de Montréal


## Problem Background

大型语言模型（LLMs）在序列决策任务中，如多臂老虎机（Multi-Armed Bandit, MAB），往往表现出短视的贪婪行为，倾向于过度利用已知高回报选项而忽视探索潜在更好选择，导致长期决策表现不佳。
论文旨在通过监督微调（SFT）和强化学习（RL）两种训练范式改进 LLMs 的探索策略，并深入分析这些方法如何塑造模型行为，尤其是在泛化能力和长期鲁棒性方面的表现，解决的关键问题是训练是否能真正提升探索能力，以及诱导的策略是否具有长期鲁棒性。

## Method

*   **监督微调（Supervised Fine-Tuning, SFT）：** 通过在专家轨迹（如 UCB 算法生成的决策序列）上训练，让 LLMs 模仿最优探索策略。训练数据包含详细的思维链（Chain-of-Thought, CoT）演示，指导模型如何计算 UCB 值并做出决策，属于离线学习方式，强调对专家行为的直接模仿。
*   **强化学习（Reinforcement Learning, RL）：** 采用 PPO（Proximal Policy Optimization）算法，通过与环境交互直接优化策略，设计了三种奖励信号以应对不同的学习挑战：
    *   **原始强盗奖励（RL-OG）：** 直接使用环境的随机奖励作为信号，但由于奖励的高方差和信用分配难题，学习效率较低。
    *   **策略性奖励（RL-STG）：** 基于即时遗憾（immediate regret）设计，即最优臂与选择臂预期奖励之差，旨在通过降低训练过程中的方差提升学习效率。
    *   **算法性奖励（RL-ALG）：** 通过模仿 UCB 专家策略的决策提供二元奖励信号（匹配为 1，不匹配为 0），简化信用分配问题，促进高效的模仿学习。
*   **技术细节：** 针对 LLM 的 token 生成特性，设计了层次化 MDP 结构，在 token 级别和回合级别分别操作，并引入双重折扣因子和 GAE（Generalized Advantage Estimator）来优化 token 级别的优势估计，确保训练过程适应 LLM 的生成模式。

## Experiment

*   **有效性：** SFT 和 RL 训练的模型在累积遗憾（cumulative regret）上显著优于预训练模型，性能接近经典算法如 UCB 和 Thompson Sampling，尤其模仿 UCB 的策略（SFT 和 RL-ALG）表现最佳，甚至在某些环境中超越教师策略。
*   **泛化性：** RL 策略在跨分布泛化（如从 Gaussian 到 Bernoulli）中表现更鲁棒，而 SFT 策略在分布外（OOD）环境中表现出更高的变异性和最差情况下的遗憾，尤其在处理负奖励时出现灾难性遗忘。
*   **行为分析：** 尽管平均性能提升，训练后的模型表现出更强的贪婪倾向（higher suffix failure rate 和 greedy frequency），容易过早放弃探索最佳选项，这种行为在 RL-ALG 中尤为明显，其策略演变为 UCB 的变体，倾向于短期利用而非长期探索。
*   **模型规模影响：** 较小的 3B 模型在直接环境奖励（RL-OG 和 RL-STG）上的学习效果有限，但在模仿学习（SFT 和 RL-ALG）中仍能显著提升，表明小模型对 RL 优化过程的适应能力较弱。
*   **实验设置合理性：** 实验覆盖了多种环境（Gaussian 和 Bernoulli）、模型规模（3B 和 7B）和评估指标（累积遗憾、平均奖励、最佳臂选择频率等），并通过行为分析揭示了平均遗憾之外的策略缺陷，设置较为全面，但由于 LLM 推理成本高，评估样本量（64 个 episode）较小，可能影响统计稳定性。

## Further Thoughts

论文中通过设计特定奖励信号（如 RL-STG 和 RL-ALG）引导 LLMs 探索行为的思路非常具有启发性，特别是 RL-ALG 的模仿学习方式展示了如何通过简单的二元奖励信号让模型自发发现近似最优策略（如 UCB 变体），这为未来在复杂任务中设计高效奖励函数提供了方向；此外，训练数据分布对泛化能力的深远影响（如 SFT 在负奖励环境中的失败）提示我们需要在训练中引入更多样化的数据或混合数学任务以避免能力退化；最后，行为分析的视角（关注 suffix failure 等指标而非仅平均遗憾）提醒我们在评估智能体时需重视长期鲁棒性，而非仅仅短期性能。