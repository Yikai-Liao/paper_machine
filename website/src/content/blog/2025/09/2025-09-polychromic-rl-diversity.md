---
title: "Polychromic Objectives for Reinforcement Learning"
pubDatetime: 2025-09-29T19:32:11+00:00
slug: "2025-09-polychromic-rl-diversity"
type: "arxiv"
id: "2509.25424"
score: 0.7193199804849694
author: "grok-3-latest"
authors: ["Jubayer Ibn Hamid", "Ifdita Hasan Orney", "Ellen Xu", "Chelsea Finn", "Dorsa Sadigh"]
tags: ["LLM", "Reinforcement Learning", "Diversity", "Sampling", "Exploration"]
institution: ["Stanford University"]
description: "本文通过集合强化学习框架和多色目标，提出多色PPO算法，有效解决强化学习微调中的熵崩溃问题，提升了策略多样性和泛化能力。"
---

> **Summary:** 本文通过集合强化学习框架和多色目标，提出多色PPO算法，有效解决强化学习微调中的熵崩溃问题，提升了策略多样性和泛化能力。 

> **Keywords:** LLM, Reinforcement Learning, Diversity, Sampling, Exploration

**Authors:** Jubayer Ibn Hamid, Ifdita Hasan Orney, Ellen Xu, Chelsea Finn, Dorsa Sadigh

**Institution(s):** Stanford University


## Problem Background

强化学习微调（Reinforcement Learning Fine-Tuning, RLFT）是提升预训练策略在下游任务中表现的重要方法，但其常见问题在于熵崩溃（entropy collapse），即微调后的策略倾向于集中于少数高回报行为，丧失了预训练分布中的多样性。这种多样性丧失限制了模型的探索能力，影响其在新任务上的泛化能力和测试时计算扩展（test-time compute scaling）的效果。

## Method

*   **集合强化学习框架（Set Reinforcement Learning, Set RL）**：提出了一种新的强化学习范式，将优化目标从单一轨迹的回报扩展到一组轨迹的集合目标。这种框架允许设计更复杂的目标函数，不仅关注回报，还能纳入其他特性如多样性。
*   **多色目标（Polychromic Objective）**：设计了一种目标函数，结合轨迹集合的回报（reward）和多样性（diversity）。具体而言，该目标通过对集合整体评分，确保策略在追求高回报的同时，生成的轨迹保持多样性，避免熵崩溃。回报和多样性均被归一化处理，确保两者在优化中的平衡。
*   **多色PPO（Polychromic PPO）**：基于近端策略优化（Proximal Policy Optimization, PPO）算法进行改进，通过以下步骤实现多色目标的优化：
    *   **藤蔓采样（Vine Sampling）**：在策略上的数据收集阶段，从选定的关键状态（rollout states）生成多条轨迹，确保数据覆盖多样化的行为路径。这种采样方式避免了指数级的数据需求，同时保证了探索性。
    *   **优势函数修改**：将传统PPO中的优势函数调整为多色优势函数，基于集合的回报和多样性评分计算，确保更新信号反映多色目标的要求。所有轨迹在集合中共享相同的学习信号，促进整体多样性。
    *   **稳定性优化**：在非关键状态使用标准PPO更新，并引入KL散度惩罚以防止策略偏离过远，同时在关键状态附近设置更新窗口，确保探索行为的持续性。
*   **关键特点**：方法不依赖于简单的熵正则化，而是通过集合级目标直接优化语义和轨迹级别的多样性，同时保持性能。

## Experiment

*   **性能表现**：在 BabyAI 和 Minigrid 环境中，多色PPO在平均回报和成功率上与最优基线（如 REINFORCE 和标准 PPO）相当或更优，尤其在 Goto 和 Pickup 任务中成功率提升明显；在 Algorithmic Creativity 任务中，性能略低于标准 PPO，但仍显著优于预训练策略。
*   **多样性提升**：通过 pass@k 指标，多色PPO显著提高了任务覆盖率，尤其在高 k 值时（如 k=80），表明其生成的轨迹更具多样性，避免了策略集中于少数行为的问题。
*   **泛化能力**：在初始状态扰动实验中，多色PPO表现出更强的鲁棒性，例如在 BabyAI 的 Goto 任务中 pass@1 达到 60.6%，远高于基线，表明多样性有助于应对未见过的状态。
*   **实验设置合理性**：实验覆盖了长距离任务和稀疏奖励环境，基线方法包括 REINFORCE、标准 PPO 及 UCB 正则化，数据统计基于多轮 rollout 和多种配置。然而，实验环境相对受限，未涉及连续控制或更高维度的任务，且藤蔓采样增加了计算开销，未充分讨论资源效率问题。

## Further Thoughts

集合强化学习框架为优化复杂目标（如多样性、鲁棒性）提供了新思路，未来可以探索将其与元学习结合，学习适应性更强的策略；此外，多色目标的动态权重调整（在训练早期强调探索，后期强调利用）可能进一步优化性能与多样性的平衡；藤蔓采样的思想也启发我们设计自适应采样策略，根据任务特性动态选择关键状态，提升探索效率。