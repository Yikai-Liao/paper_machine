---
title: "Gaze Into the Abyss -- Planning to Seek Entropy When Reward is Scarce"
pubDatetime: 2025-05-22T15:28:50+00:00
slug: "2025-05-entropy-seeking-planning"
type: "arxiv"
id: "2505.16787"
score: 0.652657158539476
author: "grok-3-latest"
authors: ["Ashish Sundar", "Chunbo Luo", "Xiaoyang Wang"]
tags: ["MBRL", "World Model", "Entropy Driven Exploration", "Hierarchical Planning", "Sparse Reward"]
institution: ["University of Exeter"]
description: "本文提出了一种基于熵驱动的提前规划方法，通过优化世界模型的信息增益和动态分层规划，显著提升了稀疏奖励环境下的探索效率和策略性能。"
---

> **Summary:** 本文提出了一种基于熵驱动的提前规划方法，通过优化世界模型的信息增益和动态分层规划，显著提升了稀疏奖励环境下的探索效率和策略性能。 

> **Keywords:** MBRL, World Model, Entropy Driven Exploration, Hierarchical Planning, Sparse Reward

**Authors:** Ashish Sundar, Chunbo Luo, Xiaoyang Wang

**Institution(s):** University of Exeter


## Problem Background

在稀疏奖励、部分可观察和随机环境中，强化学习（RL）的探索效率和样本效率仍然是一个未解决的难题。
传统的基于好奇心的探索方法多为回顾性，依赖过去偶然发现的新奇状态，缺乏实时适应性，尤其在非平稳环境中效果有限。
模型基强化学习（MBRL）通过训练世界模型预测未来状态来提高样本效率，但现有方法往往忽视世界模型本身的优化，仅将其作为策略学习的工具。
本文提出了一种新视角：通过优先优化世界模型的学习，主动寻找高熵状态以加速信息增益，从而显著提升下游策略的性能。

## Method

*   **核心思想:** 利用世界模型的潜在状态分布熵作为不确定性指标，主动寻找高熵状态以最大化信息增益，同时通过分层规划动态平衡探索与利用，加速世界模型和策略的学习。
*   **熵驱动探索:** 在 Dreamer 框架基础上，利用潜在状态的 Gaussian 分布熵（prior entropy）作为探索目标，通过将 KL 散度最小化重构为对抗性 minmax 目标，优化世界模型的信息增益（Information Gain）。
*   **反应式分层规划器:** 设计一个基于 PPO 的分层规划器，包括两个元策略（Meta-Policies）：Meta 1 决定规划 horizon 长度和熵-奖励权重，Meta 2 决定重新规划的概率。规划器在每个步骤滚动出多个候选轨迹（256 个），基于‘endorphin’评分（熵与奖励的加权和）选择最佳轨迹，并通过随机机制避免过度重新规划，确保计划承诺与灵活性的平衡。
*   **任务感知重构损失:** 对 Dreamer 的重构损失进行改进，结合 MSE、MAE 和 SSIM 损失，并对任务相关区域（如目标框）加权，提升潜在空间的语义丰富度和收敛速度。
*   **适用性:** 方法理论上适用于任何完全通过模型生成数据训练策略的 MBRL 框架，尽管实验仅在 Dreamer 上验证。

## Experiment

*   **有效性:** 在 MiniWorld 的 3D 迷宫环境中，与基线 Dreamer 和 PPO 相比，提出的方法在所有难度（通过 porosity 参数调整）下表现更优，尤其在高难度（低 porosity）环境下，完成任务的 episode length 比 Dreamer 短 20%，比 PPO 更稳定；整体收敛速度比基线 Dreamer 快 50%，策略训练所需环境步数仅为基线 Dreamer 的 60%。
*   **探索效率:** 方法保持约 10% 更高的 prior entropy，表明探索策略更广泛；KL 散度也较高，反映世界模型在学习过程中遇到了更多挑战性状态，信息增益更大。
*   **消融研究:** 对比无规划的基线 Dreamer 和 MPC 风格的每步重新规划方法，完整规划器（Full Planner）表现最佳，证明了计划承诺和动态规划的重要性；元策略动态平衡熵与奖励，未陷入单一极端。
*   **合理性与局限:** 实验设置全面，考虑了不同难度环境，并通过多种指标（episode length、prior entropy、KL divergence）评估性能；消融研究验证了各组件贡献；但实验仅在单一环境和模型（Dreamer）上进行，泛化性待进一步验证。

## Further Thoughts

论文提出将世界模型的学习置于优先地位的视角，可能适用于其他 MBRL 框架，甚至扩展到 Transformer 状态空间模型（TSSMs），为设计更复杂的模型提供了思路。
熵驱动探索结合动态规划的机制，可以启发在机器人导航或科学探索等稀疏奖励任务中设计智能探索策略。
论文提到利用大型语言模型（LLMs）提供语义提示以解决潜在过渡被低估的问题，这一设想为结合符号推理与强化学习开辟了新方向。
分层规划器动态平衡多目标（如熵与奖励）的机制，可能适用于更广泛的多目标优化问题，尤其是在需要权衡探索与利用的场景中。