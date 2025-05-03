---
title: "Q-function Decomposition with Intervention Semantics with Factored Action Spaces"
pubDatetime: 2025-04-30T05:26:51+00:00
slug: "2025-04-q-decomposition-intervention"
type: "arxiv"
id: "2504.21326"
score: 0.4347574374115348
author: "grok-3-latest"
authors: ["Junkyu Lee", "Tian Gao", "Elliot Nelson", "Miao Liu", "Debarun Bhattacharjya", "Songtao Lu"]
tags: ["Reinforcement Learning", "Factored Action Spaces", "Q-function Decomposition", "Sample Efficiency", "Causal Inference"]
institution: ["IBM T. J. Watson Research Center", "Independent", "The Chinese University of Hong Kong"]
description: "本文提出了一种基于因果干预语义的 Q 函数分解方法，通过投影动作空间和数据增强显著提高大规模分解动作空间中强化学习的样本效率，并在在线和离线环境中均取得优于基线的结果。"
---

> **Summary:** 本文提出了一种基于因果干预语义的 Q 函数分解方法，通过投影动作空间和数据增强显著提高大规模分解动作空间中强化学习的样本效率，并在在线和离线环境中均取得优于基线的结果。 

> **Keywords:** Reinforcement Learning, Factored Action Spaces, Q-function Decomposition, Sample Efficiency, Causal Inference

**Authors:** Junkyu Lee, Tian Gao, Elliot Nelson, Miao Liu, Debarun Bhattacharjya, Songtao Lu

**Institution(s):** IBM T. J. Watson Research Center, Independent, The Chinese University of Hong Kong


## Problem Background

强化学习（Reinforcement Learning, RL）在具有大规模离散分解动作空间（Factored Action Spaces）的环境中面临样本效率低下的挑战，尤其是在离线或离策设置中，数据收集成本高且受限，导致传统算法难以应对动作空间的组合爆炸性问题。
本文旨在通过分解 Q 函数，利用动作空间的结构化特性，显著提高样本效率，同时保持策略优化效果。

## Method

*   **核心思想:** 基于因果干预语义（Intervention Semantics），将大规模分解动作空间的 Q 函数分解为多个投影 Q 函数（Projected Q-functions），从而降低计算复杂性并提高样本效率。
*   **理论基础:** 利用因果统计中的‘无未观察混杂因子’（No Unobserved Confounder）设定，分析 Q 函数分解的无偏性条件，确保分解后的 Q 函数能够准确近似原始 Q 函数。
*   **投影动作空间 MDP:** 定义了投影动作空间的马尔可夫决策过程（MDP），通过干预语义分离动作对状态的影响，将动作空间划分为多个子空间，并在每个子空间上构建独立的投影 Q 函数。
*   **动作分解 RL 框架:** 提出了一种通用的动作分解强化学习（Action Decomposed RL）框架，结合模型自由（Model-Free）算法（如 DQN 和 BCQ），通过学习投影 Q 函数和全局 Q 函数的线性或非线性组合（通过混合器网络）来优化策略。
*   **数据增强与模型学习:** 在训练过程中，利用学习到的动态模型（Dynamics Model）和奖励模型（Reward Model）生成合成样本，增强样本效率，尤其是在离线环境中通过模拟投影 MDP 的过渡来弥补数据不足。
*   **关键创新:** 将因果干预的概念引入 RL，通过分解动作空间的效果（Effects）来降低复杂性，同时在理论上保证收敛性，并在实际应用中通过数据增强提升性能。

## Experiment

*   **有效性:** 在在线的 2D 点质量控制环境（2D Point-Mass Control）中，动作分解 DQN（AD-DQN）相比基线 DECQN 表现出更快的收敛速度和更高的回报，尤其在动作空间规模从 9x9 增加到 33x33 时提升显著；在离线的 MIMIC-III 脓毒症治疗环境（Sepsis Treatment）中，动作分解 BCQ（AD-BCQ）在加权重要性采样（WIS）和有效样本量（ESS）指标上显著优于基线 BCQ 和 Factored BCQ。
*   **优越性:** AD-DQN 和 AD-BCQ 通过权重共享、混合器网络设计和数据增强策略，在样本效率和性能上均优于基线方法，尤其在样本稀疏的离线场景中，Pareto 前沿分析验证了其性能优势。
*   **实验设置合理性:** 实验覆盖了在线和离线两种场景，动作空间规模从小型到大型（5x5 到 14x14 或 33x33）均有测试，超参数配置多样（如不同的混合器网络和 BCQ 阈值），数据增强步骤进一步提升了样本利用率，整体设置全面且合理。
*   **开销:** 主要增加了动态模型和奖励模型的训练成本，以及合成样本生成的时间，但这些开销在样本效率提升的背景下是可接受的。

## Further Thoughts

因果干预语义的应用为强化学习中的结构化问题提供了新视角，未来可探索将其扩展至多智能体 RL 或部分可观察环境（POMDP）；投影 Q 函数与全局 Q 函数结合的思想启发我们在深度学习中设计模块化网络，利用分层结构捕捉局部与全局依赖；数据增强与模型学习的协同策略提示我们可以在数据稀疏领域（如医疗、金融）探索类似的模型辅助训练方法，甚至结合自监督学习进一步减少对标注数据的依赖。