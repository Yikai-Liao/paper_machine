---
title: "Universal Approximation Theorem of Deep Q-Networks"
pubDatetime: 2025-05-04T22:57:33+00:00
slug: "2025-05-dqn-continuous-approximation"
type: "arxiv"
id: "2505.02288"
score: 0.4382265927087064
author: "grok-3-latest"
authors: ["Qian Qi"]
tags: ["Deep Q-Networks", "Reinforcement Learning", "Continuous Time", "Stochastic Control", "Approximation Theory"]
institution: ["Peking University"]
description: "本文在连续时间框架下证明了深度 Q 网络对最优 Q 函数的通用逼近能力，并通过随机逼近理论分析了 Q 学习算法的收敛性，为深度强化学习与随机控制的结合提供了理论基础。"
---

> **Summary:** 本文在连续时间框架下证明了深度 Q 网络对最优 Q 函数的通用逼近能力，并通过随机逼近理论分析了 Q 学习算法的收敛性，为深度强化学习与随机控制的结合提供了理论基础。 

> **Keywords:** Deep Q-Networks, Reinforcement Learning, Continuous Time, Stochastic Control, Approximation Theory

**Authors:** Qian Qi

**Institution(s):** Peking University


## Problem Background

深度 Q 网络（Deep Q-Networks, DQNs）在离散时间强化学习中取得了显著成功，但其在连续时间环境（如物理系统或高频数据场景）下的理论基础尚未充分探索。
本文旨在解决这一关键问题：证明 DQNs 在连续时间马尔可夫决策过程中是否能以任意精度逼近最优 Q 函数，并分析其训练算法的收敛性，为实际应用提供理论支持。

## Method

*   **框架构建**：提出一个基于随机控制和前后向随机微分方程（FBSDEs）的连续时间框架，将 DQNs 嵌入到由平方可积鞅驱动的连续时间 MDP 中，状态动态通过随机微分方程（SDE）描述。
*   **逼近能力分析**：利用残差网络（ResNets）的通用逼近定理，证明 DQNs 可以在紧凑集上以任意精度逼近最优 Q 函数，具体通过构建基于残差块的网络架构，并将网络深度与时间离散化步长关联（L = N, Δt = T/N）。
*   **收敛性证明**：基于随机逼近理论，分析 Q 学习算法在连续时间设置下的收敛性，特别关注 Bellman 算子的收缩性质和粘性解（viscosity solutions）在处理非光滑最优 Q 函数中的作用。
*   **理论连接**：通过 Hamilton-Jacobi-Bellman（HJB）方程和 FBSDEs 建立深度强化学习与随机控制理论的联系，为 DQN 的行为提供概率解释和数学基础。

## Experiment

*   **任务设置**：在 1D 连续控制任务中验证理论，任务目标是稳定状态到原点，环境动态由 SDE 驱动，并通过 Euler-Maruyama 方案离散化。
*   **有效性**：基于残差块的 DQN 架构在训练中表现出稳定学习，奖励曲线和损失曲线表明其能够逼近最优策略，验证了理论上的逼近能力。
*   **对比结果**：与标准多层感知机（MLP）相比，残差块架构在简单任务中表现略优（收敛速度和最终奖励），表明其潜在优势。
*   **参数影响**：测试了学习率、环境噪声、目标网络更新频率等参数的影响，高噪声环境显著增加难度，但残差块有助于缓解不稳定性。
*   **局限性**：实验任务较为简单，未能完全体现连续时间设置和高维复杂任务中的挑战，未提供逼近误差或收敛速率的量化数据，实验设置合理但覆盖范围有限。

## Further Thoughts

论文将网络深度与时间离散化步长关联（L = N, Δt = T/N）的思想非常启发性，提示我们可以在其他动态优化问题中利用深度学习架构模拟时间演化，例如在金融市场预测或机器人路径规划中设计类似结构；此外，粘性解处理非光滑问题的思路也可能推广到其他非经典可微优化场景，如对抗性训练或稀疏优化。