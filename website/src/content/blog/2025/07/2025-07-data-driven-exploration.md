---
title: "Data-Driven Exploration for a Class of Continuous-Time Linear--Quadratic Reinforcement Learning Problems"
pubDatetime: 2025-07-01T01:09:06+00:00
slug: "2025-07-data-driven-exploration"
type: "arxiv"
id: "2507.00358"
score: 0.539848962295169
author: "grok-3-latest"
authors: ["Yilie Huang", "Xun Yu Zhou"]
tags: ["Reinforcement Learning", "Continuous Time", "Exploration Strategy", "Actor Critic", "Regret Bound"]
institution: ["Columbia University"]
description: "本文提出了一种数据驱动的自适应探索机制，通过动态调整评论家熵正则化和演员策略方差，显著提升了连续时间线性-二次强化学习的学习效率，同时保持了次线性遗憾界。"
---

> **Summary:** 本文提出了一种数据驱动的自适应探索机制，通过动态调整评论家熵正则化和演员策略方差，显著提升了连续时间线性-二次强化学习的学习效率，同时保持了次线性遗憾界。 

> **Keywords:** Reinforcement Learning, Continuous Time, Exploration Strategy, Actor Critic, Regret Bound

**Authors:** Yilie Huang, Xun Yu Zhou

**Institution(s):** Columbia University


## Problem Background

在连续时间线性-二次（LQ）控制问题中，特别是在状态和控制依赖波动率的场景下，传统基于模型的方法依赖于参数估计，但现实中模型参数往往未知，导致性能不佳。
无模型强化学习（RL）通过直接与环境交互学习最优策略，但探索-利用权衡是一个核心挑战，尤其在连续时间设置下，固定或确定性探索策略（如[1]中使用的）需要大量手动调参，且无法根据学习进度动态调整，导致学习效率低下或收敛到次优策略。

## Method

*   **核心思想:** 提出一种数据驱动的自适应探索机制，通过动态调整评论家（critic）的熵正则化参数和演员（actor）的策略方差，优化探索-利用权衡，提高学习效率。
*   **具体实现:** 
    *   **评论家探索（Critic Exploration）:** 参数化价值函数为状态的二次形式，温度参数（γ）根据学习进度和预定义的单调递增序列（b_n）动态更新，公式为 γ_n = c_γ * (∫k_1 dt) / (b_n * T)，确保随迭代逐渐减小，促进收敛。
    *   **演员探索（Actor Exploration）:** 策略采用高斯分布，均值（ϕ）和方差（Γ）为可学习参数，方差通过策略梯度方法基于数据驱动更新，公式为 Γ_{n+1} = Γ_n - a_n * Z_n(T)，其中Z_n(T)结合了熵项和价值函数梯度，确保探索水平随环境反馈调整。
    *   **算法框架:** 基于演员-评论家方法，结合策略评估（Policy Evaluation）和策略改进（Policy Improvement），通过随机逼近（Stochastic Approximation）和投影技术确保参数更新稳定，时间离散化用于数值实现。
*   **关键特点:** 不依赖模型参数知识，避免了固定探索策略的过度或不足探索问题，同时通过理论分析确保收敛性和次线性遗憾界。

## Experiment

*   **有效性:** 数值实验验证了理论结果，参数Γ和ϕ的均方误差（MSE）收敛速率分别为-0.51和-0.52，与理论预测（Theorem 5.1和5.8）一致；遗憾增长斜率为0.73，符合次线性遗憾界O(N^{3/4})（Theorem 5.9）。
*   **优越性:** 对比基于模型方法（[25]改进版），自适应探索的无模型方法在ϕ收敛速率上更快（-0.52 vs -0.25），遗憾增长更慢（0.73 vs 0.84）；对比固定探索策略（[1]），自适应方法在过度探索（初始Γ=20）和不足探索（初始Γ=0.02）场景下均快速调整探索水平，累计遗憾显著低于固定策略。
*   **全面性与合理性:** 实验设置涵盖了不同初始条件（x_0≠0和x_0=0）、随机化模型参数和探索参数，运行多次独立实验（100至10000次）以确保统计可靠性，时间步长（∆t=0.01）选择合理，充分验证了方法的鲁棒性和适应性。
*   **开销:** 自适应更新增加了计算复杂度，主要体现在每次迭代中动态计算γ和Γ的梯度，但通过投影和参数化设计降低了数值不稳定性，整体计算成本在实验中可控。

## Further Thoughts

数据驱动的自适应探索策略在连续时间RL中展现了显著优势，这启发我们可以在其他复杂RL任务（如非线性系统或高维环境）中探索基于学习进度的动态调整机制；此外，论文中熵正则化和策略方差的结合方式提示我们，未来可以尝试将目标导向的约束或奖励机制融入探索策略，以进一步提升采样效率和学习稳定性。