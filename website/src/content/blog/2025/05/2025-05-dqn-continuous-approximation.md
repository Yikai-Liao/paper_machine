---
title: "Universal Approximation Theorem of Deep Q-Networks"
pubDatetime: 2025-05-04T22:57:33+00:00
slug: "2025-05-dqn-continuous-approximation"
type: "arxiv"
id: "2505.02288"
score: 0.4382265927087064
author: "grok-3-latest"
authors: ["Qian Qi"]
tags: ["Deep Q-Networks", "Reinforcement Learning", "Continuous Time", "Approximation", "Convergence"]
institution: ["Peking University"]
description: "本文提出一个连续时间框架，证明深度 Q 网络（DQNs）对最优 Q 函数的通用逼近能力，并分析 Q 学习算法的收敛性，为深度强化学习与随机控制理论的结合奠定基础。"
---

> **Summary:** 本文提出一个连续时间框架，证明深度 Q 网络（DQNs）对最优 Q 函数的通用逼近能力，并分析 Q 学习算法的收敛性，为深度强化学习与随机控制理论的结合奠定基础。 

> **Keywords:** Deep Q-Networks, Reinforcement Learning, Continuous Time, Approximation, Convergence

**Authors:** Qian Qi

**Institution(s):** Peking University


## Problem Background

本文聚焦于深度 Q 网络（Deep Q-Networks, DQNs）在连续时间环境下的理论基础，针对强化学习（Reinforcement Learning, RL）在连续时间马尔可夫决策过程（MDP）中的应用，解决现有理论分析主要局限于离散时间设置的不足，特别是在物理系统或高频数据等连续时间场景中的适用性问题。
关键问题在于证明 DQNs 是否能在连续时间框架下以任意精度逼近最优 Q 函数，并确保训练算法的收敛性。

## Method

*   **连续时间框架构建**：提出了一种基于随机控制和前后向随机微分方程（FBSDEs）的分析框架，将 DQNs 嵌入到连续时间 MDP 中，状态动态由随机微分方程（SDE）驱动，使用平方可积鞅作为噪声源，以模拟连续时间环境中的不确定性。
*   **通用逼近能力证明**：基于残差网络（ResNets）的通用逼近定理，证明 DQNs 可以在紧凑集上以任意精度逼近最优 Q 函数，强调网络深度与逼近精度的关系。
*   **收敛性分析**：通过随机逼近理论，分析 Q 学习算法在连续时间设置下的收敛性，引入 Bellman 算子的收缩性质，并利用粘性解（viscosity solutions）处理最优 Q 函数可能存在的非光滑性，确保理论分析的严谨性。
*   **架构设计与时间关联**：设计采用残差块（residual blocks）的 DQN 架构，将网络层数与时间离散化步长（∆t = T/N）相关联，旨在通过网络结构模拟连续时间动态的演化。
*   **理论工具结合**：综合深度学习、随机控制和微分方程工具，建立深度强化学习与连续时间控制理论之间的桥梁。

## Experiment

*   **实验设置**：在 1D 连续控制环境中进行数值实验，状态动态由 SDE 驱动，目标是稳定状态到原点，奖励函数鼓励接近零点并惩罚控制成本；采用 Euler-Maruyama 方案离散化时间步长。
*   **对比分析**：测试了不同配置（如学习率、残差块数量、环境噪声强度、目标网络更新频率）对性能的影响，基线配置使用 2 个残差块。
*   **效果评估**：基线配置表现出稳定学习，奖励和损失曲线表明成功逼近目标策略；增加学习率可能导致不稳定，移除残差块（即使用标准 MLP）可能略微降低性能，增加噪声显著增加任务难度。
*   **合理性与局限**：实验设置较为简单（1D 环境），可能无法完全体现高维任务中残差网络的优势，但提供了理论框架的可行性验证；结果与理论预期一致，表明 DQN 在离散化后的连续时间环境中可有效训练。

## Further Thoughts

连续时间框架与 FBSDEs 的结合为分析强化学习算法在物理系统或高频数据场景提供了新视角，粘性解处理非光滑问题的思路可能启发其他 RL 算法的理论分析；此外，将网络深度与时间步长挂钩的设计可能为适应连续动态的神经网络架构提供新思路。发散性思考：是否可以将此框架扩展到部分可观测环境（POMDP）？是否能结合 Transformer 等架构处理时间依赖性？粘性解是否适用于其他非光滑优化问题？