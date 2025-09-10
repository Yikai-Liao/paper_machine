---
title: "Learning Optimal Defender Strategies for CAGE-2 using a POMDP Model"
pubDatetime: 2025-09-08T10:51:43+00:00
slug: "2025-09-cage2-pomdp-defense"
type: "arxiv"
id: "2509.06539"
score: 0.7300757192501139
author: "grok-3-latest"
authors: ["Duc Huy Le", "Rolf Stadler"]
tags: ["POMDP", "Reinforcement Learning", "Cyber Defense", "Particle Filter", "Strategy Optimization"]
institution: ["KTH Royal Institute of Technology"]
description: "本文通过 POMDP 框架形式化 CAGE-2 网络防御场景，提出 BF-PPO 方法结合粒子滤波和 PPO 强化学习高效学习最优防御策略，在性能和收敛速度上超越现有最佳方法 CARDIFF。"
---

> **Summary:** 本文通过 POMDP 框架形式化 CAGE-2 网络防御场景，提出 BF-PPO 方法结合粒子滤波和 PPO 强化学习高效学习最优防御策略，在性能和收敛速度上超越现有最佳方法 CARDIFF。 

> **Keywords:** POMDP, Reinforcement Learning, Cyber Defense, Particle Filter, Strategy Optimization

**Authors:** Duc Huy Le, Rolf Stadler

**Institution(s):** KTH Royal Institute of Technology


## Problem Background

CAGE-2 是一个用于学习和评估网络防御策略的基准环境，模拟防御者保护 IT 基础设施免受攻击的场景。
传统入侵检测依赖人工规则，难以应对复杂动态环境，而现有 CAGE-2 防御方法多为启发式，缺乏形式化模型，无法证明策略最优性。
论文旨在通过构建部分可观测马尔可夫决策过程（POMDP）模型，定义最优防御策略，并解决大规模状态空间的计算复杂性问题。

## Method

*   **核心思想:** 使用部分可观测马尔可夫决策过程（POMDP）框架形式化 CAGE-2 场景，定义最优防御策略，并通过结合强化学习和粒子滤波的方法高效学习该策略。
*   **POMDP 建模:** 详细定义了状态空间（基础设施中各主机的状态，包括攻击者访问状态和服务状态）、动作空间（防御者动作如分析、部署诱饵、中和恶意软件、恢复主机）、状态转移（攻击者和防御者动作导致的状态变化）、观测空间（通过入侵检测系统获取的部分信息）以及奖励函数（平衡服务可用性和攻击者访问最小化）。
*   **学习方法 BF-PPO:** 提出 Belief Filter Policy Proximal Optimisation (BF-PPO) 方法，基于 Proximal Policy Optimization (PPO) 强化学习算法，使用神经网络表示策略，通过梯度上升优化累积奖励。
    *   针对 POMDP 中状态空间过大（约 10^39）导致的计算复杂性，引入粒子滤波（Particle Filter）近似贝叶斯滤波，通过采样粒子估计信念状态（Belief State）。
    *   具体流程：首先通过粒子滤波生成信念状态的粒子集合，从中采样代表性状态作为神经网络输入，最后由 PPO 训练的策略选择防御动作。
*   **关键优势:** 不需精确计算信念状态，显著降低计算开销，同时适应部分可观测环境，确保策略优化效率。

## Experiment

*   **有效性:** 在 CAGE-2 的 CybORG 环境中，BF-PPO 在两种攻击场景（B-LINE 和 MEANDER）下均取得更高的累积奖励，尤其在 MEANDER 场景中表现突出（如 T=100 时，BF-PPO 奖励为 -11.56 ± 4.22，优于 CARDIFF 的 -16.6 ± 3.83）。
*   **收敛速度:** 学习曲线显示 BF-PPO 收敛更快，迅速达到稳定奖励值，而 CARDIFF 在训练后期仍波动，策略未完全收敛。
*   **实验设置:** 实验考虑了多种时间步长（T=30, 50, 100）和攻击模式，通过多次随机种子运行确保结果稳健，设置全面合理，但未探讨更多攻击者策略的适应性。
*   **计算开销:** 主要开销来自粒子滤波的采样和 PPO 的神经网络训练，但相比精确计算信念状态已大幅降低复杂性。

## Further Thoughts

POMDP 框架在网络安全中的应用展示了其处理部分可观测动态系统的潜力，粒子滤波与强化学习的结合为解决大规模状态空间问题提供了有效工具，这种方法可推广至机器人控制或金融决策等领域；此外，将攻防交互建模为动态博弈的思路为未来研究双方策略演化提供了新方向。