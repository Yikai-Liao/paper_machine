---
title: "Reward-Independent Messaging for Decentralized Multi-Agent Reinforcement Learning"
pubDatetime: 2025-05-28T05:23:47+00:00
slug: "2025-05-marl-cpc-communication"
type: "arxiv"
id: "2505.21985"
score: 0.5630341541710155
author: "grok-3-latest"
authors: ["Naoto Yoshida", "Tadahiro Taniguchi"]
tags: ["MARL", "Emergent Communication", "Predictive Coding", "Decentralized Learning", "State Inference"]
institution: ["Kyoto University", "Ritsumeikan University"]
description: "本文提出 MARL-CPC 框架，通过集体预测编码实现去中心化多智能体强化学习中的奖励无关通信，显著提升非合作环境下的群体性能。"
---

> **Summary:** 本文提出 MARL-CPC 框架，通过集体预测编码实现去中心化多智能体强化学习中的奖励无关通信，显著提升非合作环境下的群体性能。 

> **Keywords:** MARL, Emergent Communication, Predictive Coding, Decentralized Learning, State Inference

**Authors:** Naoto Yoshida, Tadahiro Taniguchi

**Institution(s):** Kyoto University, Ritsumeikan University


## Problem Background

在多智能体强化学习（MARL）中，特别是在部分可观测环境下，智能体间的有效通信能够显著提升群体性能。然而，传统方法多依赖中心化训练或假设合作环境，而现实中智能体往往是去中心化、独立学习且奖励不一致（非合作环境），导致通信难以自发形成。本文旨在解决这一挑战，探索如何在非合作、去中心化的 MARL 场景中让独立智能体建立并利用有效通信，以提升决策能力和群体适应性。

## Method

* **核心思想：** 提出 MARL-CPC 框架，基于集体预测编码（Collective Predictive Coding, CPC），将通信建模为状态推断的辅助变量，而非传统方法中作为动作空间的一部分。这种方法不依赖奖励驱动，而是通过通信支持分布式状态估计。
* **理论基础：** 构建一个伪联合生成模型，将多智能体系统视为一个整体的生成模型，通过变分推断（Variational Inference）推导每个智能体的通信模块目标函数（ELBO），使每个智能体能独立优化通信行为。
* **具体实现：** 
  - 每个智能体基于自身观测生成消息（message），通过变分分布采样，并将联合消息用于全局状态推断。
  - 通信模块与强化学习模块分离，梯度不相互传播，确保去中心化学习。
  - 提出两种算法：Bandit-CPC 适用于上下文 bandit 问题，IPPO-CPC 结合 Proximal Policy Optimization（PPO）适用于复杂状态转移环境。
* **关键创新：** 通信学习独立于奖励机制，适用于非合作环境；通过直通梯度估计器（straight-through gradient estimator）和采样技术优化离散消息的训练。

## Experiment

* **有效性：** 在上下文 bandit 和观察者（Observer）两个非合作环境中，MARL-CPC 显著提升了群体福利（group welfare），接近共享信息条件下的性能上限，优于无通信（no-comm）和传统消息作为动作（message）基线。
* **对比分析：** 在非合作场景中，传统方法未能形成有效通信，而 MARL-CPC 成功实现了信息共享，尤其在观察者环境中，CPC 条件下的 episode length 和 group welfare 均有明显提升。
* **实验设置：** 覆盖简单（bandit）和复杂（状态转移）场景，包含消融研究验证消息实用性，设计合理；但智能体数量较少（仅两个），未测试大规模系统，训练时间较长（IPPO-CPC 需 3×10^6 步）。

## Further Thoughts

MARL-CPC 将通信视为状态推断辅助变量的思路启发了我，是否可以将这种机制扩展到其他分布式学习任务，如分布式优化或多模态数据融合？此外，CPC 基于去中心化贝叶斯推断的框架是否能结合现代生成模型（如扩散模型）进一步增强通信表达能力？另一个有趣的方向是，通信涌现与奖励无关的特性可能为研究自然语言进化或社会行为提供新视角。