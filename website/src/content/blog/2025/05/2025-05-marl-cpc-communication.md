---
title: "Reward-Independent Messaging for Decentralized Multi-Agent Reinforcement Learning"
pubDatetime: 2025-05-28T05:23:47+00:00
slug: "2025-05-marl-cpc-communication"
type: "arxiv"
id: "2505.21985"
score: 0.5630341541710155
author: "grok-3-latest"
authors: ["Naoto Yoshida", "Tadahiro Taniguchi"]
tags: ["Multi-Agent RL", "Emergent Communication", "Predictive Coding", "Decentralized Learning", "State Inference"]
institution: ["Kyoto University", "Research Organization of Science and Technology, Ritsumeikan University"]
description: "本文提出 MARL-CPC 框架，通过集体预测编码将通信嵌入状态推断过程，成功在非合作、去中心化多智能体强化学习环境中实现有效信息共享。"
---

> **Summary:** 本文提出 MARL-CPC 框架，通过集体预测编码将通信嵌入状态推断过程，成功在非合作、去中心化多智能体强化学习环境中实现有效信息共享。 

> **Keywords:** Multi-Agent RL, Emergent Communication, Predictive Coding, Decentralized Learning, State Inference

**Authors:** Naoto Yoshida, Tadahiro Taniguchi

**Institution(s):** Kyoto University, Research Organization of Science and Technology, Ritsumeikan University


## Problem Background

在多智能体强化学习（MARL）中，传统方法通常依赖中心化训练或假设智能体间合作，而在自然界中，智能体往往是去中心化、独立学习且奖励不一致的；本文针对这一问题，探索如何在非合作、奖励无关的去中心化环境中，让独立智能体自发形成有效通信机制，以提升群体决策性能。

## Method

* **核心思想**：提出 MARL-CPC 框架，基于集体预测编码（Collective Predictive Coding, CPC），将通信问题转化为全局状态推断问题，通过构建伪联合生成模型，将多智能体的观测和消息整合，并分解为每个智能体独立优化的目标函数，通信作为辅助变量支持状态估计，而非动作空间的一部分。
* **具体实现**：
  1. **CPC 模块设计**：采用变分推断方法，定义包含观测和消息的联合概率分布，通过证据下界（ELBO）分解为每个智能体的优化目标，利用直通梯度估计器处理离散消息的梯度计算。
  2. **消息采样与整合**：每个智能体基于变分分布独立采样消息，消息向量拼接为全局消息，作为后续强化学习模块的输入，辅助全局状态表征。
  3. **算法实现**：设计两种算法，Bandit-CPC 适用于上下文 bandit 问题，通过奖励直接优化策略；IPPO-CPC 结合近端策略优化（PPO），适用于复杂状态转移环境，通信模块与 RL 模块梯度分离，确保去中心化。
* **创新点**：通信学习不依赖奖励一致性，嵌入表征学习中，适用于非合作环境，与传统‘消息作为动作’的方法形成对比。

## Experiment

* **有效性**：在上下文 bandit 环境中，MARL-CPC 的群体福利接近完全信息共享的上限（接近 2.0），显著优于无通信和消息作为动作的 baseline；在 observer 环境中，CPC 智能体在群体福利和任务完成时间上均有显著提升，验证了非合作环境下的信息共享能力。
* **合理性**：实验覆盖简单（bandit）和复杂（observer）场景，测试了非合作和合作奖励结构，设置较为全面；消融实验表明 CPC 消息对性能至关重要，随机或移除消息后性能显著下降。
* **局限性**：实验规模较小（仅 2 个智能体），未探讨大规模系统的可扩展性；消息空间为离散值，可能限制通信表达能力。

## Further Thoughts

MARL-CPC 将通信嵌入表征学习中的思路启发我们思考，是否可以通过更复杂的生成模型（如扩散模型）提升通信表达能力；此外，非合作环境下的通信稳定性问题值得进一步研究，或许可以结合博弈论或进化算法设计更鲁棒的通信协议；同时，框架在异构智能体或大规模系统中的应用潜力也值得探索，可结合联邦学习思想设计分层通信机制。