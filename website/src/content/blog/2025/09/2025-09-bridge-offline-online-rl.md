---
title: "Fine-tuning Behavioral Cloning Policies with Preference-Based Reinforcement Learning"
pubDatetime: 2025-09-30T17:50:19+00:00
slug: "2025-09-bridge-offline-online-rl"
type: "arxiv"
id: "2509.26605"
score: 0.5768485829419489
author: "grok-3-latest"
authors: ["Maël Macuglia", "Paul Friedrich", "Giorgia Ramponi"]
tags: ["Reinforcement Learning", "Behavioral Cloning", "Preference Feedback", "Offline Learning", "Online Learning"]
institution: ["University of Zurich", "ETH AI Center"]
description: "本文提出 BRIDGE 算法，通过离线专家数据构建置信集并结合在线偏好反馈微调策略，首次为该混合范式提供理论遗憾界，证明离线数据显著降低在线学习遗憾。"
---

> **Summary:** 本文提出 BRIDGE 算法，通过离线专家数据构建置信集并结合在线偏好反馈微调策略，首次为该混合范式提供理论遗憾界，证明离线数据显著降低在线学习遗憾。 

> **Keywords:** Reinforcement Learning, Behavioral Cloning, Preference Feedback, Offline Learning, Online Learning

**Authors:** Maël Macuglia, Paul Friedrich, Giorgia Ramponi

**Institution(s):** University of Zurich, ETH AI Center


## Problem Background

强化学习（RL）在现实世界应用中面临两大挑战：探索过程危险且数据需求量大，以及奖励函数难以准确指定。
本文提出了一种两阶段框架，通过离线专家演示数据学习安全的初始策略，并结合在线偏好反馈进行微调，旨在避免危险探索并解决奖励误指定问题，同时理论上分析离线数据如何减少在线学习的复杂性。

## Method

*   **核心思想:** 通过离线专家演示数据构建高置信度的策略集，缩小在线偏好学习的搜索空间，从而提高学习效率并降低风险。
*   **具体步骤:** 
    *   **离线模仿学习:** 使用行为克隆（Behavioral Cloning, BC）从离线数据中学习初始策略，并通过最大似然估计（MLE）学习环境转移模型。
    *   **置信集构建:** 在轨迹分布空间中基于 Hellinger 距离构建一个围绕 BC 策略的置信集，其半径随离线数据量 n 增加而缩小（O(1/√n)），确保包含专家策略的高概率。
    *   **受限在线学习:** 在线阶段通过偏好反馈优化策略，探索范围限制在离线置信集内，同时结合偏好模型和转移模型的不确定性进行引导探索，选择最大化不确定性的策略对进行比较。
*   **理论支持:** 提供遗憾界（regret bound），证明离线数据量增加可显著降低在线遗憾，量化了离线与在线数据之间的权衡。

## Experiment

*   **有效性:** 在离散环境（StarMDP, Gridworld）和连续控制环境（Reacher, Ant）中，BRIDGE 算法的累积遗憾显著低于单独的离线行为克隆（BC）和在线偏好 RL（PbRL）基线，验证了结合离线和在线学习的优势。
*   **搜索效率:** BRIDGE 能更快精炼策略搜索空间，相比 PbRL 探索范围更小，显示出更高的样本效率。
*   **实验设置合理性:** 实验涵盖多种环境，进行了消融研究，分析了离线数据量、置信集半径、专家数据质量和嵌入函数选择的影响，设置全面且合理。
*   **局限性:** 置信集半径选择对性能敏感，过大或过小均影响效果；嵌入函数需针对环境定制，否则学习速度可能受限。

## Further Thoughts

论文通过 Hellinger 距离构建置信集并量化离线数据对在线学习的影响，这一思想可启发其他混合学习范式（如大语言模型微调）中数据效率的优化；偏好反馈替代奖励函数的灵活性适用于奖励难以定义的场景，未来可结合多模态数据扩展应用范围；不确定性引导探索的机制可在其他 RL 任务中平衡探索与利用，或许能通过动态调整置信集半径进一步优化性能。