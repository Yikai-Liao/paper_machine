---
title: "Learning to Walk with Less: a Dyna-Style Approach to Quadrupedal Locomotion"
pubDatetime: 2025-09-08T02:48:23+00:00
slug: "2025-09-dyna-style-locomotion"
type: "arxiv"
id: "2509.06296"
score: 0.5403498578463554
author: "grok-3-latest"
authors: ["Francisco Affonso", "Felipe Andrade G. Tommaselli", "Juliano Negri", "Vivian S. Medeiros", "Mateus V. Gasparino", "Girish Chowdhary", "Marcelo Becker"]
tags: ["Reinforcement Learning", "Model-Based RL", "Data Augmentation", "Locomotion Control", "Sample Efficiency"]
institution: ["University of São Paulo (EESC-USP)", "University of Illinois at Urbana-Champaign (UIUC)"]
description: "本文提出一种基于 Dyna-Style 的模型基强化学习框架，通过预测模型生成合成数据增强 PPO 算法训练过程，显著提高四足机器人运动控制的样本效率并减少模拟数据需求。"
---

> **Summary:** 本文提出一种基于 Dyna-Style 的模型基强化学习框架，通过预测模型生成合成数据增强 PPO 算法训练过程，显著提高四足机器人运动控制的样本效率并减少模拟数据需求。 

> **Keywords:** Reinforcement Learning, Model-Based RL, Data Augmentation, Locomotion Control, Sample Efficiency

**Authors:** Francisco Affonso, Felipe Andrade G. Tommaselli, Juliano Negri, Vivian S. Medeiros, Mateus V. Gasparino, Girish Chowdhary, Marcelo Becker

**Institution(s):** University of São Paulo (EESC-USP), University of Illinois at Urbana-Champaign (UIUC)


## Problem Background

四足机器人运动控制中，传统的基于强化学习（RL）的控制器面临数据效率低下的问题，需要大量环境交互数据来实现稳健性能，这导致训练成本高昂且耗时。本文旨在通过模型基强化学习（MBRL）框架，减少对模拟数据的依赖，同时保持或提升运动控制性能，解决数据效率低这一关键问题。

## Method

* **核心思想：** 基于 Dyna-Style 范式，提出一种模型基强化学习（MBRL）框架，通过一个预测模型生成合成数据（synthetic data），在不增加模拟数据需求的情况下增强训练过程，结合近端策略优化（PPO）算法提高数据效率。
* **具体实现：** 
  * 构建一个多层感知机（MLP）作为预测模型，与策略同时训练，用于近似机器人运动动力学关系，预测下一状态和奖励。
  * 在每次策略更新迭代中，从模拟环境中收集部分数据（simulated data），然后利用预测模型生成合成数据，逐步替换部分模拟数据，保持总 rollout 长度不变。
  * 引入调度策略（scheduler），根据训练迭代次数动态调整合成数据比例，避免初期模型不准确导致的学习不稳定，确保合成数据逐步融入训练过程。
* **关键点：** 合成数据仅用于扩展 rollout 的后半部分，以减少对基线 RL 控制器的干扰；预测模型采用短时预测（short-horizon prediction）以提高准确性；不直接修改策略决策过程，而是通过数据增强间接提升学习效率。

## Experiment

* **有效性：** 在 Unitree Go1 机器人模拟环境中，相比基线方法（纯 PPO 算法），本文方法通过合成数据扩展 rollout 长度，显著减少了达到目标性能所需的模拟数据量（最高减少 42.4%），同时提升了策略回报（policy return），并降低了训练过程中的方差。
* **优越性：** 消融研究表明，适当的 rollout 长度对性能至关重要，而本文方法通过合成数据模拟更长 rollout，实现了与增加模拟数据相似的效果，且在跟踪多种运动命令时表现出更低的跟踪误差，表明泛化能力更强。
* **实验设置：** 实验基于 Isaac Gym 模拟器，涵盖了不同 rollout 长度和合成数据比例的对比，设置较为全面合理；但缺乏真实机器人上的验证，可能存在模拟到现实的迁移问题。
* **开销：** 生成合成数据和训练预测模型引入了少量额外计算成本，但整体训练时间与基线方法相近（约 20.83 小时），表明方法在效率上的可行性。

## Further Thoughts

Dyna-Style 框架通过合成数据增强训练过程的思路具有广泛适用性，未来可探索将其应用于其他高数据需求的 RL 任务（如机器人操作或自动驾驶）；此外，预测模型的性能对合成数据质量至关重要，可以尝试更复杂的模型结构（如 Transformer）或结合真实数据预训练来提升准确性；另外，是否可以根据任务难度或学习阶段动态调整合成数据生成策略，以更好地匹配策略分布？