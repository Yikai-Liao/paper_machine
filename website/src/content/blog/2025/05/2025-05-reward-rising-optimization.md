---
title: "RRO: LLM Agent Optimization Through Rising Reward Trajectories"
pubDatetime: 2025-05-27T05:27:54+00:00
slug: "2025-05-reward-rising-optimization"
type: "arxiv"
id: "2505.20737"
score: 0.6078202310965273
author: "grok-3-latest"
authors: ["Zilong Wang", "Jingfeng Yang", "Sreyashi Nag", "Samarth Varshney", "Xianfeng Tang", "Haoming Jiang", "Jingbo Shang", "Sheikh Muhammad Sarwar"]
tags: ["LLM", "Reinforcement Learning", "Process Supervision", "Sampling Strategy", "Multi-Step Reasoning"]
institution: ["University of California, San Diego", "Amazon"]
description: "本文提出奖励上升优化（RRO）方法，通过动态探索和关注相对奖励趋势，显著提升大型语言模型智能体在多步任务中的性能，同时大幅降低计算成本。"
---

> **Summary:** 本文提出奖励上升优化（RRO）方法，通过动态探索和关注相对奖励趋势，显著提升大型语言模型智能体在多步任务中的性能，同时大幅降低计算成本。 

> **Keywords:** LLM, Reinforcement Learning, Process Supervision, Sampling Strategy, Multi-Step Reasoning

**Authors:** Zilong Wang, Jingfeng Yang, Sreyashi Nag, Samarth Varshney, Xianfeng Tang, Haoming Jiang, Jingbo Shang, Sheikh Muhammad Sarwar

**Institution(s):** University of California, San Diego, Amazon


## Problem Background

大型语言模型（LLMs）在复杂多步任务中作为智能体时，常常因关键步骤的细微错误导致任务失败，现有通过强化学习和过程奖励模型（Process Reward Models, PRMs）校准推理过程的方法面临扩展性挑战：为每个步骤探索大量候选动作以获取训练数据，计算成本极高；本文旨在解决如何在降低计算开销的同时，提升 LLM 智能体在多步任务中的推理和决策能力。

## Method

* **核心思想**：提出奖励上升优化（Reward Rising Optimization, RRO），通过关注连续推理步骤之间的相对奖励趋势（relative reward trend），动态调整探索范围，构建高质量训练数据以优化 LLM 智能体性能。
* **具体实现**：方法分为三个阶段：
  * **监督微调（Supervised Fine-Tuning, SFT）**：基于专家轨迹数据集对基础模型进行微调，使其具备初步的任务规划能力，确保模型能够理解任务格式和基本决策逻辑。
  * **奖励上升采样（Reward Rising Sampling）**：在探索下一个动作候选时，采用动态采样策略，持续采样直到找到一个过程奖励高于前一步的动作候选（即‘上升奖励’），从而构建偏好数据对（preferred and rejected action pairs）；此过程通过蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）估计过程奖励，避免了过度探索或不足探索，显著降低计算成本。
  * **智能体优化（Agent Optimization）**：基于直接偏好优化（Direct Preference Optimization, DPO）框架，利用采样得到的偏好数据对模型进行进一步训练，通过最大化偏好响应的对数概率比，增强模型与任务目标的对齐。
* **理论支持**：论文通过数学推导证明，至少存在一个动作候选，其过程奖励不低于前一步的奖励，为动态采样的可行性提供了理论依据。
* **关键优势**：RRO 不追求绝对最高奖励，而是通过局部奖励改进实现整体性能提升，平衡了探索质量和计算效率。

## Experiment

* **性能提升**：在 WebShop 和 InterCode-SQL 两个基准数据集上，RRO 基于 Gemma-2 2B 模型取得了最高奖励得分（WebShop: 62.91, InterCode-SQL: 55.08），相比次优方法分别提升了 1.52 和 0.40，表明其在多步任务中的优越性。
* **采样效率**：RRO 显著降低了采样数量（WebShop: 1.86, InterCode-SQL: 1.64），相比其他过程监督方法（如 IPR 和 Fixed-sized Exploration）所需的 3-5 个采样轨迹，效率提升明显，计算成本大幅减少。
* **奖励趋势分析**：RRO 优化后的模型在轨迹的不同阶段（初始、中间、最终）表现出更高的‘上升奖励’比例，尤其在 InterCode-SQL 上提升显著（最终阶段从 40.87% 提升到 45.29%），验证了方法在优化推理轨迹上的有效性。
* **实验设置合理性**：实验对比了从无后训练到结果监督和过程监督的多种基线方法，数据集涵盖网页导航和数据库查询两种典型多步任务，硬件配置（8 个 NVIDIA A100-80G GPU）支持了大规模实验需求，整体设计全面，结果可信。

## Further Thoughts

RRO 的‘上升奖励’标准提供了一种新颖的探索策略，启发我们思考是否可以通过自适应算法进一步优化奖励阈值，例如根据任务难度或轨迹阶段动态调整‘上升’的定义，以适应更广泛的任务场景；此外，这种关注相对奖励趋势而非绝对值的思路，可以推广到其他强化学习领域，如游戏 AI 或机器人控制中的策略优化，可能带来更高效的局部改进机制。