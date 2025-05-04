---
title: "MULE: Multi-terrain and Unknown Load Adaptation for Effective Quadrupedal Locomotion"
pubDatetime: 2025-05-01T12:41:35+00:00
slug: "2025-05-adaptive-quadruped-locomotion"
type: "arxiv"
id: "2505.00488"
score: 0.277596833841901
author: "grok-3-latest"
authors: ["Vamshi Kumar Kurva", "Shishir Kolathaya"]
tags: ["Reinforcement Learning", "Adaptive Control", "Quadrupedal Locomotion", "Payload Adaptation", "Terrain Adaptation"]
institution: ["Indian Institute of Science, Bengaluru"]
description: "本文提出一种基于强化学习的自适应控制框架，通过名义策略和自适应策略的协同工作，使四足机器人在未知负载和多样化地形下实现鲁棒运动控制，并在模拟与硬件实验中验证了其优越性。"
---

> **Summary:** 本文提出一种基于强化学习的自适应控制框架，通过名义策略和自适应策略的协同工作，使四足机器人在未知负载和多样化地形下实现鲁棒运动控制，并在模拟与硬件实验中验证了其优越性。 

> **Keywords:** Reinforcement Learning, Adaptive Control, Quadrupedal Locomotion, Payload Adaptation, Terrain Adaptation

**Authors:** Vamshi Kumar Kurva, Shishir Kolathaya

**Institution(s):** Indian Institute of Science, Bengaluru


## Problem Background

四足机器人在负载运输任务中具有重要应用潜力，但现有方法（如基于模型预测控制 MPC）在面对未知负载和多样化地形时，依赖预定义步态调度或轨迹生成器，缺乏实时适应性，尤其是在非结构化环境中。
负载变化会显著影响机器人动态参数（如质量、质心、惯性），而传统方法难以有效应对这些变化，因此需要一种能够动态适应负载和地形的控制框架。

## Method

*   **核心思想:** 提出一种基于强化学习（RL）的自适应控制框架，通过将控制策略分为名义策略（Nominal Policy）和自适应策略（Adaptive Policy），实现四足机器人在未知负载和复杂地形下的动态适应。
*   **具体实现:**
    *   **名义策略:** 在无负载条件下训练，负责基本运动控制和命令跟踪（如速度和基座高度），使用 Proximal Policy Optimization (PPO) 算法优化。
    *   **自适应策略:** 在负载变化条件下训练，学习提供校正动作（Corrective Actions）以补偿负载引起的动态扰动，特别关注基座高度的维持和稳定性。
    *   **两阶段训练:** 第一阶段仅训练名义策略，优化基本运动能力；第二阶段同时训练两策略，引入动态负载变化（如随机质量物体）模拟真实扰动，自适应策略通过增强观测空间（包括估计的足部接触力 GRF）感知负载变化。
    *   **奖励设计:** 自适应策略使用专门奖励函数（如 GRF 跟踪奖励），鼓励在负载增加时生成更大接触力以维持目标高度；名义策略则关注速度跟踪和稳定性。
    *   **技术细节:** 不依赖显式负载参数估计，而是通过学习直接调整动作；使用上下文估计网络（CE Net）编码观测历史提取潜在特征；动作输出为关节位置，通过 PD 控制器跟踪。
*   **关键优势:** 避免传统方法对步态切换的依赖，提高非结构化地形上的鲁棒性，同时通过模块化设计（基本控制+扰动适应）实现灵活性。

## Experiment

*   **有效性:** 在 Isaac Gym 模拟环境和 Unitree Go1 真实机器人上进行测试，自适应控制器在平地、斜坡和楼梯等多种地形下，面对静态和动态负载变化（0-10 kg）时，显著优于基线控制器（基于 DreamWaQ），特别是在高度和速度跟踪准确性上，例如在楼梯场景中基线控制器在高负载下停止运动，而自适应控制器仍保持稳定。
*   **合理性:** 实验设置全面，涵盖多种地形和负载条件，模拟中引入动态负载变化（每4秒随机调整质量）以模拟真实不确定性，硬件实验通过自由移动铁球验证动态适应能力。
*   **局限性:** 计算开销（如两阶段训练和实时策略计算）未详细讨论，可能影响实际部署；负载范围（最高10 kg）相对机器人自身重量（约12 kg）有限，极端负载下的表现未充分探讨。

## Further Thoughts

本文的双策略设计（Nominal + Adaptive）具有启发性，将基本控制与扰动适应分离的模块化思想可推广至其他机器人任务（如机械臂或无人机控制），通过‘基本策略+校正策略’应对外部扰动；此外，自适应策略通过增强观测（如足部力）感知扰动而非显式建模的‘感知-学习-适应’思路，对处理复杂动态系统和未知环境适应问题提供了新视角。