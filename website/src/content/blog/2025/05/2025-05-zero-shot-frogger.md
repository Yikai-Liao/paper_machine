---
title: "Frog Soup: Zero-Shot, In-Context, and Sample-Efficient Frogger Agents"
pubDatetime: 2025-05-06T19:51:41+00:00
slug: "2025-05-zero-shot-frogger"
type: "arxiv"
id: "2505.03947"
score: 0.6072905788617103
author: "grok-3-latest"
authors: ["Xiang Li", "Yiyang Hao", "Doug Fulop"]
tags: ["LLM", "Reinforcement Learning", "Zero-Shot Learning", "In-Context Learning", "Sample Efficiency"]
institution: ["Stanford University"]
description: "本文展示了推理型 LLM 在零样本 Atari 游戏 Frogger 中的潜力，并通过 LLM 示范数据提升传统 DQN 智能体 35.3% 的样本效率。"
---

> **Summary:** 本文展示了推理型 LLM 在零样本 Atari 游戏 Frogger 中的潜力，并通过 LLM 示范数据提升传统 DQN 智能体 35.3% 的样本效率。 

> **Keywords:** LLM, Reinforcement Learning, Zero-Shot Learning, In-Context Learning, Sample Efficiency

**Authors:** Xiang Li, Yiyang Hao, Doug Fulop

**Institution(s):** Stanford University


## Problem Background

强化学习（RL）领域的一个核心目标是开发通用智能体，能够快速适应并掌握新任务。
传统 RL 游戏智能体在 Atari 游戏中表现优异，但训练成本高、时间长，尤其是在稀疏奖励、探索挑战和长期规划需求的游戏（如 Frogger）中效率低下。
论文探索是否可以利用预训练的大型语言模型（LLM），特别是经过 RL 后训练的推理模型，在零样本（Zero-Shot）设置下直接玩 Atari 游戏，并通过上下文学习和示范数据加速传统 RL 智能体的训练。

## Method

*   **零样本 LLM 智能体**：利用最新的推理型 LLM（如 o3-mini 和 QwQ-32B），在不进行特定游戏训练的情况下，基于对象中心表示（Object-Centric Representation）直接进行游戏决策。测试了不同上下文配置（如过去步骤数量 0、3、所有，是否显示奖励）以及推理努力程度（低、中、高）对性能的影响。
*   **上下文学习（In-Context Learning）**：通过向 LLM 提供过去的状态、动作和奖励信息，使其在游戏过程中动态调整策略，探索上下文大小和奖励信息对决策改进的作用。
*   **探索与反思智能体**：设计探索型（Explorative）智能体，提示其探索环境而非追求最优动作；设计反思型（Reflective）智能体，基于 Reflexion 框架，在每局游戏后反思策略并改进后续决策。
*   **LLM 引导的 DQN**：受 DQfD 启发，使用 LLM 生成的游戏示范数据初始化优先经验回放缓冲区（Prioritized Experience Replay, PER），引导传统 DQN 智能体的训练，减少随机探索阶段的低效性，加速学习过程。

## Experiment

*   **零样本 LLM 性能**：o3-mini 在零样本设置下（中等推理努力，0 过去步骤）取得最高 32 分回合奖励，接近 Frogger 游戏 12 个车道中的第 10 个车道；QwQ-32B 取得 17 分。增加过去步骤数量反而降低性能，推理努力与性能呈弱正相关（r=0.355），但对提示策略敏感。
*   **上下文学习效果**：高推理努力下，显示过去奖励显著提升性能（如 0 过去步骤配置下奖励从 22 分提升至 45 分），但低推理努力下可能导致偏向次优动作。
*   **探索与反思局限**：探索型智能体性能提升有限（奖励从 20 分到 21 分），反思型智能体因反馈高层次而性能下降（从 45 分到 21 分），显示 LLM 在长序列规划和细节分析上的不足。
*   **LLM 引导 DQN 提升**：在 5000 回合训练后，标准 DQN 平均奖励 15 分，LLM 引导 DQN 达 24 分，提升 35.3%，且早期收敛更快。实验在资源受限环境下进行，设置合理但训练回合数可能不足以完全验证长期效果。

## Further Thoughts

LLM 与 RL 的结合展示了高层次策略指导与低层次控制互补的潜力，未来可探索动态调整 LLM 示范数据优先级以适应 RL 训练阶段需求；上下文学习对提示敏感，启发设计更鲁棒的提示框架或通过元学习优化上下文选择；反思型智能体失败提示引入细粒度反馈机制或结合蒙特卡洛树搜索（MCTS）以改进长期规划。