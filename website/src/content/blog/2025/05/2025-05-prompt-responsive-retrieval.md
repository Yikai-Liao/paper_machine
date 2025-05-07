---
title: "Prompt-responsive Object Retrieval with Memory-augmented Student-Teacher Learning"
pubDatetime: 2025-05-04T19:51:09+00:00
slug: "2025-05-prompt-responsive-retrieval"
type: "arxiv"
id: "2505.02232"
score: 0.5458617416757529
author: "grok-3-latest"
authors: ["Malte Mosbach", "Sven Behnke"]
tags: ["Foundation Model", "Reinforcement Learning", "Prompt Response", "Student-Teacher Learning", "Memory Augmentation"]
institution: ["University of Bonn", "Lamarr Institute for Machine Learning and Artificial Intelligence"]
description: "本文提出记忆增强的学生-教师学习框架，将基础模型的提示响应能力与强化学习的灵巧控制能力结合，实现了在杂乱场景中基于提示的目标物体拾取。"
---

> **Summary:** 本文提出记忆增强的学生-教师学习框架，将基础模型的提示响应能力与强化学习的灵巧控制能力结合，实现了在杂乱场景中基于提示的目标物体拾取。 

> **Keywords:** Foundation Model, Reinforcement Learning, Prompt Response, Student-Teacher Learning, Memory Augmentation

**Authors:** Malte Mosbach, Sven Behnke

**Institution(s):** University of Bonn, Lamarr Institute for Machine Learning and Artificial Intelligence


## Problem Background

本文聚焦于机器人如何基于用户提示（prompt）执行灵巧操作任务，如在杂乱场景中拾取特定物体。
当前，基础模型（Foundation Models, FMs）如 GPT-4 和 Segment Anything (SAM) 在高层次规划中表现出色，但难以直接应用于需要低层次灵巧控制的任务；强化学习（RL）虽能学习精细控制，但策略通常任务特定，缺乏提示适应性。
作者试图解决的关键问题是：如何整合基础模型的开放词汇能力和提示响应性与强化学习的低层次控制能力，以实现提示响应的灵巧操作。

## Method

*   **核心思想:** 提出一种记忆增强的学生-教师学习框架，将基础模型的感知能力与强化学习的控制能力结合，通过特权信息训练教师策略，再蒸馏到依赖不完美感知的学生策略，实现提示响应的灵巧操作。
*   **感知骨干:** 使用 Segment Anything 2 (SAM 2) 模型根据用户提示分割目标物体，尽管其检测结果不完美（存在遮挡或分割不稳定），但时间序列输出可用于隐式状态估计。
*   **教师策略训练:** 在模拟环境中，利用特权信息（如模拟器中的精确状态）通过强化学习（PPO 算法）训练教师策略，学习最优控制策略，适应多种物体形状和大小。
*   **学生策略蒸馏:** 通过模仿学习（DAgger 算法）将教师知识转移到学生策略，学生策略仅依赖 SAM 2 的不完美检测和机器人本体感知（Proprioception），并采用记忆增强架构（如 LSTM、Transformer）处理历史观测序列，隐式推断目标物体真实状态。
*   **工程优化:** 设计自动提示生成机制，利用模拟环境中的目标物体几何信息生成提示，确保训练效率；同时对 SAM 2 进行批处理优化，支持并行环境检测。
*   **关键创新:** 将感知与控制解耦，通过记忆增强机制弥补不完美感知的缺陷，实现基于提示的灵巧操作。

## Experiment

*   **教师策略效果:** 在模拟环境中，教师策略基于特权信息表现出色，桌面场景中 LSTM 架构的拾取成功率高达 94.2%，容器场景中接近 90%，表明强化学习在理想条件下能学习高效控制策略。
*   **学生策略效果:** 学生策略在仅依赖 SAM 2 检测的情况下，成功率略低于教师策略，但仍表现良好（桌面场景中 Transformer 架构拾取成功率达 88.3%），表明通过历史观测能有效推断目标状态。
*   **架构对比:** 记忆增强架构（LSTM、Transformer）优于简单 1D-CNN，特别是在较长上下文长度下，模仿损失显著降低，验证了历史信息对状态推断的重要性。
*   **真实机器人迁移:** 真实机器人测试中，策略对训练和测试物体均有较好泛化能力，成功率在 40%-60% 之间，表明方法具有一定零样本迁移能力，但受限于模拟到现实的差距。
*   **实验设置合理性:** 实验涵盖多种场景（桌面、容器）、多种物体（YCB 数据集 60 个物体，48 个训练，12 个测试）、多种架构（MLP、LSTM、Transformer），设置全面，但真实场景成功率较低，未深入探讨模拟到现实差距问题。

## Further Thoughts

学生-教师框架提供了一种将基础模型与强化学习结合的通用思路，可扩展至自动驾驶或医疗机器人等领域；记忆增强机制在处理非马尔可夫观测中的作用启发我们在时间序列任务中利用历史信息推断隐式状态；SAM 2 作为感知骨干的成功应用表明视觉基础模型在机器人任务中的潜力，未来可探索多模态基础模型以提升提示响应灵活性；自动提示生成机制的工程创新提示我们在真实场景中探索类似自动化提示生成方法。