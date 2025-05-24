---
title: "From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning"
pubDatetime: 2025-05-21T15:00:07+00:00
slug: "2025-05-pedagogical-rl-tutoring"
type: "arxiv"
id: "2505.15607"
score: 0.8422083643746894
author: "grok-3-latest"
authors: ["David Dinucu-Jianu", "Jakub Macina", "Nico Daheim", "Ido Hakimi", "Iryna Gurevych", "Mrinmaya Sachan"]
tags: ["LLM", "Reinforcement Learning", "Pedagogy", "Synthetic Data", "Multi-Turn Dialog"]
institution: ["ETH Zurich", "ETH AI Center", "Ubiquitous Knowledge Processing Lab, TU Darmstadt", "Hessian Center for AI"]
description: "本文提出了一种在线强化学习框架，通过合成学生-导师交互和可调奖励函数，将大型语言模型对齐到教学目标，使小型模型在教学质量和学生解题准确率上接近大型专有模型。"
---

> **Summary:** 本文提出了一种在线强化学习框架，通过合成学生-导师交互和可调奖励函数，将大型语言模型对齐到教学目标，使小型模型在教学质量和学生解题准确率上接近大型专有模型。 

> **Keywords:** LLM, Reinforcement Learning, Pedagogy, Synthetic Data, Multi-Turn Dialog

**Authors:** David Dinucu-Jianu, Jakub Macina, Nico Daheim, Ido Hakimi, Iryna Gurevych, Mrinmaya Sachan

**Institution(s):** ETH Zurich, ETH AI Center, Ubiquitous Knowledge Processing Lab, TU Darmstadt, Hessian Center for AI


## Problem Background

大型语言模型（LLMs）在教育领域具有作为个性化导师的潜力，但其优化目标通常是直接提供答案，而非通过引导学生独立解决问题来促进学习，与有效的教育学原则相悖。
论文提出‘教学对齐（Pedagogical Alignment）’的概念，旨在将 LLMs 从‘助手’转变为‘导师’，解决如何在不依赖昂贵人工标注或大型模型的前提下，训练 LLMs 成为有效导师，同时平衡教学质量和学生解题准确率的关键问题。

## Method

*   **核心思想:** 通过在线强化学习（Reinforcement Learning, RL）框架，将 LLMs 对齐到教学目标，利用模拟的学生-导师多轮对话，训练模型在不泄露答案的情况下引导学生解决问题。
*   **具体实现:** 
    *   **合成数据交互:** 使用合成学生-导师对话数据，避免依赖昂贵的人工标注，通过模拟多轮交互（如 Socratic 提问和针对性提示）来训练导师模型。
    *   **奖励函数设计:** 奖励函数综合考虑两方面：一是学生在对话后的解题成功率（Post-Dialog Solve Rate），通过多次采样学生答案与真实解对比计算；二是教学质量（Pedagogical Quality），由多个 LLM 评判模型评估对话是否符合教学原则（如避免答案泄露、提供建设性指导）。
    *   **可调参数平衡:** 引入惩罚参数（Penalty λ），在奖励函数中平衡解题成功率和教学质量，允许动态调整模型行为以适应不同教学目标。
    *   **优化算法:** 采用 Group Relative Policy Optimization (GRPO) 算法，基于完整对话的奖励优化模型，而非单轮反馈，确保长期教学目标的实现。
    *   **Thinking Tags 机制:** 导师模型可在隐藏标签中规划教学策略（如分析学生错误或设计引导步骤），提升模型响应的针对性和可解释性。
*   **关键特点:** 采用‘On-Policy’在线 RL 方法，模型从自身生成的对话中学习，避免传统离线数据（如 SFT 或 DPO）带来的上下文偏差，同时支持多轮对话动态调整。

## Experiment

*   **有效性:** 在 BigMath 数据集上，训练的 7B 参数模型（Qwen2.5-7B）通过调整惩罚参数 λ，在学生解题成功率（∆ Solve Rate）和答案泄露率（Leak Solution Rate）之间实现了动态平衡，例如 λ=0.75 时，解题率提升 25.3%，泄露率仅 10.6%，教学质量评分（Ped-RM）达 3.9/3.2。
*   **优越性:** 相比基线模型（如 SocraticLM、SFT、MDPO），RL 方法显著降低答案泄露率并提升教学质量；与大型专有模型（如 LearnLM）相比，7B 模型在解题率上接近甚至更优，同时保持较低泄露率。
*   **推理能力保持:** 在通用推理基准（如 MMLU、GSM8K、MATH500）上，RL 模型未出现显著性能下降，优于 SFT 等方法，表明教学对齐未牺牲核心能力。
*   **实验设置合理性:** 实验涵盖领域内（BigMath）和领域外（MathTutorBench）测试，使用多指标评估教学效果和推理能力，但局限在于仅聚焦数学领域，且学生模型单一，未完全模拟真实学习者多样性。

## Further Thoughts

在线 RL 与合成数据结合的方式，为低成本部署教育 AI 提供了新思路，特别是在资源受限场景下；奖励函数中可调参数（λ）的设计，启发我们在多目标优化任务（如医疗对话）中探索动态平衡机制；‘Thinking Tags’机制可扩展至其他需要内部规划的领域（如复杂推理），以提升模型透明度和用户信任。