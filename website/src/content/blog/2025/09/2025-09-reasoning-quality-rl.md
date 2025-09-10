---
title: "Rethinking Reasoning Quality in Large Language Models through Enhanced Chain-of-Thought via RL"
pubDatetime: 2025-09-07T11:52:18+00:00
slug: "2025-09-reasoning-quality-rl"
type: "arxiv"
id: "2509.06024"
score: 0.729472151172774
author: "grok-3-latest"
authors: ["Haoyang He", "Zihua Rong", "Kun Ji", "Chenyang Li", "Qing Huang", "Chong Xia", "Lan Yang", "Honggang Zhang"]
tags: ["LLM", "Reinforcement Learning", "Chain of Thought", "Reasoning", "Benchmark"]
institution: ["Beijing University of Posts and Telecommunications"]
description: "本文提出动态推理效率奖励（DRER）框架和 LogicTree 数据集，通过强化学习优化链式推理质量，显著提升大型语言模型在形式逻辑推理任务上的准确率和一致性。"
---

> **Summary:** 本文提出动态推理效率奖励（DRER）框架和 LogicTree 数据集，通过强化学习优化链式推理质量，显著提升大型语言模型在形式逻辑推理任务上的准确率和一致性。 

> **Keywords:** LLM, Reinforcement Learning, Chain of Thought, Reasoning, Benchmark

**Authors:** Haoyang He, Zihua Rong, Kun Ji, Chenyang Li, Qing Huang, Chong Xia, Lan Yang, Honggang Zhang

**Institution(s):** Beijing University of Posts and Telecommunications


## Problem Background

当前大型语言模型（LLMs）在推理能力上的强化学习（RL）方法主要依赖基于最终答案正确性的规则奖励函数，忽略了链式推理（Chain-of-Thought, CoT）过程的质量，无法判断推理步骤是否真正有助于正确答案；此外，现有训练数据多集中于数学和编程领域，缺乏针对纯形式逻辑推理的统一数据集，导致模型逻辑推理能力的真实水平被高估。
论文旨在解决如何通过改进奖励机制提升推理链质量，并提供专注于形式逻辑推理的基准数据集，以更准确地评估和训练模型推理能力。

## Method

*   **核心思想:** 提出动态推理效率奖励（Dynamic Reasoning Efficiency Reward, DRER）框架，通过强化学习改进链式推理质量，结合奖励信号和优势调整，鼓励生成真正有助于正确答案的推理步骤，同时控制输出长度以提高训练稳定性。
*   **具体实现:** 
    *   **推理质量奖励（Reasoning Quality Reward）:** 通过计算模型在有CoT和无CoT两种情况下对正确答案的置信度差异（log-likelihood margin），为那些显著提升正确答案概率的推理链分配更高奖励，使用 tanh 函数平滑处理奖励值以确保数值稳定性，并将其与任务奖励（如答案正确性）加权组合，纳入整体强化学习目标。
    *   **动态长度优势（Dynamic Length Advantage）:** 根据验证集响应长度的统计分布（5%和95%分位数），对偏离合理长度的响应（过短或过长）施加指数衰减惩罚，调整优势估计值，避免冗长或无效推理链，确保训练过程稳定。
*   **数据集支持:** 构建 LogicTree 数据集，基于七种经典推理规则和四种命题逻辑，动态生成具有可控难度的形式逻辑推理问题，避免语义依赖，确保评估和训练专注于纯逻辑推理。
*   **关键特点:** DRER 是一个即插即用的框架，不依赖价值模型，可直接集成到现有强化学习流程（如 DAPO 和 GRPO），通过 token 级别的奖励信号精细优化推理行为。

## Experiment

*   **有效性:** 使用 Qwen2.5-7B-Instruct-1M 模型在 LogicTree 数据集（9600个问题，8个推理深度）上训练400步，DRER 框架将准确率从13%提升至60%，在最难的深度8问题上仍保持31%准确率，优于许多更大规模模型（如 GPT-o3-mini 的18%平均准确率）；答案置信度提升约60%，token 消耗减少75%。
*   **优越性:** 相比基线方法 GRPO 和 DAPO，DRER 在准确率、收敛速度和推理质量上均有显著优势，生成的CoT更有效（通过置信度差异验证），动态长度优势机制有效控制了响应长度。
*   **全面性与合理性:** 实验设置覆盖不同推理深度和逻辑结构，评估指标包括准确率、一致性比率（逻辑等价问题稳定性）和 Fβ 分数（平衡回答率和精度），较为全面；跨数据集泛化性测试（ZebraLogic、ProntoQA、AIME24 等）显示一定提升，表明方法普适性。
*   **不足与开销:** 实验主要基于7B模型，未测试更大规模模型的效果，token 级别奖励可能增加计算成本；LogicTree 为合成数据集，与真实世界推理场景可能存在偏差。

## Further Thoughts

DRER 的推理质量奖励机制通过置信度差异评估推理步骤贡献，这一思想可扩展至多模态推理或对话系统，评估中间步骤对最终输出的影响；LogicTree 的动态构建方式启发在法律或医学推理中设计类似可控难度数据集；动态长度优势机制提示在强化学习中引入更多统计分布约束，如控制推理复杂性；逻辑一致性挑战启发探索通过元学习或规则嵌入增强模型对逻辑结构的理解，而非仅依赖数据模式。