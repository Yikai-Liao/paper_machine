---
title: "Can Generative AI agents behave like humans? Evidence from laboratory market experiments"
pubDatetime: 2025-05-12T11:44:46+00:00
slug: "2025-05-llm-market-simulation"
type: "arxiv"
id: "2505.07457"
score: 0.7041006479344938
author: "grok-3-latest"
authors: ["R. Maria del Rio-Chanona", "Marco Pangallo", "Cars Hommes"]
tags: ["LLM", "Behavioral Simulation", "Market Dynamics", "Feedback Mechanisms", "Bounded Rationality"]
institution: ["University College London", "Complexity Science Hub", "Bennett Institute for Public Policy, University of Cambridge", "CENTAI Institute", "Bank of Canada", "University of Amsterdam"]
description: "本文通过实验室市场实验框架，验证了大型语言模型在特定参数下（如记忆窗口≥3）能够部分模拟人类经济行为，特别是在正反馈和负反馈市场的动态差异上，为其在社会科学模拟中的应用奠定了基础。"
---

> **Summary:** 本文通过实验室市场实验框架，验证了大型语言模型在特定参数下（如记忆窗口≥3）能够部分模拟人类经济行为，特别是在正反馈和负反馈市场的动态差异上，为其在社会科学模拟中的应用奠定了基础。 

> **Keywords:** LLM, Behavioral Simulation, Market Dynamics, Feedback Mechanisms, Bounded Rationality

**Authors:** R. Maria del Rio-Chanona, Marco Pangallo, Cars Hommes

**Institution(s):** University College London, Complexity Science Hub, Bennett Institute for Public Policy, University of Cambridge, CENTAI Institute, Bank of Canada, University of Amsterdam


## Problem Background

传统经济学研究依赖实验室实验来理解人类行为，但这些实验成本高且难以大规模复制。
大型语言模型（LLMs）提供了一种低成本替代方案，然而，其在模拟动态市场交互和人类经济行为方面的能力尚未被充分验证。
本文聚焦于探索 LLMs 是否能在实验室市场实验中重现人类行为，特别是在正反馈和负反馈市场中的价格预期和动态行为。

## Method

*   **实验框架设计：** 基于 Heemeijer et al. (2009) 的实验室市场实验，模拟正反馈（价格预测上升导致实际价格上升）和负反馈（价格预测上升导致实际价格下降）两种市场类型，涉及 6 个智能体在 50 个时间步内预测价格，价格由平均预测和少量噪声决定。
*   **LLM 环境转化：** 通过 OpenAI API 将实验环境转化为 LLMs 模拟，测试 GPT-3.5 和 GPT-4 两种模型，设置不同的记忆窗口（Memory=1, 3, 5，决定智能体能回忆的过去时间步数）和响应变异性（Temperature=0.3, 0.7, 1.0，控制输出的随机性）。
*   **提示与交互设计：** 提供清晰的系统指令，模拟人类实验中的任务描述（如预测价格以最大化收益），并采用链式思维（Chain-of-Thought）技术，要求智能体先提供推理再输出预测值，输出格式为 JSON 结构以便数据处理。
*   **行为对齐测量：** 使用一阶启发式回归模型（First-Order Heuristic）分析 LLMs 和人类预测策略的相似性，重点评估趋势跟随（Trend Following）、天真预测（Naivety）等参数，并通过市场价格动态和个体策略分布进行比较。
*   **迭代模拟过程：** 在每个时间步，智能体接收市场信息（过去价格、自身预测、累计收益），基于上下文进行推理和预测，所有智能体预测后计算市场价格和收益，形成动态反馈循环。

## Experiment

*   **宏观行为相似性：** 在正反馈市场中，LLMs（如 GPT-3.5）与人类类似，表现出较大价格波动和趋势跟随行为，但波动速度较慢；GPT-4 更倾向于快速稳定到略高于均衡价格。在负反馈市场中，人类通常在 10 个时间步内收敛到均衡，而 GPT-3.5 需要约 25 步，GPT-4 更接近人类（10-15 步）。
*   **参数影响显著性：** 记忆窗口为 3 或 5 时，LLMs 行为更接近人类，收敛速度和趋势跟随参数（β）与人类数据对齐（如正反馈市场中 GPT-4 的 β 接近人类平均值 0.67）；温度较高（1.0）增加响应变异性，但影响不如记忆显著。
*   **行为异质性不足：** LLMs 的个体策略分布较人类集中，尤其在负反馈市场中表现出更多天真预测（依赖最近价格），缺乏人类的多集群行为模式（如趋势跟随与适应性预测的分离）。
*   **实验设置评价：** 实验设计较为全面，涵盖多种参数组合和两种市场类型，提供了系统性比较基础，但未深入探讨更复杂动态（如泡沫形成），且未充分考虑模型训练数据的潜在偏见对行为模拟的影响。

## Further Thoughts

论文中记忆窗口对 LLMs 行为的影响启发了我：是否可以通过动态调整上下文管理（如引入遗忘机制或选择性记忆）来提升模拟的真实性？此外，LLMs 行为异质性不足的问题是否可以通过引入个性特征提示（如风险偏好、投资风格）解决？更进一步，LLMs 在市场模拟中的应用是否可以扩展到其他社会科学领域，如政策制定或群体决策，通过调整参数和提示设计模拟不同文化背景下的行为模式？