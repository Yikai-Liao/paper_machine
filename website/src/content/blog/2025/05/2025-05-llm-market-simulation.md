---
title: "Can Generative AI agents behave like humans? Evidence from laboratory market experiments"
pubDatetime: 2025-05-12T11:44:46+00:00
slug: "2025-05-llm-market-simulation"
type: "arxiv"
id: "2505.07457"
score: 0.7371192868993719
author: "grok-3-latest"
authors: ["R. Maria del Rio-Chanona", "Marco Pangallo", "Cars Hommes"]
tags: ["LLM", "Behavioral Simulation", "Market Dynamics", "Feedback Mechanisms", "Bounded Rationality"]
institution: ["University College London", "Complexity Science Hub", "Bennett Institute for Public Policy", "CENTAI Institute", "Bank of Canada", "University of Amsterdam"]
description: "本文通过将实验室市场实验翻译到 LLM 环境中，展示了大型语言模型在特定参数下能部分模拟人类经济行为，尤其在正反馈和负反馈市场中重现趋势跟随和收敛动态，为低成本行为模拟提供了新工具。"
---

> **Summary:** 本文通过将实验室市场实验翻译到 LLM 环境中，展示了大型语言模型在特定参数下能部分模拟人类经济行为，尤其在正反馈和负反馈市场中重现趋势跟随和收敛动态，为低成本行为模拟提供了新工具。 

> **Keywords:** LLM, Behavioral Simulation, Market Dynamics, Feedback Mechanisms, Bounded Rationality

**Authors:** R. Maria del Rio-Chanona, Marco Pangallo, Cars Hommes

**Institution(s):** University College London, Complexity Science Hub, Bennett Institute for Public Policy, CENTAI Institute, Bank of Canada, University of Amsterdam


## Problem Background

本文探讨了大型语言模型（LLMs）是否能在经济市场实验中模拟人类行为，特别是在动态反馈环境中，多个代理的决策随时间相互影响市场价格。
传统实验室实验成本高昂且难以大规模复制，而 LLMs 提供了一种低成本替代方案，关键问题在于评估 LLMs 是否能准确重现个体行为和市场互动，尤其是在正反馈（价格预期上升导致需求增加）和负反馈（价格预期上升导致供给增加）市场中的表现。

## Method

*   **实验设计**：基于 Heemeijer 等人（2009）的实验室市场实验，模拟正反馈和负反馈两种市场类型，每种市场包含 6 个代理（LLM 或人类），进行 50 个时间步的预测任务，目标是预测下一时间步的市场价格以最大化收益。
*   **LLM 环境翻译**：通过 OpenAI API 使用 GPT-3.5 和 GPT-4 模型，将实验指令和交互过程翻译为文本提示，模拟市场动态，代理在每个时间步接收历史价格、自身预测和累计收益信息，并输出下一时间步的价格预测。
*   **参数调整**：调整两个关键参数以影响 LLM 行为：上下文窗口（Memory，设置为 1、3、5 步）决定代理能记住的历史信息量；温度（Temperature，设置为 0.3、0.7、1.0）控制输出随机性和多样性，温度越高，预测越具异质性。
*   **提示与推理**：采用清晰的系统提示和链式思维（Chain-of-Thought）技术，要求代理先提供推理（30-50 字）再输出预测，确保其决策过程接近人类参与者的任务目标，并以 JSON 格式结构化输出以便数据处理。
*   **对齐度量**：使用一阶启发式回归模型（First-Order Heuristic）分析代理的预测策略，量化其是否表现出趋势跟随（Trend Following）、基本面主义（Fundamentalism）、天真预测（Naivety）等行为模式，并通过市场动态和行为参数估计比较 LLM 与人类的对齐程度。

## Experiment

*   **有效性**：在记忆 ≥ 3 和较高温度（如 1.0）设置下，LLM 代理能够部分重现人类行为：在正反馈市场中表现出趋势跟随行为，价格波动较大，类似人类实验中的泡沫现象；在负反馈市场中最终趋于均衡价格，但 LLM（尤其是 GPT-3.5）收敛速度较慢，振荡更多。
*   **优越性与差异**：GPT-4 比 GPT-3.5 更接近人类行为，尤其在负反馈市场中收敛速度更快（10-15 步 vs. 25 步），但 LLM 代理整体行为异质性低于人类，缺乏人类中明显的策略分化（如纯趋势跟随者与适应性预测者的集群）。
*   **实验设置合理性**：实验设计较为全面，涵盖多种参数组合（记忆和温度）、两种市场类型和两种模型，并通过市场动态和行为参数估计进行多维度比较；但局限性在于未探索更复杂的动态（如多市场交互）或人口统计学变量的影响。
*   **开销**：使用 OpenAI API 的计算成本相对较低，但模拟 50 个时间步的多次实验需要大量 API 调用，可能对大规模模拟构成资源挑战。

## Further Thoughts

论文揭示了上下文记忆对 LLM 模拟人类行为的关键作用，记忆 ≥ 3 时效果最佳，启发我们未来可能通过优化记忆‘质量’而非‘数量’来提升模拟效率；此外，LLM 行为异质性不足的问题提示我们可以通过个性化提示（如引入文化或风险偏好变量）增强多样性，这对经济学中的代理建模（Agent-Based Modeling）可能产生深远影响；最后，LLMs 在动态反馈环境中的潜力（如模拟市场泡沫）启发我们探索其作为政策模拟工具的可能性，用于预测现实经济现象或测试干预措施。