---
title: "Architectural Precedents for General Agents using Large Language Models"
pubDatetime: 2025-05-11T18:29:54+00:00
slug: "2025-05-cognitive-patterns-llm-agents"
type: "arxiv"
id: "2505.07087"
score: 0.5370173860760918
author: "grok-3-latest"
authors: ["Robert E. Wray", "James R. Kirk", "John E. Laird"]
tags: ["LLM", "Cognitive Architecture", "Agentic Systems", "Reasoning", "Memory Design"]
institution: ["Center for Integrated Cognition, IQMRI, Ann Arbor, MI, USA"]
description: "本文通过提炼传统认知架构中的认知设计模式，为基于大型语言模型的代理系统提供理论分析框架，识别其局限性并预测未来研究方向，以推动通用智能的实现。"
---

> **Summary:** 本文通过提炼传统认知架构中的认知设计模式，为基于大型语言模型的代理系统提供理论分析框架，识别其局限性并预测未来研究方向，以推动通用智能的实现。 

> **Keywords:** LLM, Cognitive Architecture, Agentic Systems, Reasoning, Memory Design

**Authors:** Robert E. Wray, James R. Kirk, John E. Laird

**Institution(s):** Center for Integrated Cognition, IQMRI, Ann Arbor, MI, USA


## Problem Background

本文旨在探索如何利用大型语言模型（LLMs）构建通用智能代理（General Agents），并解决 LLMs 在复杂推理、记忆管理和非单调推理等方面的局限性。
作者从传统认知架构（Cognitive Architectures）研究出发，试图通过提炼通用的认知设计模式（Cognitive Design Patterns），为基于 LLMs 的代理系统（Agentic LLM Systems）提供理论指导，识别现有系统的不足，并预测未来研究方向，以推动人工通用智能（AGI）的实现。

## Method

*   **核心思想：** 通过从传统认知架构中提取通用的认知设计模式，将其应用于基于 LLMs 的代理系统，以分析现有系统的功能实现情况，并提出改进方向。
*   **具体步骤：**
    *   **模式提炼：** 从传统认知架构（如 Soar, ACT-R, BDI）中总结出常见的功能和表示模式，例如‘观察-决策-行动’（Observe-Decide-Act）、‘情景记忆’（Episodic Memory）、‘知识编译’（Knowledge Compilation）等，作为分析框架。
    *   **模式映射：** 分析现有 Agentic LLM Systems（如 ReAct, Generative Agents）中是否体现了这些模式，评估其实现程度和局限性，例如 ReAct 部分实现了‘观察-决策-行动’模式，但缺乏‘承诺’（Commitment）机制。
    *   **未探索模式识别：** 提出一些在传统架构中常见但在 LLMs 系统中尚未被充分研究的模式，如‘重新考虑’（Reconsideration）和‘知识编译’（Knowledge Compilation），并讨论其潜在应用价值。
    *   **新模式探索：** 基于 LLMs 的生成特性，提出可能的新模式，如‘逐步反思’（Step-wise Reflection），并分析其与传统模式的异同。
*   **方法特点：** 这种方法是一种理论性分析和跨领域整合，注重抽象功能比较，而非具体算法实现，旨在为 LLMs 在 AGI 研究中的应用提供系统性指导。

## Experiment

*   **实验性质：** 本文是一篇理论性和综述性论文，未提供具体的实验数据或定量结果，而是通过文献调研和案例分析支持其观点。
*   **案例分析：** 作者通过分析现有 Agentic LLM Systems（如 ReAct 和 Generative Agents）中认知设计模式的体现情况，评估其功能实现和局限性。例如，ReAct 系统在‘观察-决策-行动’模式上的应用提升了跨领域任务的表现，但缺乏‘承诺’机制可能限制其推理深度；Generative Agents 在‘情景记忆’方面有所创新，但未完全符合编码特异性原则（Encoding Specificity）。
*   **合理性与局限：** 分析设置较为全面，涵盖了多种模式和系统，但缺乏实验验证，更多是提出假设和研究方向，而非结论性成果。

## Further Thoughts

本文提出的认知设计模式作为分析框架的思路非常具有启发性，可以推广到其他 AI 领域，如多模态模型或强化学习系统，探索更多跨领域借鉴的可能性。
此外，‘重新考虑’和‘知识编译’等未探索模式若能在 LLMs 中实现，可能通过设计特定提示策略或外部记忆模块显著提升其动态适应性和推理效率。
最后，‘逐步反思’作为 LLMs 独特生成机制带来的新模式，提示我们是否可以利用其生成不确定性来模拟人类认知中的创造性思维，值得进一步研究。