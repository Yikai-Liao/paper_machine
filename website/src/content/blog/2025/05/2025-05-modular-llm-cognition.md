---
title: "Procedural Memory Is Not All You Need: Bridging Cognitive Gaps in LLM-Based Agents"
pubDatetime: 2025-05-06T11:18:34+00:00
slug: "2025-05-modular-llm-cognition"
type: "arxiv"
id: "2505.03434"
score: 0.7703743387578753
author: "grok-3-latest"
authors: ["Schaun Wheeler", "Olivier Jeunen"]
tags: ["LLM", "Modular Architecture", "Semantic Memory", "Associative Learning", "Reinforcement Learning"]
institution: ["aampe"]
description: "本文提出一种模块化架构，通过解耦程序性、语义和关联功能，弥补大型语言模型在‘恶劣’学习环境中的认知缺陷，为构建适应复杂现实世界的 AI 代理提供了理论框架。"
---

> **Summary:** 本文提出一种模块化架构，通过解耦程序性、语义和关联功能，弥补大型语言模型在‘恶劣’学习环境中的认知缺陷，为构建适应复杂现实世界的 AI 代理提供了理论框架。 

> **Keywords:** LLM, Modular Architecture, Semantic Memory, Associative Learning, Reinforcement Learning

**Authors:** Schaun Wheeler, Olivier Jeunen

**Institution(s):** aampe


## Problem Background

大型语言模型（LLMs）在程序性任务上表现出色，但由于其架构主要依赖程序性记忆（Procedural Memory），在动态、规则变化、反馈模糊的‘恶劣’学习环境（Wicked Learning Environments）中表现不佳，缺乏语义记忆（Semantic Memory）和关联学习（Associative Learning）能力，导致无法进行灵活推理、跨会话记忆整合或自适应决策。

## Method

* **核心思想**：通过模块化架构（Modular Architecture）弥补 LLMs 的认知缺陷，将程序性能力与其他认知功能解耦，构建适应复杂环境的 AI 代理。
* **具体实现**：
  * **模块划分**：系统分为程序性模块（由 LLMs 负责生成连贯输出）、语义模块（管理结构化知识，形成抽象、可泛化的表示）和关联模块（通过强化学习建立状态与行动间的关系，形成跨上下文链接）。
  * **代理学习者与执行者分离**：引入‘代理学习者’（Agentic Learners）和‘代理执行者’（Agentic Actors）。学习者通过强化学习（Reinforcement Learning）和探索-利用机制（如 Thompson Sampling）构建上下文元数据，专注于自适应推理和长期学习；执行者（即 LLMs）根据上下文元数据生成具体输出。
  * **上下文元数据整合**：学习者生成的上下文元数据作为前缀传递给 LLMs，确保程序性输出受到语义和关联洞察的指导，提升输出的上下文相关性和决策适应性。
  * **针对性设计**：模块化系统特别针对‘恶劣’学习环境，学习者通过概率更新用户偏好假设，处理不确定性，而非追求确定性答案，模拟人类在不确定性中的决策过程。
* **关键优势**：不改造 LLMs 核心架构，而是通过外部模块补充其不足，充分发挥各组件优势，避免单体架构的局限性。

## Experiment

* **理论分析而非实验数据**：论文未提供具体的实验结果或定量评估，而是通过理论分析和案例描述（如客户参与场景）说明 LLMs 在‘恶劣’环境中的不足及模块化架构的潜在优势。
* **合理性与局限**：提出的模块化方法在概念上全面，基于认知科学和决策研究（如 Hogarth 的学习环境理论），考虑了不同环境需求，但缺乏实际数据支持，无法判断效果是否显著或实现成本是否可控。
* **未来验证需求**：模块间协调、错误传播等问题需通过实验进一步验证，当前仅为理论框架。

## Further Thoughts

模块化认知系统的设计理念启发我们思考如何在 AI 中引入更多生物启发的机制，模拟人类大脑不同认知功能的分工，而不仅仅追求单一模型的规模扩展；此外，区分‘善良’和‘恶劣’学习环境并针对性设计架构的思路，为动态环境适配提供了新视角，未来可探索如何实时识别环境类型并切换系统模式；最后，强化学习与 LLMs 的结合提示我们可以在 LLMs 之外引入更多决策优化工具，提升其在复杂任务中的适应性。