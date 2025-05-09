---
title: "Procedural Memory Is Not All You Need: Bridging Cognitive Gaps in LLM-Based Agents"
pubDatetime: 2025-05-06T11:18:34+00:00
slug: "2025-05-modular-llm-architecture"
type: "arxiv"
id: "2505.03434"
score: 0.7703743387578753
author: "grok-3-latest"
authors: ["Schaun Wheeler", "Olivier Jeunen"]
tags: ["LLM", "Modular Architecture", "Semantic Memory", "Associative Learning", "Reinforcement Learning"]
institution: ["aampe"]
description: "本文提出一种模块化架构，通过分离程序性、语义和关联功能，弥补大型语言模型在复杂‘恶劣’学习环境中的认知缺陷，为构建适应性AI代理提供了理论基础。"
---

> **Summary:** 本文提出一种模块化架构，通过分离程序性、语义和关联功能，弥补大型语言模型在复杂‘恶劣’学习环境中的认知缺陷，为构建适应性AI代理提供了理论基础。 

> **Keywords:** LLM, Modular Architecture, Semantic Memory, Associative Learning, Reinforcement Learning

**Authors:** Schaun Wheeler, Olivier Jeunen

**Institution(s):** aampe


## Problem Background

大型语言模型（LLMs）在程序性任务上表现出色，但由于其架构主要依赖程序性记忆（Procedural Memory），在面对动态规则、模糊反馈和充满新奇性的‘恶劣’（Wicked）学习环境时，缺乏语义记忆（Semantic Memory）和关联学习（Associative Learning）能力，导致无法进行灵活推理、跨会话记忆和适应性决策，限制了其在复杂现实世界应用中的表现。

## Method

*   **核心思想:** 提出一种模块化架构，将认知功能分为程序性、语义和关联模块，以弥补LLMs在复杂环境中的局限性，并实现更强的适应性决策能力。
*   **模块分工:** 
    *   **程序性模块:** 由LLMs承担，负责基于上下文生成流畅的自然语言响应，专注于执行任务。
    *   **语义模块:** 管理结构化知识，将学习到的动作和概念组织成抽象、可泛化的表示，类似于人类语义记忆对事实和规则的编码。
    *   **关联模块:** 通过形成和检索状态与动作之间的关系，连接不同经验，类似于人类认知中的关联绑定。
*   **实现机制:** 
    *   引入‘代理学习者’（Agentic Learner）和‘代理执行者’（Agentic Actor）的概念。学习者通过强化学习（Reinforcement Learning, RL）和探索-利用机制（如Thompson Sampling），从用户交互数据流中构建上下文元数据，逐步更新对用户偏好或环境的理解。
    *   学习者生成的上下文向量作为前缀传递给LLM（执行者），指导其生成过程，确保响应既流畅又符合动态上下文。
*   **设计优势:** 模块化分离避免了单一模型处理不擅长任务的缺陷，提高了系统的可解释性和可维护性，同时通过学习者模块的慢速适应和执行者模块的实时响应，平衡了适应性与效率。

## Experiment

*   **有效性:** 论文未提供具体的实验数据或定量结果，而是通过理论分析和案例讨论（如客户互动场景）说明模块化架构的潜在优势，例如在处理用户偏好变化或模糊反馈时，系统能够逐步更新假设并生成上下文相关的响应。
*   **局限性:** 由于缺乏实际实验验证，方法的实际提升效果无法直接评估；同时，论文未详细讨论模块间协调、错误传播等问题，实验设置的全面性和合理性存在疑问。
*   **理论合理性:** 尽管缺乏数据支持，模块化架构在理论上为区分‘善良’（Kind）和‘恶劣’（Wicked）学习环境提供了清晰框架，具备一定的指导意义。

## Further Thoughts

模块化架构的设计理念令人启发，未来可以探索如何进一步结合神经符号架构（Neural-Symbolic Architectures）或稀疏记忆模型（Sparse Memory Models）来增强语义和关联模块的功能；此外，学习环境分类（Kind vs. Wicked）也为AI系统设计提供了新视角，可以尝试量化环境的‘恶劣程度’，并据此动态调整模块配置或资源分配。