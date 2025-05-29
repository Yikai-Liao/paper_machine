---
title: "The Coverage Principle: A Framework for Understanding Compositional Generalization"
pubDatetime: 2025-05-26T17:55:15+00:00
slug: "2025-05-coverage-principle"
type: "arxiv"
id: "2505.20278"
score: 0.580817985248669
author: "grok-3-latest"
authors: ["Hoyeon Chang", "Jinho Park", "Hanseul Cho", "Sohee Yang", "Miyoung Ko", "Hyeonbin Hwang", "Seungpil Won", "Dohaeng Lee", "Youbin Ahn", "Minjoon Seo"]
tags: ["LLM", "Compositional Generalization", "Pattern Matching", "Data Scaling", "Reasoning"]
institution: ["KAIST", "UCL", "LG AI Research"]
description: "本文提出覆盖原则这一数据驱动框架，揭示了依赖模式匹配的模型在组合泛化中的根本边界，并通过实验验证了其对 Transformer 模型泛化能力的预测能力。"
---

> **Summary:** 本文提出覆盖原则这一数据驱动框架，揭示了依赖模式匹配的模型在组合泛化中的根本边界，并通过实验验证了其对 Transformer 模型泛化能力的预测能力。 

> **Keywords:** LLM, Compositional Generalization, Pattern Matching, Data Scaling, Reasoning

**Authors:** Hoyeon Chang, Jinho Park, Hanseul Cho, Sohee Yang, Miyoung Ko, Hyeonbin Hwang, Seungpil Won, Dohaeng Lee, Youbin Ahn, Minjoon Seo

**Institution(s):** KAIST, UCL, LG AI Research


## Problem Background

大型语言模型（LLMs）在模式匹配任务中表现出色，但往往在需要系统性组合泛化（Compositional Generalization）的任务中表现不佳。
论文的出发点是现有研究缺乏一个统一的框架来预测模型在组合任务上的成功或失败，因此提出了‘覆盖原则’（Coverage Principle），旨在通过数据驱动的视角揭示模式匹配机制在组合泛化中的根本边界，解决如何理解和预测 Transformer 模型在组合任务上的泛化能力这一关键问题。

## Method

*   **核心思想：** 提出‘覆盖原则’作为数据驱动的框架，指出依赖模式匹配的模型只能在训练数据定义的‘覆盖范围’内可靠地进行组合泛化，超出覆盖范围的输入预测不受约束。
*   **具体步骤：**
    *   **功能等价性（Functional Equivalence）定义：** 两个输入片段在相同上下文中产生相同输出时被视为功能等价，通过训练数据中的观察次数（以参数 k 表示证据强度）量化这种等价性。
    *   **覆盖范围（Coverage）构建：** 基于功能等价性构建‘替代图’（Substitution Graph），覆盖范围包括训练数据中观察到的输入以及通过功能等价替代可达的所有输入。
    *   **覆盖原则（Coverage Principle）声明：** 模型仅在覆盖范围内能可靠泛化，超出范围的预测不受训练数据约束。
*   **实验验证：** 通过合成组合任务（如 2-Hop, Non-Tree）验证覆盖原则，分析 Transformer 模型在不同数据量和任务结构下的泛化表现，并探讨链式思维（Chain-of-Thought, CoT）监督对数据效率的影响。
*   **关键特点：** 该框架不依赖特定模型架构，聚焦于数据特性，适用于任何依赖模式匹配的学习系统。

## Experiment

*   **有效性：** 实验验证了覆盖原则的预测能力，例如在 2-Hop 任务中，训练数据量需至少以 token 集大小的二次方增长才能实现完全的域内（In-Domain, ID）泛化，模型在覆盖范围外的域外（Out-of-Domain, OOD）测试数据上表现接近随机水平。
*   **数据效率：** 即使将模型参数规模从 68M 提升至 1.5B（20 倍增长），数据需求量未显著减少，表明限制主要来自数据特性而非模型容量。
*   **路径歧义影响：** 在 Non-Tree 任务中，路径歧义导致模型无法形成统一的中间状态表示，泛化性能和可解释性均受损，即使提供近乎穷尽的训练数据也难以完全泛化。
*   **CoT 监督效果：** 链式思维（CoT）监督显著提升多跳任务的数据效率，例如 3-Hop 任务的幂律指数从 2.58 降至 1.76，但对路径歧义任务的改进有限。
*   **实验设置合理性：** 实验涵盖多种合成任务（2-Hop, Parallel 2-Hop, 3-Hop, Non-Tree），在不同模型规模和数据集大小下测试，数据支持理论预测，设置全面且合理。

## Further Thoughts

覆盖原则作为一个数据驱动的框架，启发我们重新思考模型泛化能力的边界，其形式化的‘功能等价性’和‘替代图’概念为分析组合任务提供了新视角，可扩展至视觉推理或多模态任务，探索模式匹配在其他数据分布中的局限性；此外，论文提出的泛化机制分类（结构基础、属性基础、共享操作）为设计更具系统性泛化能力的架构提供了理论指导，值得探索如何将这些机制融入模型设计。