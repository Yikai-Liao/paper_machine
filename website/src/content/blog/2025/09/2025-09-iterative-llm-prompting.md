---
title: "Another Turn, Better Output? A Turn-Wise Analysis of Iterative LLM Prompting"
pubDatetime: 2025-09-08T14:54:31+00:00
slug: "2025-09-iterative-llm-prompting"
type: "arxiv"
id: "2509.06770"
score: 0.5753559271650983
author: "grok-3-latest"
authors: ["Shashidhar Reddy Javaji", "Bhavul Gauri", "Zining Zhu"]
tags: ["LLM", "Iterative Refinement", "Multi-Turn Interaction", "Prompting Strategy", "Semantic Drift"]
institution: ["Stevens Institute of Technology", "Meta"]
description: "本文通过系统性实验框架，揭示了大型语言模型在多轮迭代优化中的行为模式，为不同任务领域和反馈策略下的迭代效果提供了量化指导。"
---

> **Summary:** 本文通过系统性实验框架，揭示了大型语言模型在多轮迭代优化中的行为模式，为不同任务领域和反馈策略下的迭代效果提供了量化指导。 

> **Keywords:** LLM, Iterative Refinement, Multi-Turn Interaction, Prompting Strategy, Semantic Drift

**Authors:** Shashidhar Reddy Javaji, Bhavul Gauri, Zining Zhu

**Institution(s):** Stevens Institute of Technology, Meta


## Problem Background

大型语言模型（LLMs）在多轮交互中的迭代优化行为缺乏系统性评估，尤其是在不同任务领域（如创意生成、代码生成、数学推理）中，迭代是否有助于提升输出质量、何时会导致性能下降或语义漂移尚不明确。
现有研究多关注结构化提示技术，而对常见模糊反馈（如‘improve it’）的效果了解不足，同时多轮交互中模型性能下降和过度自信等问题亟需解决。

## Method

*   **任务领域与数据集**：选取三个认知需求不同的领域——创意生成（Ideation，从 LiveIdeaBench 抽取 50 个任务）、代码生成（Coding，从 DS-1000 抽取 50 个任务）和数学推理（Math，从 Omni-MATH 抽取 50 个高难度任务），以覆盖多样化任务需求。
*   **迭代优化协议**：设计自动化多轮交互实验，每任务进行 12 轮对话，第一轮为初始生成，后续轮次基于前一轮输出改进，采用‘无记忆’模式（不重复提供初始提示），以测试模型内部一致性和改进能力。
*   **提示策略**：分为两组对比实验：模糊反馈组（Vague Feedback，使用‘Improve it’‘Make it better’‘Refine it’三种近义提示，测试模型默认行为）和具体指导组（Specific Steering，根据领域设计针对性提示，如创意任务的‘更具新颖性’或‘更实用’，代码任务的‘优化速度’或‘提高可读性’，数学任务的‘详细阐述’或‘探索替代方法’），以分析反馈类型对迭代轨迹的影响。
*   **测试模型**：选用四种主流 LLMs（GPT-3.5-Turbo、Claude-Sonnet-4.0、Llama-3.1-8B-Instruct、GPT-OSS-20B），覆盖不同架构和训练方法，确保结果普适性，统一设置温度为 0.7 和最大 token 数为 10K。
*   **评估框架**：构建多维度评估体系，包括：1）结果与效率指标（数学和代码任务的正确性，通过单元测试和答案等价性评估）；2）行为动态指标（语义漂移 Drift from Origin、轮间变化 Turn-to-Turn Volatility、词汇新颖性 Lexical Novelty、输出增长 Growth Factor，使用 Qwen3-Embedding-0.6B 模型计算语义相似度）；3）语义质量指标（通过 Gemini 2.5 Pro 作为评判模型，评估创意任务的原创性和实用性、代码任务的实用性和可读性、数学任务的逻辑严谨性和解释清晰度）。

## Experiment

*   **有效性与提升**：迭代优化效果高度依赖任务领域和反馈类型。创意生成和代码生成中，改进主要在早期轮次（前 3-4 轮），之后模糊反馈常导致质量停滞或下降；数学推理中，晚期轮次（8-12 轮）在具体指导（如‘详细阐述’）下显著提升正确率，例如 Llama-3.1-8B 准确率从 6.9% 提升至 40.5%，相对提升超 480%。具体指导在所有领域中均优于模糊反馈，尤其在数学任务中‘详细阐述’提示效果最佳。
*   **实验设置合理性**：实验覆盖三种任务领域、四种模型、两种反馈类型，12 轮迭代捕捉长期动态，评估指标结合客观正确性和主观质量，并通过行为动态指标分析迭代过程。任务数量（每领域 50 个）和模型多样性增强了结果普适性，但论文指出模型和任务数量有限，提示空间探索不足。
*   **领域特异性**：创意生成中语义漂移大，易陷入重复；代码生成中输出规模增长明显（增长因子高达 40 倍），但语义变化小，常过度复杂化；数学推理中初始固定性强，但晚期阐述可突破正确路径。
*   **数据验证**：数学任务中‘s1_elaboration’提示在晚期准确率显著高于模糊反馈（如 Llama-3.1-8B 第 12 轮达 82%，而‘v1_improve’仅 34%-44%）；代码任务中 Claude 和 GPT-OSS-20B 第 1 轮即达最高正确率（90%），后续迭代无显著提升；创意任务中 Claude 和 GPT-OSS-20B 保持高词汇新颖性（最终 0.843 和 0.812），Llama-3.1-8B 降至 0.084，结果与结论一致。

## Further Thoughts

迭代优化效果依赖任务领域和提示策略的发现启发我们设计‘阶段性策略’，如创意任务早期用新颖性提示激发发散思维，后期用实用性提示收敛；在数学任务中鼓励晚期深入阐述挖掘正确路径。此外，论文提出的多代理或多模型框架（如一个模型生成，另一个精炼）具有潜力，未来可探索基于实时行为指标（如语义漂移）动态调整提示或切换模型的适应性系统，构建更稳定高效的人工智能交互模式。