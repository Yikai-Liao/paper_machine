---
title: "An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents"
pubDatetime: 2025-05-21T05:09:43+00:00
slug: "2025-05-rl-search-agents"
type: "arxiv"
id: "2505.15117"
score: 0.5700049578337293
author: "grok-3-latest"
authors: ["Bowen Jin", "Jinsung Yoon", "Priyanka Kargupta", "Sercan Ö. Arık", "Jiawei Han"]
tags: ["LLM", "Reinforcement Learning", "Search Agent", "Reasoning", "Retrieval"]
institution: ["University of Illinois at Urbana-Champaign", "Google Cloud AI Research"]
description: "本文通过系统性实验研究了强化学习训练大型语言模型搜索代理的关键设计因素，揭示了格式奖励、通用型模型和高质量搜索工具对性能的显著提升，为构建高效搜索代理提供了实用指导。"
---

> **Summary:** 本文通过系统性实验研究了强化学习训练大型语言模型搜索代理的关键设计因素，揭示了格式奖励、通用型模型和高质量搜索工具对性能的显著提升，为构建高效搜索代理提供了实用指导。 

> **Keywords:** LLM, Reinforcement Learning, Search Agent, Reasoning, Retrieval

**Authors:** Bowen Jin, Jinsung Yoon, Priyanka Kargupta, Sercan Ö. Arık, Jiawei Han

**Institution(s):** University of Illinois at Urbana-Champaign, Google Cloud AI Research


## Problem Background

大型语言模型（LLMs）在复杂任务中需要与外部环境交互（如调用搜索工具）以实现多轮推理和动态知识获取，但现有方法如提示工程和监督微调依赖昂贵的手动标注，难以扩展。
论文旨在通过强化学习（RL）训练 LLM 搜索代理，解决如何优化 RL 设计以提升代理性能的问题，重点研究奖励设计、底层 LLM 选择和搜索工具质量的影响。

## Method

*   **奖励设计：** 基于最终结果的奖励（Outcome Reward）作为主要优化信号，评估最终答案的正确性（如通过精确字符串匹配）。此外，引入格式奖励（Format Reward），通过特殊标记（如 <think>、<search>）验证模型输出格式是否正确，辅助模型学习搜索工具调用；引入中间检索奖励（Intermediate Retrieval Reward），评估每次检索结果的相关性，鼓励生成高质量查询。
*   **底层 LLM 选择：** 对比通用型 LLM（如 Qwen2.5 系列）和推理专用型 LLM（如 DeepSeek-R1-Distill-Qwen）在 RL 训练中的表现，分析模型类型对指令遵循和推理能力的影响；同时研究模型规模（3B、7B、14B、32B）对性能的贡献，探索规模与任务需求的匹配。
*   **搜索工具选择：** 测试不同质量的搜索工具在训练和推理阶段的作用，包括随机噪声（无信息）、BM25（基于词频的稀疏检索）、E5 嵌入模型（基于语义的密集检索）等，分析工具质量如何影响训练动态（如搜索调用频率）和代理鲁棒性。
*   **RL 算法：** 采用 Proximal Policy Optimization (PPO) 和 Group Relative Policy Optimization (GRPO) 进行策略优化，通过最大化累积奖励和控制 KL 散度（与参考模型的分布差异）来训练模型在多轮推理-搜索循环中的决策能力。
*   **核心流程：** 模型在每个迭代中执行推理（分析上下文）、生成搜索查询、获取外部信息并更新上下文，最终输出答案，整个过程通过 RL 奖励信号动态调整。

## Experiment

*   **奖励设计效果：** 格式奖励显著提升性能，尤其对基础 LLM（如 Qwen2.5-3B），在多个数据集（如 NQ、HotpotQA）上平均准确率提升明显（从 0.303 到 0.389），并加速训练收敛；中间检索奖励效果有限，甚至可能因过度约束检索轨迹而降低性能（如在 Qwen2.5-7B 上准确率下降）。
*   **底层 LLM 选择效果：** 通用型 LLM（如 Qwen2.5）优于推理专用型 LLM（如 DeepSeek-R1），在 PPO 和 GRPO 训练下平均准确率更高（例如 Qwen2.5-7B 达 0.434 vs. DeepSeek-R1 的 0.344），原因可能是通用模型在指令遵循和搜索调用学习上更高效；模型规模增加带来性能提升，但收益递减（如 3B 到 32B 在 Bamboogle 数据集上准确率提升趋缓）。
*   **搜索工具选择效果：** 高质量搜索工具（如 E5 Exact）在训练时带来更稳定动态和更高性能（平均准确率 0.430），低质量工具（如随机噪声）导致模型放弃检索；推理时，代理对不同工具表现出较强泛化能力，且更强工具（如 Google Search）显著提升下游性能（例如在 SimpleQA 上准确率达 0.603）。
*   **实验设置合理性：** 实验覆盖多个数据集（通用问答和多跳问答）、模型规模、RL 算法和搜索工具，数据量和变量控制较为全面，支持了结论的可靠性；同时，案例研究进一步揭示了搜索工具质量对推理和检索行为的影响。

## Further Thoughts

奖励设计的细粒度控制（如格式奖励）启发我们在其他代理任务中引入类似辅助信号，引导模型学习特定行为模式；搜索工具质量对训练动态的影响提示未来可以探索联合优化检索算法和 LLM 策略，例如通过动态调整检索模块或引入可学习检索机制；模型规模收益递减的现象表明资源受限时应优先优化外部知识获取能力，而非单纯扩大参数规模。