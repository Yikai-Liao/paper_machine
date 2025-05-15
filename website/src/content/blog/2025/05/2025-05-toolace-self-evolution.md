---
title: "ToolACE-DEV: Self-Improving Tool Learning via Decomposition and EVolution"
pubDatetime: 2025-05-12T12:48:30+00:00
slug: "2025-05-toolace-self-evolution"
type: "arxiv"
id: "2505.07512"
score: 0.7260514866178546
author: "grok-3-latest"
authors: ["Xu Huang", "Weiwen Liu", "Xingshan Zeng", "Yuefeng Huang", "Xinlong Hao", "Yuxian Wang", "Yirong Zeng", "Chuhan Wu", "Yasheng Wang", "Ruiming Tang", "Defu Lian"]
tags: ["LLM", "Tool Learning", "Self-Evolution", "Task Decomposition", "Sampling"]
institution: ["University of Science and Technology of China", "Huawei Noah’s Ark Lab", "Huawei Technologies Co., Ltd"]
description: "本文提出 ToolACE-DEV 框架，通过任务分解和自进化机制，使轻量级大型语言模型在工具学习任务上实现自主改进，显著提升工具调用准确率并降低对高级模型的依赖。"
---

> **Summary:** 本文提出 ToolACE-DEV 框架，通过任务分解和自进化机制，使轻量级大型语言模型在工具学习任务上实现自主改进，显著提升工具调用准确率并降低对高级模型的依赖。 

> **Keywords:** LLM, Tool Learning, Self-Evolution, Task Decomposition, Sampling

**Authors:** Xu Huang, Weiwen Liu, Xingshan Zeng, Yuefeng Huang, Xinlong Hao, Yuxian Wang, Yirong Zeng, Chuhan Wu, Yasheng Wang, Ruiming Tang, Defu Lian

**Institution(s):** University of Science and Technology of China, Huawei Noah’s Ark Lab, Huawei Technologies Co., Ltd


## Problem Background

大型语言模型（LLMs）在工具使用（Tool Learning）方面展现出潜力，可以通过调用外部工具（如搜索引擎、API）获取实时信息并处理复杂任务。然而，现有方法主要依赖于通过高级模型（如 GPT-4）合成数据进行蒸馏（Distillation），这带来了高昂的推理成本、数据兼容性问题（目标模型与高级模型知识范围差异导致幻觉）和数据隐私风险。论文提出了一种自进化（Self-Evolution）框架 ToolACE-DEV，旨在减少对高级模型的依赖，通过任务分解和自主数据生成提升轻量级模型的工具使用能力。

## Method

* **核心思想**：通过任务分解（Task Decomposition）和自进化机制（Self-Evolution），让轻量级 LLMs 在不依赖高级模型的情况下，逐步构建并提升工具使用能力。
* **具体实现**：
  - **工具文档适应（Tool Documentation Adaption）**：针对工具文档设计指令微调任务，让模型熟悉工具定义、语法和使用约束，提升对工具的理解能力。这类似于领域特定的持续预训练，但专注于工具相关知识，特别针对已完成指令微调的模型设计，确保保留其指令跟随能力。
  - **查询感知的工具生成与调用（Query-Aware Tool Generation and Invocation）**：将工具学习任务分解为两个子任务：一是根据用户查询生成候选工具（Tool Generation），二是基于查询和候选工具生成工具调用（Tool Invocation）。通过这种分解，模型不仅能使用现有工具，还能自主扩展工具集，为后续自进化奠定基础。训练目标是分别优化生成工具和调用工具的损失函数。
  - **自进化机制（Self-Evolution）**：在面对新查询时，模型首先生成候选工具（通过指令引导），然后生成工具调用（采用 Top-k 采样和自一致性解码策略以提高质量），并通过规则检查过滤无效数据（如格式错误或幻觉工具）。最终，将生成的高质量数据用于更新模型参数，形成迭代改进循环，逐步提升工具使用性能。
* **关键特点**：方法不依赖外部高级模型，生成的训练数据与目标模型的知识范围更兼容，同时通过规则检查和自一致性解码确保数据质量，降低幻觉风险。

## Experiment

* **有效性**：ToolACE-DEV 在 8B 参数规模下，性能显著优于基线模型 LLaMA-3.1-8B-Instruct，在 Berkeley Function Call Leaderboard (BFCL) 上与 xLAM-8x22B-r 等大型混合专家模型相当，超越了许多更大规模模型（如 LLaMA-3-70B）和闭源模型（如 Claude、Gemini）。在 API-Bank 和 T-Eval 等其他基准测试中，也展现出明显提升，尤其在真实场景（Live 子集）中表现更优。
* **自进化效果**：通过三轮自进化迭代，模型在 Non-Live、Live 和整体指标上持续改进，尤其在复杂任务（Live 子集）中提升显著，但随着迭代次数增加，收益递减，可能是由于模型信心饱和或自生成数据多样性不足。
* **实验设置合理性**：实验覆盖了不同规模（1.5B 到 8B）和不同系列的模型（LLaMA、Qwen、Mistral），验证了方法的普适性。消融研究清晰展示了各模块（如工具文档适应、工具生成任务）的贡献，证明任务分解和自进化的必要性。不足之处在于未测试更大规模模型（如 14B 以上），可能限制了对自进化潜力的全面评估。
* **开销**：自进化过程增加了数据生成和规则检查的计算成本，但通过 vLLM 框架加速生成，且整体资源需求远低于依赖高级模型的数据合成方法。

## Further Thoughts

论文的自进化机制为工具学习提供了一种新范式，启发我们可以在其他复杂任务（如多步推理或多模态交互）中尝试类似的自生成数据迭代改进策略。此外，任务分解的思路也具有普适性，将复杂目标拆分为可逐步解决的子任务（如工具生成和调用），可能适用于提升模型在其他领域的泛化能力。自一致性解码（Self-Consistency Decoding）在数据质量控制中的应用也值得关注，通过多解采样和投票机制筛选高质量数据，这种方法或可在其他自监督学习场景中发挥作用，例如在代码生成或数学推理任务中用于过滤错误解法。