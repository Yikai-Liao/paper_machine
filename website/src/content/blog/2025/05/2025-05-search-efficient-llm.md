---
title: "SEM: Reinforcement Learning for Search-Efficient Large Language Models"
pubDatetime: 2025-05-12T09:45:40+00:00
slug: "2025-05-search-efficient-llm"
type: "arxiv"
id: "2505.07903"
score: 0.70998004030645
author: "grok-3-latest"
authors: ["Zeyang Sha", "Shiwen Cui", "Weiqiang Wang"]
tags: ["LLM", "Reinforcement Learning", "Search Efficiency", "Tool Invocation", "Reasoning"]
institution: ["Ant Group"]
description: "本文提出 *SEM* 强化学习框架，通过平衡数据集、结构化模板和奖励机制，优化大型语言模型的搜索行为，显著提升了搜索效率和答案准确性。"
---

> **Summary:** 本文提出 *SEM* 强化学习框架，通过平衡数据集、结构化模板和奖励机制，优化大型语言模型的搜索行为，显著提升了搜索效率和答案准确性。 

> **Keywords:** LLM, Reinforcement Learning, Search Efficiency, Tool Invocation, Reasoning

**Authors:** Zeyang Sha, Shiwen Cui, Weiqiang Wang

**Institution(s):** Ant Group


## Problem Background

大型语言模型（LLMs）在推理和调用外部工具（如搜索引擎）方面表现出色，但如何让模型准确判断何时需要调用搜索、何时依赖内部知识仍是一个挑战。
现有强化学习方法常导致模型过度调用搜索工具，造成资源浪费和效率低下，例如对简单问题（如‘1+1=?’）也进行多次不必要的搜索，增加了计算成本和响应时间。

## Method

*   **核心思想:** 提出一种名为 *SEM* 的后训练强化学习框架，旨在优化 LLMs 的搜索行为，使其在需要时有效调用搜索工具，而在不需要时避免冗余操作。
*   **数据集构建:** 结合 Musique（多跳事实性问题，通常需要外部知识）和 MMLU（学术性问题，通常在模型预训练知识范围内）两个数据集，形成一个平衡的训练集，帮助模型学习区分‘已知’和‘未知’问题。
*   **奖励机制:** 采用 Group Relative Policy Optimization (GRPO) 框架，设计一个复杂的奖励函数，鼓励模型在已知问题上直接回答（避免搜索并给予正反馈），而在未知问题上主动调用搜索工具并基于检索结果改进答案（给予正奖励）；同时对不必要的搜索行为施加惩罚，并对输出格式的规范性进行约束。
*   **训练模板:** 定义一个结构化输出格式，包含 `<think>`（初始推理）、`<answer>`（初步答案）、`<search>`（搜索查询）、`<result>`（检索结果）等标签，确保模型在推理过程中清晰表达决策步骤，便于评估和优化搜索行为的合理性。
*   **训练过程:** 在训练中，模型根据奖励函数逐步调整搜索决策，学习在准确性和效率之间取得平衡，而非依赖静态提示或启发式规则。

## Experiment

*   **有效性:** 在需要搜索的数据集（如 HotpotQA 和 MuSiQue）上，*SEM* 显著提高了准确率，例如在 HotpotQA 上，7B-Instruct 模型的 Exact Match (EM) 从 Naive RAG 的 18.01 提升到 35.84，同时保持较高的搜索比例（Search Ratio, SR 达 97.54%），表明搜索行为的优化提升了答案质量。
*   **效率:** 在不需要搜索的数据集（如 MMLU 和 GSM8k）上，*SEM* 成功降低了搜索比例，例如在 MMLU 上，7B-Instruct 模型的 SR 仅为 1.77%（远低于 Naive RAG 的 47.98%），而 EM 仍达到 70.88%，表明模型学会了避免不必要的搜索。
*   **对比分析:** 相较于基线方法 ReSearch，*SEM* 在训练稳定性（F1 分数随训练步数增长更平滑）和搜索决策合理性上表现更优，ReSearch 模型常因过度或不足搜索导致性能不稳定。
*   **实验设置:** 数据集选择覆盖‘已知’和‘未知’场景，评价指标（EM、LJ、SR）全面反映准确性和效率，设置合理；但训练步数较少（仅 200 步）可能限制潜力发挥，且未深入探讨模型规模对结果的影响。

## Further Thoughts

本文通过平衡数据集模拟‘已知’和‘未知’场景的思路，可推广至其他工具调用或决策优化任务，如优化 API 或数据库调用行为；结构化输出模板结合奖励机制的做法，为提升模型推理透明度和可控性提供了新思路，可尝试应用于对话系统等领域；此外，奖励函数中对冗余行为的惩罚机制启发我思考是否可以通过动态调整惩罚强度，进一步优化模型决策的精细度。