---
title: "CoT-Self-Instruct: Building high-quality synthetic prompts for reasoning and non-reasoning tasks"
pubDatetime: 2025-07-31T17:38:50+00:00
slug: "2025-07-cot-self-instruct"
type: "arxiv"
id: "2507.23751"
score: 0.6929299073158118
author: "grok-3-latest"
authors: ["Ping Yu", "Jack Lanchantin", "Tianlu Wang", "Weizhe Yuan", "Olga Golovneva", "Ilia Kulikov", "Sainbayar Sukhbaatar", "Jason Weston", "Jing Xu"]
tags: ["LLM", "Synthetic Data", "Chain of Thought", "Instruction Tuning", "Data Filtering"]
institution: ["FAIR at Meta", "NYU"]
description: "本文提出 CoT-Self-Instruct 方法，通过链式思维引导大型语言模型生成高质量合成指令，并结合自动过滤机制显著提升推理和非推理任务的表现。"
---

> **Summary:** 本文提出 CoT-Self-Instruct 方法，通过链式思维引导大型语言模型生成高质量合成指令，并结合自动过滤机制显著提升推理和非推理任务的表现。 

> **Keywords:** LLM, Synthetic Data, Chain of Thought, Instruction Tuning, Data Filtering

**Authors:** Ping Yu, Jack Lanchantin, Tianlu Wang, Weizhe Yuan, Olga Golovneva, Ilia Kulikov, Sainbayar Sukhbaatar, Jason Weston, Jing Xu

**Institution(s):** FAIR at Meta, NYU


## Problem Background

大型语言模型（LLMs）的训练依赖于大量高质量数据，但人类生成的数据成本高、稀缺且可能存在偏见或错误，合成数据成为替代方案；
现有合成数据生成方法（如 Self-Instruct）在质量和有效性上不足，尤其是在复杂推理和非推理任务的指令生成方面，论文旨在解决如何生成高质量合成指令以提升模型性能的问题。

## Method

* **核心思想**：提出 CoT-Self-Instruct 方法，利用链式思维（Chain-of-Thought, CoT）引导大型语言模型（LLMs）基于少量种子指令生成高质量合成指令，并通过自动过滤机制筛选优质数据用于训练。
* **具体步骤**：
  * **合成指令生成**：从种子指令池中随机抽取少量高质量指令作为 few-shot 示例，提示 LLM 逐步分析种子指令的领域、复杂度和目的，然后通过 CoT 推理和规划生成新的合成指令；对于推理任务，同时生成指令和答案，对于非推理任务仅生成指令。
  * **数据筛选**：针对可验证推理任务，采用 Answer-Consistency 过滤方法，即通过多次生成答案并取多数投票结果，若与合成答案不一致则剔除数据；针对非可验证任务，采用 RIP（Rejecting Instruction Preferences）方法，基于奖励模型对多次生成响应的评分分布筛选高质量指令。
  * **模型训练**：利用筛选后的合成数据进行强化学习训练，对于推理任务使用 GRPO（基于规则验证奖励的强化学习），对于非推理任务使用 DPO（直接偏好优化），包括离线和在线两种模式。
* **关键创新**：通过 CoT 提升生成指令的质量，避免直接生成可能导致的低质量问题；结合自动过滤减少低质量数据对训练的影响，确保合成数据的有效性。

## Experiment

* **推理任务效果**：在 MATH500, AMC23, AIME24 和 GPQA-Diamond 等基准上，CoT-Self-Instruct 生成的合成数据显著优于 Self-Instruct 和现有数据集（如 s1k 和 OpenMathReasoning）；例如，使用 Answer-Consistency 过滤后，平均准确率从 53.0% 提升至 57.2%，明显优于 Self-Instruct 的 49.5%（未过滤）和 53.6%（过滤后）；增加数据量至 10,000 后，准确率进一步提升至 58.7%。
* **非推理任务效果**：在 AlpacaEval 2.0 和 Arena-Hard 基准上，CoT-Self-Instruct 结合 RIP 过滤后平均得分从 53.9 提升至 54.7，优于 Self-Instruct 的 47.4 和 49.1，也优于人类数据（如 WildChat）的 46.8 和 50.7；使用在线 DPO 训练后，性能进一步提升至 67.1，显著高于人类数据的 63.1。
* **实验设置合理性**：实验涵盖多种模型（如 Qwen3-4B 和 Llama 3.1-8B）、不同数据规模（从 893 到 10,000）、多种过滤策略（Self-Consistency, RIP, Answer-Consistency）以及领域分类和长度归一化等细节，避免了生成过长响应的偏差，设计全面且支持结论。

## Further Thoughts

CoT-Self-Instruct 的链式思维引导方法启发我们可以在其他生成任务中引入类似推理规划步骤，以提升复杂任务（如多步推理或跨领域任务）的数据质量；
自动过滤机制（如 Answer-Consistency 和 RIP）展示了利用 LLM 自身或奖励模型评估数据质量的潜力，未来可以探索多模型投票或更复杂的奖励机制来进一步优化筛选；
合成数据超越人类数据的表现提示在数据稀缺领域可更多依赖合成数据，同时需研究如何平衡合成与真实数据的比例以避免过拟合或偏见。