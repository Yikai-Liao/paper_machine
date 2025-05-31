---
title: "What Has Been Lost with Synthetic Evaluation?"
pubDatetime: 2025-05-28T20:12:32+00:00
slug: "2025-05-synthetic-evaluation-loss"
type: "arxiv"
id: "2505.22830"
score: 0.5281641532106033
author: "grok-3-latest"
authors: ["Alexander Gill", "Abhilasha Ravichander", "Ana Marasovic"]
tags: ["LLM", "Synthetic Data", "Evaluation Benchmark", "Reading Comprehension", "Reasoning"]
institution: ["University of Utah", "University of Washington"]
description: "本文通过对比 LLM 生成与人类创作的阅读理解评估基准，揭示了合成数据在有效性上表现良好但难度不足的问题，为未来评估数据生成提供了关键洞见。"
---

> **Summary:** 本文通过对比 LLM 生成与人类创作的阅读理解评估基准，揭示了合成数据在有效性上表现良好但难度不足的问题，为未来评估数据生成提供了关键洞见。 

> **Keywords:** LLM, Synthetic Data, Evaluation Benchmark, Reading Comprehension, Reasoning

**Authors:** Alexander Gill, Abhilasha Ravichander, Ana Marasovic

**Institution(s):** University of Utah, University of Washington


## Problem Background

随着大型语言模型（LLMs）越来越多地用于数据生成，特别是在创建评估基准（evaluation benchmarks）方面，研究者关注其是否能替代传统的人工众包（crowdsourcing）来构建高质量数据集。
论文探讨的核心问题是：LLM 生成的阅读理解基准与人类创作的基准相比，在有效性（validity）和难度（difficulty）上是否存在显著差异，尤其是在测试文本推理能力时。

## Method

*   **核心思想:** 使用大型语言模型（LLMs）生成阅读理解评估基准，重点是针对否定推理（CondaQA 数据集）和数量推理（DROP 数据集）的复杂问题和对比编辑（contrastive edits），并评估其与人类创作内容的质量差异。
*   **提示工程（Prompt Engineering）:** 采用 gpt-4-turbo-2024-04-09 模型，通过迭代优化提示生成符合原始标注指南的内容。初始提示基于人类标注指南，后调整为更适合 LLM 的形式，包括明确任务目标和约束条件（如问题需针对否定含义或要求离散推理）。
*   **过度生成后过滤（Overgenerate-then-Filter）:** 为每个段落生成多个候选问题或编辑（例如 8 个问题），然后通过自动化过滤（另一 LLM 调用）或手动验证，筛选出符合质量标准的内容，提升有效性。
*   **任务分解（Task Decomposition）:** 将复杂任务分解为子步骤，例如在生成 CondaQA 的 scope edit 时，先识别否定内容，再生成相关问题，最后编辑文本以改变答案，确保模型逐步理解任务需求。
*   **用户研究与验证:** 招募 NLP 研究人员进行偏好研究，比较人类和 LLM 生成内容是否符合标注指南；通过手动验证评估生成内容的有效性，确保其与原始数据集标准一致。
*   **基准测试（Benchmarking）:** 在多个模型（如 GPT-4-Turbo, GPT-4o, o3-mini, Llama-3.3-70B, Qwen2.5-72B）上测试人类和生成数据集的性能，衡量难度差异，使用准确率（accuracy）、token F1 和一致性（consistency）作为指标。

## Experiment

*   **有效性结果:** LLM 生成的内容在有效性上表现良好，例如 CondaQA 问题的有效性为 72.8%，DROP 问题为 88.7%，部分编辑类型（如 paraphrase 和 affirmative edits）接近 100%。用户研究显示，NLP 研究人员更偏好 LLM 生成内容，因其更严格遵循标注指南。
*   **难度对比:** 生成数据集在难度上显著低于人类创作数据集。多个模型在生成数据集上的准确率和一致性普遍更高，例如 GPT-4o 在 CondaQA 生成数据集准确率为 84.1%（人类数据集为 69.8%），在 DROP 上 token F1 从 61.7% 提升到 84.8%，表明生成数据对模型挑战性不足。
*   **实验设置合理性:** 实验覆盖两个不同类型的阅读理解数据集（否定推理和数量推理），测试多种模型以避免单一模型偏差，并通过用户研究和众包答案收集减少主观偏见。然而，样本量较小（尤其是 DROP 数据集）可能影响结果泛化性。
*   **显著性分析:** 难度差异显著，生成数据集性能提升在所有测试模型中几乎一致，暗示其可能存在系统性模式或缺乏人类创作的多样性和创造性。

## Further Thoughts

论文揭示了‘有效性不等于难度’的洞见，即使 LLM 生成内容符合标注指南并被人类评判为高质量，其挑战性仍可能不足。这启发我们未来在设计评估基准时，可以探索对抗性生成（adversarial generation）或结合人类反馈的迭代优化，增强生成数据的难度。此外，‘生成 AI 悖论’（LLM 能生成自己难以理解的内容）也提示我们可利用这一特性设计更具挑战性的评估任务。