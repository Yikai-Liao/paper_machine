---
title: "Do Large Language Models Excel in Complex Logical Reasoning with Formal Language?"
pubDatetime: 2025-05-22T17:57:23+00:00
slug: "2025-05-formal-reasoning-evaluation"
type: "arxiv"
id: "2505.16998"
score: 0.7942759348153536
author: "grok-3-latest"
authors: ["Jin Jiang", "Jianing Wang", "Yuchen Yan", "Yang Liu", "Jianhua Zhu", "Mengdi Zhang", "Xunliang Cai", "Liangcai Gao"]
tags: ["LLM", "Logical Reasoning", "Formal Language", "Generalization", "Data Augmentation"]
institution: ["Peking University", "Meituan Group", "Zhejiang University"]
description: "本文通过系统性评估揭示了大型语言模型在形式化语言逻辑推理中的能力和局限，并提出拒绝微调方法显著增强小模型性能。"
---

> **Summary:** 本文通过系统性评估揭示了大型语言模型在形式化语言逻辑推理中的能力和局限，并提出拒绝微调方法显著增强小模型性能。 

> **Keywords:** LLM, Logical Reasoning, Formal Language, Generalization, Data Augmentation

**Authors:** Jin Jiang, Jianing Wang, Yuchen Yan, Yang Liu, Jianhua Zhu, Mengdi Zhang, Xunliang Cai, Liangcai Gao

**Institution(s):** Peking University, Meituan Group, Zhejiang University


## Problem Background

大型语言模型（LLMs）在复杂逻辑推理任务中表现出突破性性能，但现有研究多集中于使用形式化语言引导模型生成可靠推理路径，而对其能力的系统性评估仍显不足。
论文旨在全面评估 LLMs 在利用形式化语言解决各类逻辑推理问题时的表现，揭示其局限性，并探索如何通过数据增强提升模型能力。

## Method

*   **评估框架设计**：构建了一个涵盖三个维度的系统性评估架构，包括模型种类（Thinking 和 Instruct 模型）、任务类型（演绎、归纳、溯因及混合推理）和轨迹格式（自然语言 Text 及形式化语言如 Python/PoT, Z3, CSP），使用 66 个数据集进行全面测试。
*   **零样本评估**：在不提供任务特定训练的情况下，测试模型在不同轨迹格式下的推理表现，分析其在不同任务类型中的能力差异和局限性。
*   **泛化能力分析**：通过粗粒度（按任务类型或格式分组）和细粒度（任务类型与格式组合）实验，评估模型在训练与测试数据格式或任务类型不一致时的迁移能力，揭示形式化语言间的结构差异对泛化的影响。
*   **数据增强与拒绝微调（RFT）**：提出一种拒绝微调方法，利用 GPT-4o 作为教师模型，通过多次采样生成形式化语言推理数据，并筛选出可执行且正确的样本，用于增强小模型在形式化推理任务上的能力，训练数据量控制一致以确保公平性。
*   **核心思想**：通过多维度评估揭示 LLMs 在形式化语言推理中的真实能力，并利用高质量数据增强小模型性能，探索任务-格式对齐对推理效果的影响。

## Experiment

*   **模型表现**：Thinking 模型（如 QwQ-32B）在大多数任务中显著优于 Instruct 模型，尤其在使用形式化语言时，表明其推理模式更适合复杂逻辑任务。
*   **任务与格式偏好**：不同任务对轨迹格式有明显偏好，例如 PoT 格式在结构化任务（如数学计算）中表现最佳，Z3 格式适合严格逻辑任务，而 Text 格式在语言理解任务中更优。
*   **局限性**：所有模型在归纳推理任务中表现较差，无论是否使用形式化语言；小模型在形式化语言任务上的表现尤其不佳，复杂任务中性能下降显著。
*   **泛化能力**：PoT 格式数据在跨格式泛化中表现最佳（例如对 Text, Z3, CSP 均有正向迁移），而 CSP 格式泛化能力较差；Deductive-CSP 任务最容易通过训练数据提升。
*   **数据增强效果**：通过 RFT 方法，小模型（如 Qwen2.5-7B）在形式化语言任务上的准确率显著提升，例如 CSP 格式从 20.0% 提升至 37.0%，执行率从 52.2% 提升至 68.1%，整体平均准确率从 34.0% 提升至 42.0%，与开源模型相比竞争力增强。
*   **实验设置合理性**：实验覆盖了从 7B 到 72B 的多种模型、66 个数据集、多种任务类型和轨迹格式，零样本设置避免了任务特定优化偏差，数据增强实验中训练数据量控制一致，确保公平性，整体设置全面合理。

## Further Thoughts

任务-格式对齐的概念非常具有启发性，未来可以设计自适应机制，在推理时根据任务特性动态选择最适合的形式化语言格式，甚至结合多格式推理结果（如通过多数投票）进一步提升性能；此外，拒绝微调（RFT）方法展示了通过高质量数据增强小模型潜力的可能性，未来可以探索结合强化学习或更复杂的采样策略，进一步优化数据生成质量。