---
title: "Benchmarking Information Retrieval Models on Complex Retrieval Tasks"
pubDatetime: 2025-09-08T22:11:10+00:00
slug: "2025-09-complex-retrieval-benchmark"
type: "arxiv"
id: "2509.07253"
score: 0.8079779027045022
author: "grok-3-latest"
authors: ["Julian Killingback", "Hamed Zamani"]
tags: ["LLM", "Information Retrieval", "Complex Tasks", "Benchmarking", "Query Rewriting"]
institution: ["University of Massachusetts Amherst"]
description: "本文通过构建CRUMB基准，评估了最先进的检索模型在复杂检索任务上的表现，揭示其性能不足，并为未来模型改进提供了统一评估资源和研究方向。"
---

> **Summary:** 本文通过构建CRUMB基准，评估了最先进的检索模型在复杂检索任务上的表现，揭示其性能不足，并为未来模型改进提供了统一评估资源和研究方向。 

> **Keywords:** LLM, Information Retrieval, Complex Tasks, Benchmarking, Query Rewriting

**Authors:** Julian Killingback, Hamed Zamani

**Institution(s):** University of Massachusetts Amherst


## Problem Background

随着大型语言模型（LLMs）的广泛应用，用户对信息检索系统的期望提高，期望其能处理包含多方面、多约束的复杂检索任务。然而，现有检索基准（如TREC、MSMARCO）主要聚焦简单查询，缺乏对复杂任务的全面评估资源，限制了对检索模型真实能力的理解和下一代模型的创新。

## Method

* **基准构建（CRUMB）**：作者从现有数据集中改编了8个复杂检索任务（包括Tip-of-the-Tongue、StackExchange QA、Paper Retrieval等），每个任务的查询具有多方面或多约束特性，覆盖不同领域和场景；文档被标准化为Markdown格式，并采用上下文化分块策略（Contextualized Chunking），保留标题等结构信息以提升语义理解。
* **模型评估**：选择了多种最先进的神经检索模型（如GTE Qwen 7B、Promptriever）和基线BM25，评估指标包括nDCG@10（排名精度）、R@100和R@1000（召回能力）；模型选择考虑了性能、规模、训练数据和检索范式（密集与稀疏）的多样性。
* **查询重写实验**：探索了基于LLM的查询重写对性能的影响，采用三种策略：Query-to-Answer CoT（生成直接答案）、Query-to-Doc CoT（生成相关文档）和Query-as-Reasoning-Trace（使用推理过程作为查询），以增强查询表达能力。

## Experiment

* **整体效果**：即使最先进的检索模型在复杂任务上的表现也不佳，平均nDCG@10仅为0.346，R@100为0.587，远低于简单检索数据集上的表现，表明复杂任务对模型提出了巨大挑战。
* **模型表现**：GTE Qwen 7B、Promptriever和GTE Qwen 1.5B表现最佳，模型规模和训练数据多样性对性能有显著影响；任务特性（如术语重叠低、逻辑操作）导致性能差异，例如SetOps任务因集合逻辑操作导致模型普遍表现不佳。
* **查询重写**：LLM辅助的查询重写对较弱模型有帮助（如Snowflake的nDCG@10从0.182提升到0.237），但对最强模型（如GTE Qwen 7B）反而有害，可能是由于引入噪声或改变查询分布。
* **实验设置**：实验覆盖多种模型、任务和指标，设置较为全面；但部分任务文档集合较小（如StackExchange），可能不完全反映真实大规模场景；查询重写仅使用一种LLM（Gemma-3 27B），可能存在模型偏见。

## Further Thoughts

CRUMB基准通过整合多样化的复杂任务，为检索模型泛化能力评估提供了新思路，未来可扩展至更多跨领域或多语言任务；上下文化分块策略启发我们利用文档结构（如表格、标题）增强语义表示；指令微调显著提升了Promptriever等模型性能，提示可以在检索模型中引入查询级别指令以动态调整行为；查询重写的局限性表明需探索联合优化检索与查询生成的方法，避免噪声干扰。