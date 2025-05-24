---
title: "Shared Path: Unraveling Memorization in Multilingual LLMs through Language Similarities"
pubDatetime: 2025-05-21T16:30:18+00:00
slug: "2025-05-multilingual-memorization-similarity"
type: "arxiv"
id: "2505.15722"
score: 0.7706600659215922
author: "grok-3-latest"
authors: ["Xiaoyu Luo", "Yiyi Chen", "Johannes Bjerva", "Qiongxiu Li"]
tags: ["Multilingual LLM", "Memorization", "Language Similarity", "Cross-lingual Transfer", "Graph Analysis"]
institution: ["Aalborg University"]
description: "本文通过语言相似性图分析多语言大型语言模型中的记忆化行为，揭示低资源语言在相似语言群中更高的记忆化率，强调语言感知视角对评估和缓解记忆化风险的重要性。"
---

> **Summary:** 本文通过语言相似性图分析多语言大型语言模型中的记忆化行为，揭示低资源语言在相似语言群中更高的记忆化率，强调语言感知视角对评估和缓解记忆化风险的重要性。 

> **Keywords:** Multilingual LLM, Memorization, Language Similarity, Cross-lingual Transfer, Graph Analysis

**Authors:** Xiaoyu Luo, Yiyi Chen, Johannes Bjerva, Qiongxiu Li

**Institution(s):** Aalborg University


## Problem Background

多语言大型语言模型（Multilingual Large Language Models, MLLMs）因其训练数据的长尾分布（高资源语言数据多，低资源语言数据少）和语言间的相似性，面临独特的记忆化（Memorization）问题，即模型可能记住训练数据，导致隐私泄露或版权风险。
传统研究主要聚焦单语言模型，认为记忆化与训练数据量高度相关，但这一假设在多语言场景下并不完全成立，语言间的关系可能对记忆化模式产生重要影响。

## Method

*   **核心思想:** 提出一种语言感知（language-aware）视角，通过语言相似性分析多语言模型中的记忆化行为，挑战传统数据量决定记忆化的假设。
*   **具体实现:** 
    *   **语言相似性度量:** 从多语言模型的嵌入空间中提取语言特定子空间（language-specific subspace），使用余弦相似性计算语言对之间的相似度。
    *   **图构建与分析:** 将语言表示为图中的节点，边表示相似性（通过阈值控制稀疏性），定义图平滑度（graph smoothness）和图交叉平滑度（graph cross-smoothness）来分析记忆化率和训练数据量在语言拓扑上的变化。
    *   **图相关系数:** 提出一种新的图相关系数（Graph-based Correlation Coefficient），捕捉记忆化与数据量在语言相似性结构下的关系，超越传统 Pearson 相关性分析。
    *   **拓扑分析:** 区分内部拓扑（intra-topology，相似语言群）和跨拓扑（cross-topology，不同语言群），进一步揭示记忆化模式。
*   **关键点:** 该方法通过结构化语言关系，揭示了传统分析无法捕捉的记忆化趋势，尤其是在相似语言群中的低资源语言。

## Experiment

*   **有效性:** 实验表明，通过图相关系数（ρ_G），记忆化与训练数据量在相似语言群中呈显著负相关，即低资源语言往往有更高的记忆化率，这一趋势在传统 Pearson 相关性分析中不明显。
*   **全面性与合理性:** 实验覆盖 95 种语言，涉及多种模型架构（M T5 编码器-解码器系列，M GPT 解码器系列）和规模（300M 到 13B 参数），采用多种记忆化度量（Exact Memorization, Relaxed Memorization, Reconstruct Likelihood Memorization），并在不同提示长度和模型规模下验证了结果一致性。
*   **额外发现:** 提示长度和模型规模对记忆化有影响，较长提示和较大模型（如 M GPT-13B）通常记忆化率更高，但 M T5-Large 因生成不稳定而例外；语言级记忆化分布在不同设置下保持稳定，表明记忆化是语言内在特性。
*   **局限性:** 实验主要基于预训练模型，未探索微调或指令调整的影响。

## Further Thoughts

论文提出的语言相似性图分析方法具有广泛适用性，不仅限于记忆化研究，还可推广至跨语言迁移（Cross-lingual Transfer）或语言间干扰分析；此外，低资源语言在相似语言群中更高记忆化率的发现，提示我们在多语言模型设计中需特别关注低资源语言的隐私风险，可能通过针对性去噪或数据增强来缓解。