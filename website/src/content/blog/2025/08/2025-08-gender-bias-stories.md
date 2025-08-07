---
title: "Investigating Gender Bias in LLM-Generated Stories via Psychological Stereotypes"
pubDatetime: 2025-08-05T10:10:26+00:00
slug: "2025-08-gender-bias-stories"
type: "arxiv"
id: "2508.03292"
score: 0.6252059014260978
author: "grok-3-latest"
authors: ["Shahed Masoudian", "Gustavo Escobedo", "Hannah Strauss", "Markus Schedl"]
tags: ["LLM", "Gender Bias", "Stereotypes", "Narrative Generation", "Prompt Design"]
institution: ["Johannes Kepler University", "Linz Institute of Technology", "University of Innsbruck"]
description: "本文通过心理学刻板印象提示设计，揭示了大型语言模型在儿童故事生成中的性别偏见，发现属性条件和模型规模显著影响偏见表现。"
---

> **Summary:** 本文通过心理学刻板印象提示设计，揭示了大型语言模型在儿童故事生成中的性别偏见，发现属性条件和模型规模显著影响偏见表现。 

> **Keywords:** LLM, Gender Bias, Stereotypes, Narrative Generation, Prompt Design

**Authors:** Shahed Masoudian, Gustavo Escobedo, Hannah Strauss, Markus Schedl

**Institution(s):** Johannes Kepler University, Linz Institute of Technology, University of Innsbruck


## Problem Background

大型语言模型（LLMs）在生成内容时可能放大训练数据中的性别偏见，而现有研究多集中于显性偏见或短文本任务，忽略了长篇叙事生成中隐性偏见的表现形式。
本文旨在探究 LLMs 在开放式儿童故事生成任务中如何受到心理学刻板印象的影响，揭示这些刻板印象（如男性与‘侵略性’、女性与‘八卦’的关联）如何塑造生成内容的性别分布。

## Method

*   **核心思想：** 通过基于心理学刻板印象的提示设计，分析 LLMs 在叙事生成中的性别偏见，探索不同条件下的偏见变化。
*   **数据集构建：** 创建 StereoBias-Stories (SBS) 数据集，包含约 15 万个由五个 LLMs 生成的儿童故事，模型来自 OpenAI（GPT-4o、GPT-4o-mini）和 DeepSeek-R1（R1-7B、R1-32B、R1-70B）系列。
*   **属性选择与分类：** 从心理学文献中选取 25 个性别刻板印象属性（如‘侵略性’与男性相关，‘关怀’与女性相关）和 3 个任务相关结局属性（如‘悲伤结局’），按性别关联（男性、女性、中性）和情感倾向（积极、消极、中性）分类。
*   **提示设置：** 设计四种提示条件：无属性（No-Attribute，作为基线）、单一属性（Single-Attribute）、双属性（Two-Attribute）和多属性（Multi-Attribute，六个属性组合），以观察不同条件对性别偏见的影响。
*   **偏见度量：** 通过统计故事中性别标识符（如‘他’、‘她’或具体名字）的比例计算性别贡献（Gender Contribution），定义性别差距（Gender Gap）为男性与女性贡献的差值，并计算相对于无属性基线的变化（ΔGap），以量化属性条件对偏见的影响。
*   **分析维度：** 评估模型规模、属性组合和情感倾向对偏见的作用，并通过与心理学刻板印象的 alignment 分析模型行为与人类偏见的一致性。

## Experiment

*   **数据集质量：** 通过词汇指标（困惑度、N-gram 多样性）、用户研究和模型自评验证 SBS 数据集质量，GPT-4o 在质量和属性表达上表现最佳，R1-7B 表现最差。
*   **偏见表现：** 在无属性条件下，大多数模型（除 R1-7B 外）表现出显著男性偏见（男性贡献 0.61，女性 0.39）；引入属性条件后偏见减少，单一和双属性条件下效果最明显，多属性条件下偏见略有回升（可能因数据稀疏性）。
*   **属性与情感效应：** 男性刻板印象属性（如‘侵略性’）放大男性偏见，女性属性（如‘关怀’）缓解偏见；同性别属性组合强化偏见，异性别属性组合相互抵消；消极和中性情感下偏见符合刻板印象，积极情感下男性属性也可能缓解偏见。
*   **模型规模影响：** 较大模型（如 GPT-4o 和 R1-70B）与心理学刻板印象一致性更高（平均 60.1%，GPT-4o 达 64.7%），表明规模增加可能使其更接近人类社会偏见。
*   **实验设置评价：** 实验覆盖多种提示条件、模型规模和评估方法，设置全面合理；但多属性条件数据稀疏性和仅关注二元性别是局限，部分结果需谨慎解读。

## Further Thoughts

心理学刻板印象框架为评估 LLMs 偏见提供了新视角，未来可扩展至其他社会偏见或跨文化研究；属性组合效应提示通过平衡提示设计可能缓解偏见；模型规模与偏见一致性的关系引发思考，是否可以通过调整训练数据或规模控制偏见程度，或许在训练阶段引入反向刻板印象数据能进一步减少生成内容的偏见。