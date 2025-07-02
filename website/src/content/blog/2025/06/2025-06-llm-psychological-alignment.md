---
title: "Measuring How LLMs Internalize Human Psychological Concepts: A preliminary analysis"
pubDatetime: 2025-06-29T01:56:56+00:00
slug: "2025-06-llm-psychological-alignment"
type: "arxiv"
id: "2506.23055"
score: 0.6986079573596783
author: "grok-3-latest"
authors: ["Yuki Yamada", "Hiro Taiyo Hamada", "Ippei Fujisawa", "Genji Kawakita"]
tags: ["LLM", "Concept Alignment", "Psychological Concepts", "Semantic Similarity", "Classification"]
institution: ["Kyushu University", "Araya Inc.", "Imperial College London"]
description: "本文提出了一种基于心理问卷的定量框架，系统评估大型语言模型内化人类心理概念的能力，验证了其与人类响应的显著相关性，为 AI 与人类认知对齐研究提供了新工具。"
---

> **Summary:** 本文提出了一种基于心理问卷的定量框架，系统评估大型语言模型内化人类心理概念的能力，验证了其与人类响应的显著相关性，为 AI 与人类认知对齐研究提供了新工具。 

> **Keywords:** LLM, Concept Alignment, Psychological Concepts, Semantic Similarity, Classification

**Authors:** Yuki Yamada, Hiro Taiyo Hamada, Ippei Fujisawa, Genji Kawakita

**Institution(s):** Kyushu University, Araya Inc., Imperial College London


## Problem Background

大型语言模型（LLMs）在生成类人文本方面表现出色，但其是否能准确内化塑造人类思维和行为的心理概念（Psychological Concepts）尚不明确。
随着 LLMs 在心理学等领域的应用日益广泛，理解其与人类认知的对齐程度（Concept Alignment）至关重要，以揭示潜在的表示偏差并提升模型可解释性。
本文试图解决如何定量评估 LLMs 与人类心理维度的对齐程度，并验证其是否能重现心理概念的分类和语义关系这一关键问题。

## Method

*   **核心思想**：通过标准化心理问卷作为基准，定量评估 LLMs 对人类心理概念的内化能力，分析其分类性能和语义关系保留程度。
*   **数据准备**：从‘Psychological Scales’数据库中提取 43 个英文心理问卷，每份包含 30 个或更少的文本项，涵盖 2-6 个专家标注的心理构建子类别（如好奇心、焦虑）。
*   **成对相似性分析**：使用 6 个语言模型（BERT、OpenAI Embedding、GPT-3.5 和 GPT-4 的不同版本）计算问卷项间的语义相似性；对于 BERT 和嵌入模型，采用余弦相似性构建矩阵；对于 GPT 模型，通过提示要求在 [-1, 1] 范围内评分（1 为完全相似，-1 为完全相反）。
*   **分类与聚类**：对相似性矩阵应用层次聚类（Hierarchical Clustering），以预定类别数进行分类，并与专家标注的真实类别对比，计算分类准确率和调整后的兰德指数（ARI）。
*   **跨问卷语义分析**：计算不同问卷间的项间相似性，构建语义矩阵，并与人类响应的皮尔逊相关系数对比，验证 LLMs 是否保留心理构建间的语义关系。
*   **额外测试**：探讨提示类型（连续 vs 离散 Likert 量表）和问卷项顺序（Order Effect）对分类性能的影响。
*   **关键点**：该方法利用标准化问卷作为客观基准，结合相似性分析和聚类技术，系统评估模型对心理概念的理解能力。

## Experiment

*   **分类性能**：GPT-4（版本 1106）表现最佳，分类准确率达 66.2%，显著高于 GPT-3.5（55.9%）和 BERT（48.1%），所有模型均超过随机基线（31.9%）；统计检验显示 GPT-4 优于 GPT-3.5（p < 0.001）。
*   **提示类型影响**：连续和离散提示在 GPT-4 上的分类准确率无显著差异（p=0.45 和 p=0.69），表明模型对提示形式不敏感。
*   **顺序效应**：在 8 个问卷中，部分问卷（如 WDGOI 和 FFMQ-24）在项顺序反转后分类性能显著变化（p < 0.001），揭示 LLMs 对输入顺序的敏感性。
*   **语义关系保留**：GPT-4 估计的语义相似性与人类响应的皮尔逊相关系数显著相关（均值 r=0.412, p<0.05；中位数 r=0.731, p<1e-5），表明 LLMs 能保留心理构建间的语义关系。
*   **实验设置评估**：实验设计全面，涵盖多模型对比、分类与语义关系任务、以及影响因素分析；结果显示 GPT-4 在内化心理概念方面有显著提升，但顺序效应揭示了模型局限性。

## Further Thoughts

本文提出的基于心理问卷的概念对齐评估框架，不仅适用于心理学研究，还可能扩展到文化语义或伦理判断等领域，为构建更可解释的 AI 系统提供新思路；此外，语义相似性可视化可作为研究人类认知结构的新工具，尤其在跨文化研究中；顺序效应的发现提示我们在模型交互设计中需考虑输入顺序的标准化，或探索对顺序不敏感的模型架构；进一步思考，是否可以通过特定训练数据或微调策略，让 LLMs 更好地内化特定文化或群体的心理概念？