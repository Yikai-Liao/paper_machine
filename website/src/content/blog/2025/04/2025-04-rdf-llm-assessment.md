---
title: "RDF-Based Structured Quality Assessment Representation of Multilingual LLM Evaluations"
pubDatetime: 2025-04-30T13:06:40+00:00
slug: "2025-04-rdf-llm-assessment"
type: "arxiv"
id: "2504.21605"
score: 0.3512201277968338
author: "grok-3-latest"
authors: ["Jonas Gwozdz", "Andreas Both"]
tags: ["LLM", "Quality Assessment", "Multilingual Evaluation", "Knowledge Conflict", "Structured Representation"]
institution: ["Leipzig University of Applied Sciences"]
description: "本文提出一种基于 RDF 的框架，结构化表示多语言 LLM 质量评估结果，通过消防安全领域实验揭示上下文主导性和语言特定性能差异，支持标准化和可查询分析。"
---

> **Summary:** 本文提出一种基于 RDF 的框架，结构化表示多语言 LLM 质量评估结果，通过消防安全领域实验揭示上下文主导性和语言特定性能差异，支持标准化和可查询分析。 

> **Keywords:** LLM, Quality Assessment, Multilingual Evaluation, Knowledge Conflict, Structured Representation

**Authors:** Jonas Gwozdz, Andreas Both

**Institution(s):** Leipzig University of Applied Sciences


## Problem Background

大型语言模型（LLMs）作为知识接口广泛应用，但其在处理冲突信息时的可靠性评估面临挑战，尤其是在关键领域中事实准确性至关重要；现有评估方法缺乏标准化的结构化表示，特别是在多语言场景下，模型性能差异及上下文与训练知识的交互作用未被充分探索，且评估结果未遵循 FAIR 原则（Findable, Accessible, Interoperable, Reusable），限制了其可用性。

## Method

* **核心思想**：提出一种基于 RDF（资源描述框架）的框架，用于结构化表示多语言 LLM 质量评估结果，以实现标准化、可查询和可重用的评估分析。
* **RDF 词汇设计**：构建包含 14 个类（如 :Question, :Answer）和 57 个属性（如 :isValid）的 RDF 词汇，支持多语言评估和四种上下文条件（完整、缺失、冲突、无上下文）的表示；词汇符合 FAIR 原则，与 PROV Ontology 和 Dublin Core 标准对齐，支持 SPARQL 查询和外部知识图谱链接。
* **上下文条件测试**：设计四种上下文条件测试 LLM 响应，揭示模型在不同信息环境下的行为，如是否优先上下文或训练知识。
* **多语言支持**：通过语言标记的字面量（如 :hasText）支持多语言评估，捕捉模型在不同语言（如德语和英语）中的行为差异。
* **数据模型验证**：通过消防安全领域的实验验证框架，分析知识泄露（模型偏向训练数据而非上下文）、错误检测和跨语言一致性。

## Experiment

* **有效性**：实验在消防安全领域使用 GPT-4o-mini 和 Gemini-2.0-Flash 模型，针对 28 个问题在德语和英语下测试四种上下文条件，发现模型在 89-93% 的冲突情况下优先复制上下文（即使错误），揭示了上下文主导性。
* **多语言差异**：英语模型在不完整信息条件下表现更好，德语模型在无上下文条件下展现更强基准知识，表明语言特定性能差异。
* **统计显著性**：成对比较显示德语上下文下模型差异不显著（p > 0.05），但英语无上下文条件下 GPT-4o-mini 显著优于 Gemini-2.0-Flash（p = 0.0039，准确率差距 32.1 个百分点）；置信区间较宽，部分结论需更多数据支持。
* **实验设置合理性**：设计全面，涵盖多种上下文和语言，问题数量适中，SPARQL 查询验证了 RDF 框架实用性，但样本量和语言覆盖有扩展空间。

## Further Thoughts

RDF 作为结构化评估工具的潜力令人瞩目，其语义丰富的数据表示和 SPARQL 查询能力可扩展到其他 AI 评估领域（如多模态模型）；上下文与训练知识冲突的系统化研究方法启发了对模型决策机制的深入探索；多语言评估揭示的性能差异提示在全球化应用中需关注语言和文化背景。