---
title: "RDF-Based Structured Quality Assessment Representation of Multilingual LLM Evaluations"
pubDatetime: 2025-05-01T15:51:54Z
slug: "2025-04-rdf-llm-assessment"
type: "arxiv"
id: "2504.21605"
score: 0.8493116441508083
author: "grok-3-latest"
authors: ["Jonas Gwozdz", "Andreas Both"]
tags: ["LLM", "Quality Assessment", "Multilingual Evaluation", "Knowledge Conflict", "Structured Representation"]
institution: ["Leipzig University of Applied Sciences"]
description: "本文提出基于 RDF 的结构化框架，用于多语言大型语言模型质量评估，通过消防安全领域实验验证了其在捕捉知识冲突和语言特定行为方面的有效性。"
---

> **Summary:** 本文提出基于 RDF 的结构化框架，用于多语言大型语言模型质量评估，通过消防安全领域实验验证了其在捕捉知识冲突和语言特定行为方面的有效性。 

> **Keywords:** LLM, Quality Assessment, Multilingual Evaluation, Knowledge Conflict, Structured Representation
> **Recommendation Score:** 0.8493116441508083

**Authors:** Jonas Gwozdz, Andreas Both
**Institution(s):** Leipzig University of Applied Sciences

## Problem Background

大型语言模型（LLMs）作为知识接口日益普及，但在处理冲突信息时可靠性评估仍面临挑战，尤其是在关键领域中事实准确性至关重要。
现有评估方法缺乏标准化的结构化表示，特别是在多语言场景下，模型性能差异和上下文依赖性（训练数据与提供上下文的混淆）尚未被系统性研究，且评估结果未遵循 FAIR 原则（Findable, Accessible, Interoperable, Reusable），限制了其可用性和广泛应用。

## Method

*   **核心思想:** 提出一个基于 RDF（Resource Description Framework）的框架，用于结构化表示 LLMs 的质量评估结果，支持多语言和不同上下文条件下的系统分析。
*   **词汇表设计:** 构建一个 RDF 词汇表，包含 14 个类（如 :Question, :Answer）和 57 个属性（如 :isValid, :matchesFactual），用于描述问题、答案、验证结果和材料之间的关系，支持 SPARQL 查询以分析知识泄露和跨语言一致性。
*   **多语言支持:** 通过语言标记的字面量（如 :hasText）处理不同语言的响应，确保框架在多语言评估中的适用性。
*   **上下文条件:** 定义四种上下文条件（完整、缺失、冲突、无上下文），以测试 LLMs 在不同信息场景下的行为，如上下文优先级和训练知识依赖。
*   **数据完整性与标准:** 使用 OWL/SHACL 约束确保数据完整性，并通过 PROV Ontology 和 Dublin Core 等标准实现 FAIR 原则，支持推理和外部知识图谱链接。
*   **优势:** 相比传统 CSV 表格，RDF 框架支持联邦查询和语义推理，具有更高的灵活性和扩展性。

## Experiment

*   **有效性:** 在消防安全领域实验中，RDF 框架成功捕捉了 LLMs（如 GPT-4o-mini 和 Gemini-2.0-Flash）在德语和英语下的行为模式，例如在冲突上下文下模型倾向于复制错误信息（89-93%），而不是依赖训练知识。
*   **显著性:** 通过 SPARQL 查询和统计分析（如 McNemar 测试），发现多语言差异（英语模型在不完整信息下表现更好，德语模型在无上下文时基线知识更强），部分条件（如英语无上下文）模型间性能差异显著（p=0.0039，GPT-4o-mini 优于 Gemini-2.0-Flash）。
*   **合理性与局限:** 实验设置涵盖四种上下文条件和两种语言，领域选择合理（消防安全知识明确且准确性要求高），但样本量（28 个问题）较小，统计分析中部分条件不一致对数量不足（b+c<5），结论普适性有限，且未涉及低资源语言或更多领域。

## Further Thoughts

RDF 框架不仅适用于 LLMs 评估，还可扩展至其他 AI 系统的质量评估，尤其在跨领域、跨语言一致性分析中；上下文条件设计为研究‘知识泄露’提供了系统化思路，启发未来模型鲁棒性改进；FAIR 原则的应用提示构建开放、可重用数据集和框架的重要性，或将推动 AI 评估的社区协作与标准化。