---
title: "Protoknowledge Shapes Behaviour of LLMs in Downstream Tasks: Memorization and Generalization with Knowledge Graphs"
pubDatetime: 2025-05-21T13:22:34+00:00
slug: "2025-05-protoknowledge-llm-kg"
type: "arxiv"
id: "2505.15501"
score: 0.7169642650251533
author: "grok-3-latest"
authors: ["Federico Ranaldi", "Andrea Zugarini", "Leonardo Ranaldi", "Fabio Massimo Zanzotto"]
tags: ["LLM", "Knowledge Graph", "Memorization", "Generalization", "Reasoning"]
institution: ["University of Rome Tor Vergata", "University of Edinburgh", "expert.ai"]
description: "本文提出‘protoknowledge’概念，揭示大型语言模型如何通过词汇、层级和拓扑形式内化知识图谱信息，并通过实验证明其对下游任务如 Text-to-SPARQL 的影响，同时为检测语义级数据污染提供分析框架。"
---

> **Summary:** 本文提出‘protoknowledge’概念，揭示大型语言模型如何通过词汇、层级和拓扑形式内化知识图谱信息，并通过实验证明其对下游任务如 Text-to-SPARQL 的影响，同时为检测语义级数据污染提供分析框架。 

> **Keywords:** LLM, Knowledge Graph, Memorization, Generalization, Reasoning

**Authors:** Federico Ranaldi, Andrea Zugarini, Leonardo Ranaldi, Fabio Massimo Zanzotto

**Institution(s):** University of Rome Tor Vergata, University of Edinburgh, expert.ai


## Problem Background

大型语言模型（LLMs）在预训练过程中通过记忆大量数据（包括知识图谱的结构化信息）展现出强大的能力，但如何将这些记忆转化为可复用的知识并通过泛化应用于下游任务仍是一个核心问题。
论文旨在研究 LLMs 如何内化知识图谱（KGs）中的信息（定义为‘protoknowledge’），以及这种内化能力如何影响下游任务（如 Text-to-SPARQL）的表现，同时探讨预训练数据中的语义偏见（semantic bias）对模型泛化能力的限制和闭合预训练模型中语义级数据污染的检测方法。

## Method

*   **核心概念：Protoknowledge**：论文提出‘protoknowledge’概念，将 LLMs 内化的知识图谱信息分为三种形式：
    *   **词汇（Lexical）**：模型基于自然语言表面形式（如标签、别名）回忆实体和属性的能力，涉及符号映射。
    *   **层级（Hierarchical）**：模型识别和推理知识图谱中分类关系（如 subclassOf, instanceOf）的能力，涉及结构化理解。
    *   **拓扑（Topological）**：模型推断和遍历实体间多跳关系路径的能力，涉及复杂图结构推理。
*   **评估方法：知识激活任务（KATs）**：设计针对每种 protoknowledge 的特定任务，通过专用测试集评估模型表现：
    *   词汇任务：标签到 URI 的翻译任务，测试符号识别能力。
    *   层级任务：直接和逆向 subsumption 任务，测试分类关系的推理能力。
    *   拓扑任务：SVO 三元组补全任务及 SPS 评分，测试多跳推理能力。
*   **下游任务分析：Text-to-SPARQL**：通过不同提示策略（Original, No Label, No URI）测试 protoknowledge 在实际任务中的影响，分析上下文支持减少时模型对内化知识的依赖程度。
*   **分析框架**：提出逐查询分析框架，评估 protoknowledge 激活是否与任务成功相关，定义‘正向一致性（Positive Agreement）’等指标，为语义级数据污染检测提供工具。

## Experiment

*   **模型与数据集**：实验在多个 LLMs（如 GPT-4, GPT-3.5 Turbo, Llama-3 系列）上进行，使用 DBpedia 和 Wikidata 知识图谱构建测试集，覆盖词汇、层级和拓扑 protoknowledge 的评估及 Text-to-SPARQL 任务。
*   **效果显著性**：
    *   词汇 protoknowledge：GPT-4 在 URI 识别任务中准确率最高（高达 74.35%），尤其在高频实体和属性上表现优异，Llama 模型表现较弱。
    *   层级 protoknowledge：GPT-4 在直接和逆向 subsumption 任务中表现最佳（部分类别准确率超 90%），对稀有类别也展现较强泛化能力。
    *   拓扑 protoknowledge：GPT 模型在 DBpedia 上的 SPS 评分高于 Wikidata，模型规模越大（如 Llama-3 70B 对比 8B）表现越好。
    *   Text-to-SPARQL：随着提示上下文减少（从 Original 到 No URI），性能下降明显，但 GPT-4 表现最稳定（F1 评分在 No URI 下仍达 25.14%-29.09%），表明其 protoknowledge 激活能力更强。
*   **实验合理性与局限**：实验设置全面，覆盖不同模型、任务和知识图谱，但仅限于 DBpedia 和 Wikidata，未扩展到其他知识图谱或更多数据集（如 LC-QuAD）；语义偏见显著，模型对高频项目表现更好，反映预训练数据分布的影响。

## Further Thoughts

论文提出的‘protoknowledge’概念启发我们从多层次视角理解 LLMs 的知识内化能力，是否可以通过针对性预训练数据或任务设计增强特定形式（如拓扑）的 protoknowledge，以提升复杂推理任务表现？此外，语义偏见的发现提示未来模型训练需关注数据分布平衡，或通过去偏技术减少对高频内容的依赖；分析框架为检测语义级数据污染提供了新思路，是否可扩展到其他任务（如常识推理）以评估模型的知识污染程度？