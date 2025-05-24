---
title: "Protoknowledge Shapes Behaviour of LLMs in Downstream Tasks: Memorization and Generalization with Knowledge Graphs"
pubDatetime: 2025-05-21T13:22:34+00:00
slug: "2025-05-protoknowledge-llm-kg"
type: "arxiv"
id: "2505.15501"
score: 0.7169642650251533
author: "grok-3-latest"
authors: ["Federico Ranaldi", "Andrea Zugarini", "Leonardo Ranaldi", "Fabio Massimo Zanzotto"]
tags: ["LLM", "Knowledge Graph", "Memorization", "Generalization", "Semantic Bias"]
institution: ["University of Rome Tor Vergata", "School of Informatics, University of Edinburgh", "expert.ai"]
description: "本文通过定义和测量 Protoknowledge，揭示了大型语言模型如何通过记忆和泛化利用知识图谱信息，并提出一个分析框架评估其对下游任务的影响，同时指出了语义偏差的限制。"
---

> **Summary:** 本文通过定义和测量 Protoknowledge，揭示了大型语言模型如何通过记忆和泛化利用知识图谱信息，并提出一个分析框架评估其对下游任务的影响，同时指出了语义偏差的限制。 

> **Keywords:** LLM, Knowledge Graph, Memorization, Generalization, Semantic Bias

**Authors:** Federico Ranaldi, Andrea Zugarini, Leonardo Ranaldi, Fabio Massimo Zanzotto

**Institution(s):** University of Rome Tor Vergata, School of Informatics, University of Edinburgh, expert.ai


## Problem Background

大型语言模型（LLMs）在预训练过程中吸收了大量数据，包括知识图谱（Knowledge Graphs, KGs）等结构化信息，但如何通过记忆（Memorization）和泛化（Generalization）将这些信息应用于下游任务仍是一个未充分探索的问题。
论文关注 LLMs 是否能够有效内化知识图谱中的结构化知识（定义为 Protoknowledge），以及这种内化是否受到预训练数据的语义偏差（Semantic Bias）限制，尤其是在知识图谱相关任务（如 Text-to-SPARQL）中的表现。

## Method

*   **核心思想:** 提出 'Protoknowledge' 概念，将 LLMs 内化的知识图谱信息分为词汇型（Lexical）、层级型（Hierarchical）和拓扑型（Topological）三种形式，分别对应不同的知识抽象层次，从简单的实体识别到复杂的图结构推理。
*   **测量方式:** 设计知识激活任务（Knowledge Activation Tasks, KATs），针对每种 Protoknowledge 形式构造特定的测试任务和数据集，例如词汇型任务通过标签到 URI 的识别测试模型的实体映射能力，层级型任务通过类-子类关系推理测试模型的分类能力，拓扑型任务通过三元组补全测试模型的多跳推理能力。
*   **下游任务评估:** 在 Text-to-SPARQL 任务中，通过三种提示策略（Original, No Label, No URI）测试 Protoknowledge 在不同上下文支持下的影响，揭示模型对内化知识的依赖程度。
*   **分析框架:** 提出一个逐查询分析框架，评估 Protoknowledge 的激活是否与任务成功相关联，从而揭示语义偏差和数据污染的影响。
*   **关键点:** 方法不依赖于模型的微调或任务特定监督，而是直接利用预训练模型的黑箱测试，适用于闭源模型（Closed-Pretraining Models）的评估。

## Experiment

*   **有效性:** 实验在多个 LLMs（如 GPT-4、GPT-3.5 Turbo、Llama-3 系列）上进行，结果表明不同形式的 Protoknowledge 在 KATs 中表现出明显差异，例如 GPT-4 在词汇型任务（URI 识别）中准确率达 74.35%，远超 Llama-3-70B 的 35.90%；拓扑型 Protoknowledge 的 SPS 评分也显示 GPT-4 在精确激活知识方面更强。
*   **语义偏差影响:** 模型性能与知识图谱项的流行度（Popularity）高度相关，表明 Protoknowledge 受到预训练数据语义偏差的显著限制，尤其是在低频项上的表现较差。
*   **下游任务表现:** 在 Text-to-SPARQL 任务中，模型性能随上下文支持减少（从 Original 到 No URI）而下降，但 Protoknowledge 的激活与任务成功高度相关，尤其在无上下文支持时，拓扑型 Protoknowledge 的作用尤为关键。
*   **实验设置合理性:** 实验设计覆盖了多种模型、任务（KATs 和 Text-to-SPARQL）和数据集（DBpedia, Wikidata），并通过流行度分析揭示语义偏差的影响；但局限在于仅测试了有限的知识图谱和模型，未涉及其他结构化数据或更广泛的基准，可能影响结论的普适性。

## Further Thoughts

论文提出的 Protoknowledge 概念及其与语义偏差的关系启发我们思考：是否可以通过调整预训练数据的分布（例如增加低频知识图谱项的曝光）来减轻语义偏差对泛化能力的限制？此外，Protoknowledge 的分层定义是否可以扩展到其他结构化数据（如表格、树形结构）上，以研究 LLMs 的知识内化机制？同时，论文的分析框架作为检测语义级数据污染的工具，是否可以应用于其他任务（如 Text-to-SQL）以揭示更多预训练数据的影响？