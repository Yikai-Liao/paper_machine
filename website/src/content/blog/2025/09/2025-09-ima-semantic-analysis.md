---
title: "Combining Knowledge Graphs and NLP to Analyze Instant Messaging Data in Criminal Investigations"
pubDatetime: 2025-09-30T16:32:26+00:00
slug: "2025-09-ima-semantic-analysis"
type: "arxiv"
id: "2509.26487"
score: 0.5835375500327253
author: "grok-3-latest"
authors: ["Riccardo Pozzi", "Valentina Barbera", "Renzo Alva Principe", "Davide Giardini", "Riccardo Rubini", "Matteo Palmonari"]
tags: ["Knowledge Graph", "NLP", "Entity Extraction", "Semantic Search", "Criminal Investigation"]
institution: ["University of Milano-Bicocca"]
description: "本文提出了一种结合知识图谱和 NLP 技术的语义增强方法，用于刑事调查中的即时通讯数据分析，通过图谱建模、多媒体转录和实体提取显著提升数据探索效率，并在真实案例中验证了应用价值。"
---

> **Summary:** 本文提出了一种结合知识图谱和 NLP 技术的语义增强方法，用于刑事调查中的即时通讯数据分析，通过图谱建模、多媒体转录和实体提取显著提升数据探索效率，并在真实案例中验证了应用价值。 

> **Keywords:** Knowledge Graph, NLP, Entity Extraction, Semantic Search, Criminal Investigation

**Authors:** Riccardo Pozzi, Valentina Barbera, Renzo Alva Principe, Davide Giardini, Riccardo Rubini, Matteo Palmonari

**Institution(s):** University of Milano-Bicocca


## Problem Background

在刑事调查中，调查人员需要分析从嫌疑人手机中提取的大量即时通讯应用（IMA，如 WhatsApp）数据，包括文本、语音等多模态内容，这一过程极其耗时且资源密集，尤其是在处理语音消息时；现有工具仅支持简单的语法搜索，缺乏语义整合和多媒体分析能力，导致证据挖掘效率低下。

## Method

* **核心思想**：通过结合知识图谱（Knowledge Graph, KG）和自然语言处理（NLP）技术，对 IMA 数据进行语义增强和整合，帮助调查人员高效搜索和获取洞察，同时确保信息可验证性。
* **数据建模与知识图谱构建**：将从手机提取的聊天数据（包括元数据如参与者、时间戳，以及消息内容）组织成一个属性图谱，使用 Neo4j 存储，支持关系查询和可视化探索。
* **多媒体数据增强**：针对语音消息，采用 OpenAI 的 Whisper 模型进行语音转文本（Speech-to-Text），生成转录内容以便进一步搜索和分析；未来计划扩展到图像和视频处理。
* **实体提取与语义标注**：通过一个端到端的命名实体识别与链接（NEEL）管道，提取消息中的实体（如人名、组织、地点），并将其链接到知识图谱或外部知识库（如意大利语 Wikipedia）；对未链接实体（NIL）进行聚类以识别潜在关联；具体包括基于 SpaCy-transformers 的 NER、BLINK 架构的 NEL、逻辑回归分类器的 NIL 预测，以及基于 xgboost 和 Louvain 社区检测的实体聚类。
* **数据探索与用户界面**：提供两种界面支持数据探索，一是 Neo4j 自带的图谱查询与可视化界面，用于关系分析；二是基于语义搜索的 DAVE 应用，支持分面搜索（Faceted Search）和文档探索，允许用户基于实体和元数据过滤结果，并支持人工校正标注以确保准确性。

## Experiment

* **有效性**：实验基于两起真实刑事调查数据（涉及欺诈和腐败，数据量大，如第二起案件包含 1442 个聊天、364690 条消息），结果表明知识图谱和语义搜索显著提升了数据探索效率；语音转录（Whisper 模型）质量较高，调查人员对其错误容忍度较高，因可核对原始数据。
* **合理性**：实验设置覆盖真实调查场景，数据量和类型具有代表性，验证了方法在实际应用中的可行性；两阶段开发（第一阶段聚焦图谱构建，第二阶段加入 NEEL 和多媒体增强）体现了逐步迭代的合理性。
* **局限性**：NEEL 管道的命名实体识别（NER）性能不佳（F1 分数仅为 28.8），因 IMA 数据特殊性（如人名缩写、不规则大小写）需领域内微调；Neo4j 查询对非技术用户有门槛。
* **反馈**：调查人员反馈积极，认为图谱可视化和语义搜索直观有用，语音转录技术被认为是‘变革性’的，但提出改进需求，如增加音频片段播放功能和更友好的查询界面。

## Further Thoughts

本文启发我们思考如何在其他领域（如医疗、新闻分析）中结合多模态数据处理和语义整合技术，以解锁更多数据价值；同时，DAVE 应用的人机协同设计（支持人工校正标注）提示我们在敏感领域引入 AI 时，应始终保持人类控制和可解释性；此外，NEEL 管道在特定领域性能不足的问题，启发我们探索高效构建领域特定数据集和模型的方法。