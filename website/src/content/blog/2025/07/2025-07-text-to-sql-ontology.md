---
title: "Text-to-SQL Task-oriented Dialogue Ontology Construction"
pubDatetime: 2025-07-31T09:08:59+00:00
slug: "2025-07-text-to-sql-ontology"
type: "arxiv"
id: "2507.23358"
score: 0.5869138938618085
author: "grok-3-latest"
authors: ["Renato Vukovic", "Carel van Niekerk", "Michael Heck", "Benjamin Ruppik", "Hsien-chin Lin", "Shutong Feng", "Nurul Lubis", "Milica Gaši´c"]
tags: ["LLM", "Ontology Construction", "Task-Oriented Dialogue", "Text-to-SQL", "Dialogue State Tracking"]
institution: ["Heinrich Heine University Düsseldorf"]
description: "本文提出 TeQoDO 方法，利用大型语言模型的 Text-to-SQL 能力结合对话理论，无监督构建任务导向对话本体，在多个数据集上显著优于现有方法并展现出良好的泛化性。"
---

> **Summary:** 本文提出 TeQoDO 方法，利用大型语言模型的 Text-to-SQL 能力结合对话理论，无监督构建任务导向对话本体，在多个数据集上显著优于现有方法并展现出良好的泛化性。 

> **Keywords:** LLM, Ontology Construction, Task-Oriented Dialogue, Text-to-SQL, Dialogue State Tracking

**Authors:** Renato Vukovic, Carel van Niekerk, Michael Heck, Benjamin Ruppik, Hsien-chin Lin, Shutong Feng, Nurul Lubis, Milica Gaši´c

**Institution(s):** Heinrich Heine University Düsseldorf


## Problem Background

大型语言模型（LLMs）在任务导向对话（Task-Oriented Dialogue, TOD）系统中表现出色，但其知识存储在参数中，缺乏可解释性和可信度；传统 TOD 本体构建依赖人工标注或监督训练，成本高且效率低，现有自动化方法常因分步处理导致信息丢失和错误传播。

## Method

* **核心思想:** 利用大型语言模型（LLMs）的 SQL 编程能力，结合对话理论，从头开始无监督地构建任务导向对话（TOD）本体，将对话数据转化为结构化的数据库形式。
* **具体步骤:** 
  * **查询现有数据库信息:** 首先通过 PRAGMA 和 SELECT 查询获取当前数据库的表和列信息，确保后续更新操作与现有结构一致，初始时数据库为空。
  * **对话状态追踪（Dialogue State Tracking, DST）:** 基于对话理论，分析对话内容，区分已有数据库信息和新信息，将对话状态以表-列-值的形式结构化表示。
  * **数据库更新:** 根据 DST 结果和对话中缺失的信息，生成 SQL 更新查询，包括 CREATE TABLE（创建新实体类型表）、ALTER TABLE（添加新属性列）、INSERT INTO（插入新实体或值）和 UPDATE（更新现有实体信息），优先复用现有表结构。
  * **对话成功（Dialogue Success）引导:** 在更新提示中引入用户目标实现的概念，确保数据库更新支持对话中用户意图的达成。
* **技术细节:** 使用 SQLite 作为数据库管理系统，SQL 表的结构（表-列-值）与 TOD 本体（领域-槽-值）天然对齐；通过语义相似性匹配和列值示例增强一致性，减少重复条目。
* **关键优势:** 无需监督训练，避免传统两步法（术语提取和关系提取分开）的信息丢失，增量构建本体适应动态对话数据。

## Experiment

* **有效性:** 在 MultiWOZ 和 SGD 两个 TOD 数据集上，TeQoDO 显著优于监督基线方法（如 DORE 和 GenDSI），在 MultiWOZ 上平均 Continuous F1 分数达 65.25，在 SGD 上达 61.64，尤其在高层次概念（如领域和槽）预测上表现突出；下游对话状态追踪任务中，TeQoDO 构建的本体性能接近甚至部分超过真实本体。
* **泛化性:** 在通用本体数据集（Wikipedia 和 ArXiv）上，TeQoDO 虽不如监督方法 OLLM，但在无监督情况下仍具竞争力，特别是在 ArXiv 数据集上图结构 F1 分数达 89.89，显示出较强的结构诱导能力。
* **消融研究:** 对话理论（DST 和 Success）的引入对性能提升显著，Success 概念使 SQL 错误率从 37.31% 降至 4.71%（MultiWOZ），生成的表数量更接近真实领域数量；语义相似性匹配和列值示例进一步提升一致性和准确性。
* **实验设置合理性:** 实验覆盖 TOD 和通用本体数据集，采用多种评价指标（Literal, Fuzzy, Continuous F1），全面评估本体质量；但在 Wikipedia 数据集上因规模和层次结构差异表现受限，提示未来优化方向。
* **计算开销:** 主要开销在于每次对话处理时的多步 SQL 查询和更新生成，以及语义相似性计算，但整体仍适用于中小规模数据库。

## Further Thoughts

TeQoDO 利用 LLM 的 SQL 生成能力构建结构化知识的思路启发我们，可以探索其他结构化表示（如图数据库）来适应更复杂的本体层次结构；对话理论（DST 和 Success）在无监督学习中的作用提示我们可以在其他 NLP 任务中引入领域特定理论来约束和优化 LLM 输出；此外，是否可以通过多模态数据（如语音对话）进一步丰富本体构建的输入来源，值得未来研究。