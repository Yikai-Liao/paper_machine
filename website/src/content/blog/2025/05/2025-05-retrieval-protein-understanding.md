---
title: "Rethinking Text-based Protein Understanding: Retrieval or LLM?"
pubDatetime: 2025-05-26T06:25:43+00:00
slug: "2025-05-retrieval-protein-understanding"
type: "arxiv"
id: "2505.20354"
score: 0.7076311294123943
author: "grok-3-latest"
authors: ["Juntong Wu", "Zijing Liu", "He Cao", "Hao Li", "Bin Feng", "Zishan Shu", "Ke Yu", "Li Yuan", "Yu Li"]
tags: ["LLM", "Retrieval-Augmented Generation", "Protein Understanding", "Evaluation Metrics", "Data Leakage"]
institution: ["Peking University", "International Digital Economy Academy (IDEA)"]
description: "本文提出检索增强蛋白质建模（RAPM）框架，通过结合检索方法和 LLM 推理能力，并设计新基准与生物学实体聚焦的评估指标，显著提升了蛋白质理解任务在分布外数据上的表现。"
---

> **Summary:** 本文提出检索增强蛋白质建模（RAPM）框架，通过结合检索方法和 LLM 推理能力，并设计新基准与生物学实体聚焦的评估指标，显著提升了蛋白质理解任务在分布外数据上的表现。 

> **Keywords:** LLM, Retrieval-Augmented Generation, Protein Understanding, Evaluation Metrics, Data Leakage

**Authors:** Juntong Wu, Zijing Liu, He Cao, Hao Li, Bin Feng, Zishan Shu, Ke Yu, Li Yuan, Yu Li

**Institution(s):** Peking University, International Digital Economy Academy (IDEA)


## Problem Background

近年来，大型语言模型（LLMs）在蛋白质-文本理解任务中被广泛应用，但其是否真正理解蛋白质知识仍存疑。
现有基准存在严重的数据泄露问题（训练集与测试集高度相似），导致模型可能仅通过模式匹配而非生物学理解获得高分；同时，传统自然语言处理（NLP）评估指标无法准确反映模型在生物学领域的表现。
论文旨在解决现有基准和评估方法的局限性，并探讨 LLMs 是否在蛋白质理解任务中优于传统检索方法。

## Method

*   **核心思想:** 提出检索增强蛋白质建模（Retrieval-Augmented Protein Modeling, RAPM）框架，基于检索增强生成（RAG）范式，通过结合检索方法的精确性和 LLM 的推理能力，提升蛋白质理解任务的表现。
*   **具体实现:**
    *   **蛋白质知识数据库构建:** 从多个生物学数据库（如 InterPro, EC-GO, Mol-Instructions）收集蛋白质-注释数据，构建双重索引数据库，包括基于 k-mer 倒排索引的序列索引和基于 ESM-2 嵌入向量与 HNSW 算法的特征索引，以提高检索效率和准确性。
    *   **检索增强生成过程:** 在推理时，针对输入蛋白质查询，从数据库中检索 Top-K 相关样本，并将其格式化为 [Confidence, Annotation] 形式（Confidence 基于相似性分数分为 High, Medium, Low），结合查询和少样本示例（few-shot examples）构建输入提示（prompt），供 LLM 生成上下文感知的答案。
    *   **新基准与评估指标:** 提出 Mol-Instructions-OOD 数据集，通过序列聚类和泄露消除，确保训练集与测试集分布差异，解决数据泄露问题；设计 Entity-BLEU 指标，专注于生物学实体的匹配，而非普通文本重叠，以更准确评估生物学理解能力。
*   **关键优势:** 无需对 LLM 进行昂贵的微调，仅在推理时通过检索增强提示即可提升性能，同时通过新基准和指标提高评估的科学性和公平性。

## Experiment

*   **有效性:** 在 Mol-Instructions-OOD 数据集上，RAPM 在 Entity-BLEU 指标上显著优于微调 LLM、纯检索方法和任务提示 LLM，尤其在蛋白质功能、领域/基序和催化活性任务中表现突出，例如 RAPM 结合 GPT-4.1 在多个任务中取得最高分（Entity-BLEU 最高达 46.6）。
*   **对比分析:** 微调 LLM 在 ROUGE-L 上得分较高，但在 Entity-BLEU 上表现较差，表明其更多依赖文本模式而非生物学理解；纯检索方法在 OOD 数据集上性能下降明显，适应性不足；RAPM 则在计算资源需求较低（无需微调）的情况下展现出更强的泛化能力。
*   **实验设置合理性:** 实验涵盖多种模型架构（BioT5+, Llama, BioGPT 等）和任务类型（蛋白质功能、描述、领域、催化活性），并通过消融研究验证了检索样本数量（K 值）、数据库索引方法和提示构建对性能的影响，例如增加 K 值初期提升性能，但过大时引入低置信样本导致下降。
*   **结论:** RAPM 的提升显著，尤其在分布外数据上的表现优于其他方法，验证了检索与推理结合的有效性。

## Further Thoughts

RAPM 框架通过检索提供事实依据并结合 LLM 推理的混合方法，启发我们在其他领域（如医学诊断、法律文本分析）中应用类似策略以解决 LLM '幻觉' 问题；Entity-BLEU 的设计提示我们需针对领域核心内容定制评估指标，这对跨领域 AI 研究有借鉴意义；此外，数据泄露问题的揭示提醒我们在构建数据集时需关注分布差异，尤其是在高相似性领域数据中。