---
title: "Judging Quality Across Languages: A Multilingual Approach to Pretraining Data Filtering with Language Models"
pubDatetime: 2025-05-28T11:06:54+00:00
slug: "2025-05-multilingual-data-filtering"
type: "arxiv"
id: "2505.22232"
score: 0.6290779345956098
author: "grok-3-latest"
authors: ["Mehdi Ali", "Manuel Brack", "Max Lübbering", "Elias Wendt", "Abbas Goher Khan", "Richard Rutmann", "Alex Jude", "Maurice Kraus", "Alexander Arno Weber", "Felix Stollenwerk", "David Kaczér", "Florian Mai", "Lucie Flek", "Rafet Sifa", "Nicolas Flores-Herr", "Joachim Köhler", "Patrick Schramowski", "Michael Fromm", "Kristian Kersting"]
tags: ["LLM", "Data Filtering", "Multilingual Learning", "Embedding Model", "Pre-Training"]
institution: ["Lamarr Institute", "Fraunhofer IAIS", "DFKI SAINT", "Hessian AI", "Computer Science Department, TU Darmstadt", "AI Sweden"]
description: "本文提出 JQL 方法，通过人类标注、LLM 评判和蒸馏技术构建轻量级多语言数据标注器，显著提升预训练数据质量并实现跨语言泛化，优于传统启发式过滤方法。"
---

> **Summary:** 本文提出 JQL 方法，通过人类标注、LLM 评判和蒸馏技术构建轻量级多语言数据标注器，显著提升预训练数据质量并实现跨语言泛化，优于传统启发式过滤方法。 

> **Keywords:** LLM, Data Filtering, Multilingual Learning, Embedding Model, Pre-Training

**Authors:** Mehdi Ali, Manuel Brack, Max Lübbering, Elias Wendt, Abbas Goher Khan, Richard Rutmann, Alex Jude, Maurice Kraus, Alexander Arno Weber, Felix Stollenwerk, David Kaczér, Florian Mai, Lucie Flek, Rafet Sifa, Nicolas Flores-Herr, Joachim Köhler, Patrick Schramowski, Michael Fromm, Kristian Kersting

**Institution(s):** Lamarr Institute, Fraunhofer IAIS, DFKI SAINT, Hessian AI, Computer Science Department, TU Darmstadt, AI Sweden


## Problem Background

大型语言模型（LLMs）的预训练数据质量对模型性能至关重要，但现有启发式数据过滤方法缺乏语义质量评估能力，尤其在多语言环境下，难以适应低资源语言和实现跨语言迁移，导致数据质量和模型训练效果受限。

## Method

*   **核心思想:** 提出 JQL（Judging Quality across Languages），一种系统性多语言数据过滤方法，通过人类标注、LLM 评判和蒸馏技术，构建高效的轻量级标注器，以低计算成本筛选高质量预训练数据。
*   **具体实现:** 方法分为四个阶段：
    *   **人类标注阶段:** 收集少量英语文档（511 个）的人类标注作为地面真实数据，并翻译至 35 种语言，形成多语言基准数据集，用于后续评估。
    *   **LLM 评判阶段:** 利用多个强大的多语言 LLM（如 Gemma-3-27B、Mistral-3.1-24B、LLaMA-3.3-70B）对大规模多语言数据进行质量评分，评估其教育价值（0-5 分），并选择表现最佳的模型生成合成标注数据。
    *   **蒸馏轻量级标注器:** 以预训练多语言嵌入模型（如 Snowflake Arctic Embed v2.0）为骨干，冻结其权重，附加轻量级回归头（多层感知机 MLP，参数占比不到 1%），基于 LLM 合成数据训练，实现高效质量评分。
    *   **数据过滤与应用:** 使用轻量级标注器的评分，基于百分位阈值（如 0.6 或 0.7）过滤高质量数据，并通过集成多个标注器（Gemma、Mistral、LLaMA 基模型）减少个体偏差，提供质量与数量的可控权衡。
*   **关键特点:** 方法语言无关，可扩展至任意过滤标准（如代码质量、数学准确性），并在未见语言上展现零样本泛化能力，同时计算成本低，适合大规模数据处理。

## Experiment

*   **有效性:** JQL 在 35 种欧洲语言上显著优于启发式方法（如 Fineweb2），在下游任务（如 MMLU、HellaSwag、ARC）中，使用 JQL 过滤的数据（0.7 百分位阈值）使模型性能提升 6.7%-7.2%，同时保留更多数据（相比 Fineweb2）。
*   **跨语言能力:** 轻量级标注器在未见语言（如阿拉伯语、泰语、汉语）上表现出较强的零样本性能，Spearman 相关性与欧洲语言相当，验证了嵌入模型的跨语言对齐能力，但对某些孤立语言（如巴斯克语、爱尔兰语）性能略有下降。
*   **计算效率:** 轻量级标注器在单张 A100 GPU 上每分钟处理约 11,000 个文档（平均 690 token/文档），远低于直接使用 LLM 标注的成本，展现了蒸馏方法的实用性。
*   **实验设置合理性:** 实验覆盖 35 种语言及多种下游任务，语言家族分布均衡，设置较为全面；但由于计算限制，模型规模固定为 20 亿参数，未能验证更大规模模型上的性能趋势。

## Further Thoughts

JQL 的共享嵌入骨干与多任务回归头的设计启发了我，是否可以将这一框架扩展到多模态数据过滤，例如结合文本和图像嵌入模型，评估多模态内容质量？此外，嵌入模型的跨语言对齐能力提示我们，未来可以通过对比学习或多语言预训练进一步增强嵌入一致性，提升低资源语言的表现。