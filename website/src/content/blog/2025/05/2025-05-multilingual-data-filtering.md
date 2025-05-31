---
title: "Judging Quality Across Languages: A Multilingual Approach to Pretraining Data Filtering with Language Models"
pubDatetime: 2025-05-28T11:06:54+00:00
slug: "2025-05-multilingual-data-filtering"
type: "arxiv"
id: "2505.22232"
score: 0.6290779345956098
author: "grok-3-latest"
authors: ["Mehdi Ali", "Manuel Brack", "Max Lübbering", "Elias Wendt", "Abbas Goher Khan", "Richard Rutmann", "Alex Jude", "Maurice Kraus", "Alexander Arno Weber", "Felix Stollenwerk", "David Kaczér", "Florian Mai", "Lucie Flek", "Rafet Sifa", "Nicolas Flores-Herr", "Joachim Köhler", "Patrick Schramowski", "Michael Fromm", "Kristian Kersting"]
tags: ["LLM", "Data Filtering", "Multilingual Learning", "Embedding Models", "Pre-Training"]
institution: ["Lamarr Institute", "Fraunhofer IAIS", "DFKI SAINT", "Hessian AI", "Computer Science Department, TU Darmstadt", "AI Sweden"]
description: "本文提出JQL框架，通过人类标注、LLM评判和轻量级模型蒸馏，实现多语言预训练数据的高效高质量过滤，显著优于启发式方法并泛化至未见语言。"
---

> **Summary:** 本文提出JQL框架，通过人类标注、LLM评判和轻量级模型蒸馏，实现多语言预训练数据的高效高质量过滤，显著优于启发式方法并泛化至未见语言。 

> **Keywords:** LLM, Data Filtering, Multilingual Learning, Embedding Models, Pre-Training

**Authors:** Mehdi Ali, Manuel Brack, Max Lübbering, Elias Wendt, Abbas Goher Khan, Richard Rutmann, Alex Jude, Maurice Kraus, Alexander Arno Weber, Felix Stollenwerk, David Kaczér, Florian Mai, Lucie Flek, Rafet Sifa, Nicolas Flores-Herr, Joachim Köhler, Patrick Schramowski, Michael Fromm, Kristian Kersting

**Institution(s):** Lamarr Institute, Fraunhofer IAIS, DFKI SAINT, Hessian AI, Computer Science Department, TU Darmstadt, AI Sweden


## Problem Background

大型语言模型（LLM）的预训练数据质量对模型性能至关重要，但现有高质量多语言数据集稀缺，且多依赖启发式过滤方法，缺乏跨语言适应性和可扩展性，尤其对低资源语言效果不明；此外，许多前沿AI实验室的数据处理策略不公开，阻碍学术研究。本文旨在解决如何在多语言环境下高效筛选高质量预训练数据，并确保跨语言（包括未见语言）的泛化能力，同时降低计算成本。

## Method

* **核心思想**：提出 JQL（Judging Quality across Languages），一个系统化的多语言数据过滤框架，通过结合人类标注、LLM评判和轻量级模型蒸馏，实现高效、高质量的多语言预训练数据筛选。
* **具体流程**：
  * **阶段1 - 人类标注**：通过15名人类标注员对英语文档的教育价值进行评分（0-5分），并翻译至35种语言，构建多语言地面真实数据集（Ground Truth），共511个文档，每文档3次独立标注，通过多数投票或平均值聚合。
  * **阶段2 - LLM评判**：测试多个多语言LLM（如Gemma-3-27B-it, Mistral-3.1-24B-it, LLaMA-3.3-70B-it）对文档质量的评分能力，基于Spearman相关性选择表现最佳的模型，生成大规模合成标注数据（覆盖35种语言，约1400万文档）。
  * **阶段3 - 轻量级标注器蒸馏**：以预训练多语言嵌入模型（如Snowflake Arctic Embed v2.0）为骨干，附加轻量级回归头（MLP，仅占参数1%），利用合成数据训练，实现低成本、高效的多语言质量评分，支持跨语言泛化。
  * **阶段4 - 数据过滤**：使用轻量级标注器对大规模数据集（如Fineweb2）进行评分，基于百分位阈值（而非绝对分数）筛选高质量数据，通过三个标注器（基于不同LLM）的集成决策增强鲁棒性。
* **关键创新**：利用嵌入模型的跨语言对齐能力，确保未见语言的零样本性能；共享嵌入骨干支持多任务扩展；百分位阈值策略避免评分分布偏差。

## Experiment

* **有效性**：JQL轻量级标注器在35种语言上的Spearman相关性与人类标注高度一致，平均略优于原始LLM评判者，表明其质量评估能力强大。
* **跨语言泛化**：对未见语言（如阿拉伯语、泰语、汉语）表现出零样本性能，与欧洲语言结果相当，验证了嵌入模型跨语言对齐的有效性。
* **性能提升**：与启发式方法（如Fineweb2）相比，JQL过滤的数据在下游任务（如MMLU, HellaSwag, ARC）中显著提升模型性能，例如在0.7百分位阈值下，平均性能提升6.7%-7.2%，且在部分语言（如西班牙语）保留更多数据（比Fineweb2多9%）。
* **计算效率**：轻量级标注器在单张A100 GPU上每分钟处理约11,000个文档，计算成本低，适合大规模应用。
* **实验设置**：覆盖35种欧洲语言及3种未见语言，考虑语言家族拓扑差异，采用百分位阈值避免评分偏差，实验设计全面；但由于资源限制，模型规模固定在20亿参数，可能未完全反映大规模预训练效果。

## Further Thoughts

JQL的共享嵌入骨干+轻量级任务头设计启发我们，是否可以将类似框架应用于多模态数据过滤（如图像-文本对质量评估），通过复用预训练嵌入降低多任务成本？此外，百分位阈值策略是否可推广至其他主观性评估任务，如情感分析或用户反馈排序？最后，跨语言泛化能力提示嵌入模型可能是多语言任务的基础设施，是否可以通过引入更多低资源语言的少量标注数据，进一步优化嵌入对齐，增强对边缘语言的支持？