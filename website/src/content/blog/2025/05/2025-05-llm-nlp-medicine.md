---
title: "The Evolving Landscape of Generative Large Language Models and Traditional Natural Language Processing in Medicine"
pubDatetime: 2025-05-15T13:11:14+00:00
slug: "2025-05-llm-nlp-medicine"
type: "arxiv"
id: "2505.10261"
score: 0.501919774553364
author: "grok-3-latest"
authors: ["Rui Yang", "Huitao Li", "Matthew Yu Heng Wong", "Yuhe Ke", "Xin Li", "Kunyu Yu", "Jingchi Liao", "Jonathan Chong Kai Liew", "Sabarinath Vinod Nair", "Jasmine Chiat Ling Ong", "Irene Li", "Douglas Teodoro", "Chuan Hong", "Daniel Shu Wei Ting", "Nan Liu"]
tags: ["LLM", "Natural Language Processing", "Medical Applications", "Topic Modeling", "Text Analysis"]
institution: ["Duke-NUS Medical School, Singapore", "University of Cambridge, UK", "Singapore General Hospital, Singapore", "Nanyang Technological University, Singapore", "The University of Tokyo, Japan", "University of Geneva, Switzerland", "Duke School of Medicine, USA", "Duke Clinical Research Institute, USA", "Singapore Eye Research Institute, Singapore", "Stanford University, USA", "Singapore Health Services, Singapore", "National University of Singapore, Singapore"]
description: "本文通过大规模文献分析和主题建模，揭示了生成式大语言模型和传统自然语言处理方法在医学领域的任务分布差异，为未来医疗 NLP 技术发展提供了方向性指引。"
---

> **Summary:** 本文通过大规模文献分析和主题建模，揭示了生成式大语言模型和传统自然语言处理方法在医学领域的任务分布差异，为未来医疗 NLP 技术发展提供了方向性指引。 

> **Keywords:** LLM, Natural Language Processing, Medical Applications, Topic Modeling, Text Analysis

**Authors:** Rui Yang, Huitao Li, Matthew Yu Heng Wong, Yuhe Ke, Xin Li, Kunyu Yu, Jingchi Liao, Jonathan Chong Kai Liew, Sabarinath Vinod Nair, Jasmine Chiat Ling Ong, Irene Li, Douglas Teodoro, Chuan Hong, Daniel Shu Wei Ting, Nan Liu

**Institution(s):** Duke-NUS Medical School, Singapore, University of Cambridge, UK, Singapore General Hospital, Singapore, Nanyang Technological University, Singapore, The University of Tokyo, Japan, University of Geneva, Switzerland, Duke School of Medicine, USA, Duke Clinical Research Institute, USA, Singapore Eye Research Institute, Singapore, Stanford University, USA, Singapore Health Services, Singapore, National University of Singapore, Singapore


## Problem Background

医学领域每天产生大量非结构化文本数据（如电子健康记录、生物医学文献），高效提取和结构化这些信息对提升医疗质量、支持临床决策和推动医学研究至关重要。
传统自然语言处理（NLP）方法在信息提取和分析任务中占据主导地位，而生成式大语言模型（LLMs）在开放性任务中展现出潜力，但两者的研究重点和应用场景差异尚未被充分探索。
本研究旨在揭示生成式 LLMs 和传统 NLP 方法在医学领域的任务分布和应用场景差异，以指导未来医疗 NLP 技术的发展方向。

## Method

*   **研究设计**：通过系统性文献分析和主题建模，比较生成式 LLMs 和传统 NLP 方法在医学领域的应用分布。
*   **数据收集与处理**：从 PubMed, Embase, Scopus, 和 Web of Science 数据库检索相关研究，共分析 19,123 篇文献（其中 4,295 篇与生成式 LLMs 相关，14,828 篇与传统 NLP 相关），通过去重、筛选无摘要或非文章记录，并基于标题和摘要中的关键词进行分类。
*   **主题建模技术**：采用 BERTopic 工具进行主题建模，使用 MedCPT Article Encoder 模型生成文献嵌入，通过 UMAP 算法将嵌入降维至低维空间以保留语义信息，随后利用层次密度聚类（HDBSCAN）算法自动生成初始 40 个主题，最终由医学专家合并为 26 个主题。
*   **分析方法**：通过嵌入空间的可视化（UMAP 降维后的密度分布）和主题分布比例，分析生成式 LLMs 和传统 NLP 方法在不同医疗任务（如医疗教育、文本摘要、电子健康记录、命名实体识别）上的研究重点差异。
*   **补充细节**：分类和主题建模过程结合了自动化工具和人工专家知识，确保结果的准确性和医学相关性，同时提供了详细的关键词列表和搜索策略以支持可重复性。

## Experiment

*   **结果显著性**：生成式 LLMs 在开放性任务中占比更高，如在‘医疗教育’主题中占比 72.23%，在‘文本摘要’中占比 19.95%，在‘医学图像分析’中占比 9.80%，显示其在内容生成和跨模态分析中的潜力；传统 NLP 方法在结构化信息提取任务中占主导，如在‘电子健康记录’主题中占比 23.62%，在‘命名实体识别’中占比 13.70%，表明其在高精度任务中的优势。
*   **实验设置合理性**：研究覆盖了多个数据库，分析了 19,123 篇文献，数据量大且来源广泛；通过 BERTopic 和 UMAP 等先进工具结合医学专家的主题合并，确保了主题建模的科学性和结果的可信度。
*   **局限性**：实验未深入探讨 LLMs 在临床实践中的具体性能（如准确性、解释性），也未在同一任务上直接对比两种技术的效果，更多聚焦于研究分布而非实际应用效果。

## Further Thoughts

论文揭示了生成式 LLMs 和传统 NLP 方法在医学领域的互补优势，启发我思考是否可以在信息提取任务中结合传统 NLP 的高精度特性与 LLMs 的推理能力，例如先用传统 NLP 提取结构化信息，再由 LLMs 进行复杂推理或生成报告；此外，LLMs 在医疗教育中的高占比提示其在个性化教育内容生成中的潜力，如根据医学生进度动态调整教学内容；另外，LLMs 在多模态处理（如医学图像分析）中的应用趋势，表明未来医疗 AI 可能需要更多跨模态数据的整合，以提升诊断和报告的全面性。