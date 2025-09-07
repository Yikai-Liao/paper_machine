---
title: "NER Retriever: Zero-Shot Named Entity Retrieval with Type-Aware Embeddings"
pubDatetime: 2025-09-04T08:42:23+00:00
slug: "2025-09-ner-retriever-embeddings"
type: "arxiv"
id: "2509.04011"
score: 0.6714335364039901
author: "grok-3-latest"
authors: ["Or Shachar", "Uri Katz", "Yoav Goldberg", "Oren Glickman"]
tags: ["LLM", "Entity Retrieval", "Contrastive Learning", "Embedding", "Zero-Shot"]
institution: ["Bar-Ilan University"]
description: "本文提出 NER Retriever，一个基于 LLM 中层表示和对比学习的零样本命名实体检索框架，通过类型感知嵌入实现对开放式实体类型的高效检索，显著优于传统基线。"
---

> **Summary:** 本文提出 NER Retriever，一个基于 LLM 中层表示和对比学习的零样本命名实体检索框架，通过类型感知嵌入实现对开放式实体类型的高效检索，显著优于传统基线。 

> **Keywords:** LLM, Entity Retrieval, Contrastive Learning, Embedding, Zero-Shot

**Authors:** Or Shachar, Uri Katz, Yoav Goldberg, Oren Glickman

**Institution(s):** Bar-Ilan University


## Problem Background

传统命名实体识别（NER）依赖预定义类型和大量标注数据，无法适应用户临时定义的开放式实体类型。
论文提出即席命名实体检索任务，目标是根据查询时用户提供的任意类型描述，从文档中检索包含该类型实体的文本，解决零样本设置下对新型、细粒度实体类型的泛化问题，满足信息检索、问答和知识库构建的实际需求。

## Method

*   **核心思想**：构建一个零样本检索框架 NER Retriever，将实体提及和用户提供的类型描述嵌入到共享语义空间，通过相似性搜索实现检索。
*   **嵌入选择**：利用大型语言模型（LLM，如 LLaMA 3.1 8B）的内部表示，发现中层 Transformer 块的‘值向量’（value vectors，例如第 17 层）在区分细粒度实体类型方面优于顶层输出，通过实验验证不同层和子组件的类型敏感性，选择最优表示作为基础嵌入。
*   **类型感知优化**：设计一个轻量级多层感知机（MLP），采用对比学习（contrastive learning）优化嵌入空间，使用三元组损失（triplet loss）训练，使相同类型实体嵌入更接近，不同类型嵌入分离，训练数据包括类型描述（锚点）、同类型实体（正样本）和不同类型实体（负样本）。
*   **系统实现**：分为三个阶段：（1）实体检测，使用预训练模型识别文本中的实体跨度；（2）索引阶段，预计算并存储所有实体提及的类型感知嵌入；（3）检索阶段，将用户输入的类型描述嵌入到同一空间，通过最近邻搜索返回相关文档。
*   **关键创新**：不依赖任务特定微调，利用 LLM 内部中层表示的类型信息，通过轻量级对比学习实现零样本泛化，同时降低存储开销（嵌入维度仅为 500）。

## Experiment

*   **有效性**：在 Few-NERD 和 MultiCoNER 2 数据集上，NER Retriever 的 R-Precision 分别为 0.34 和 0.32，显著优于基线模型（如 BM25、E5-Mistral、NV-Embed v2），提升幅度达 3-4 倍，证明类型感知嵌入在细粒度和低上下文场景下的优势。
*   **局限性表现**：在 NERetrieve Test 数据集上，R-Precision 为 0.28，与 NV-Embed v2（0.29）和 BM25（0.27）相当，未显著优于基线，原因是数据集描述性强，实体类型常直接出现在文本中，利于词法方法。
*   **实验设置合理性**：实验覆盖多种场景（手动标注、银标注、长尾类型），采用零样本设置，符合实际应用需求；消融实验验证了中层表示、实体 token 选择和 MLP 投影的重要性。
*   **瓶颈分析**：实体检测准确性对性能影响较大，使用 oracle 实体边界时性能提升约 11%，表明改进实体检测可进一步提升系统效果。

## Further Thoughts

论文揭示了 LLM 中层表示在捕捉细粒度类型信息方面的潜力，启发我们可以在其他 NLP 任务中探索内部层表示的特化性，设计任务特定的层选择策略；此外，通过轻量级对比学习优化嵌入空间的思路，提示在资源受限场景下小规模监督训练即可提升零样本性能；实体级嵌入相比句子级嵌入的精准性和存储效率优势，也可扩展到其他检索任务，如知识图谱构建或问答系统。