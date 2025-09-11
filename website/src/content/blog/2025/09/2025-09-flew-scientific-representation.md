---
title: "FLeW: Facet-Level and Adaptive Weighted Representation Learning of Scientific Documents"
pubDatetime: 2025-09-09T09:08:44+00:00
slug: "2025-09-flew-scientific-representation"
type: "arxiv"
id: "2509.07531"
score: 0.7708565781731148
author: "grok-3-latest"
authors: ["Zheng Dou", "Deqing Wang", "Fuzhen Zhuang", "Jian Ren", "Yanlin Hu"]
tags: ["Scientific Document", "Representation Learning", "Contrastive Learning", "Facet-Level Embedding", "Adaptive Weighting"]
institution: ["Beihang University (School of Computer Science, SKLSDE)", "Beihang University (Institute of Artificial Intelligence)", "Zhongguancun Laboratory", "National Computer Network Emergency Response Technical Team/Coordination Center of China"]
description: "本文提出 FLeW 框架，通过结构化采样、文本分割和自适应加权统一引用结构对比训练、细粒度表示和任务自适应学习，显著提升科学文档表示的性能和跨任务、跨领域适用性。"
---

> **Summary:** 本文提出 FLeW 框架，通过结构化采样、文本分割和自适应加权统一引用结构对比训练、细粒度表示和任务自适应学习，显著提升科学文档表示的性能和跨任务、跨领域适用性。 

> **Keywords:** Scientific Document, Representation Learning, Contrastive Learning, Facet-Level Embedding, Adaptive Weighting

**Authors:** Zheng Dou, Deqing Wang, Fuzhen Zhuang, Jian Ren, Yanlin Hu

**Institution(s):** Beihang University (School of Computer Science, SKLSDE), Beihang University (Institute of Artificial Intelligence), Zhongguancun Laboratory, National Computer Network Emergency Response Technical Team/Coordination Center of China


## Problem Background

随着科学出版物的快速增长，高质量的文档表示对于支持下游任务（如分类、检索和搜索）变得至关重要。
科学文档具有独特的引用结构和浓缩知识，但现有方法面临三大挑战：
1. 引用结构信号的对比训练未充分利用引用信息，且单向量表示无法捕捉细粒度信息；
2. 细粒度表示学习整合成本高且缺乏跨领域泛化能力；
3. 任务感知学习依赖手动任务分类，忽略细微差异并需要额外训练数据。
论文旨在统一这三种方法，克服局限性，生成更具适用性和鲁棒性的科学文档表示。

## Method

*   **核心思想:** 提出 FLeW 框架，统一引用结构对比训练、细粒度多向量表示和任务自适应学习，生成高质量科学文档表示。
*   **结构化采样 (Structural Sampling):** 
    *   利用引用意图（背景、方法、结果）和引用频率构建三个加权子图，从中采样三元组（query, positive, negative）用于对比训练。
    *   这种方法增强了引用结构信号的利用，确保训练数据反映科学文档的结构化特性。
*   **文本分割 (Textual Splitting):** 
    *   将摘要文本按引用意图划分为三个方面（背景、方法、结果），使用经过指令微调的大型语言模型（如 Llama3.1-8B-Instruct）完成分割。
    *   分割后的文本用于训练，确保细粒度表示的领域通用性，避免了领域特定标注数据的依赖。
*   **预训练 (Pre-Training):** 
    *   基于三个方面的三元组，分别训练三个编码器（基于 SciBERT），每个编码器专注于一个方面的表示学习。
    *   使用三元组边际损失（Triplet Margin Loss）进行对比训练，使查询与正样本更接近，与负样本更远。
*   **推理与加权整合 (Inferring):** 
    *   推理时使用完整摘要和标题作为输入，通过三个编码器生成三个方面的嵌入向量。
    *   通过网格搜索确定每个方面的权重，计算加权和作为最终文档表示，实现任务自适应而无需任务特定微调。

## Experiment

*   **有效性:** 在 SciRepEval 基准（19 个任务）上，FLeW 在 13 个任务中表现最佳，平均性能（60.81）比第二名（SPECTER-2 的 60.21）高 0.6 个百分点；在 MDCR 数据集（19 个领域）上，FLeW 在几乎所有领域表现最佳，尤其在科学领域（如生物学、化学）提升显著。
*   **局限性:** 在人文学科领域（如哲学）略逊于 SPECTER-2，可能因其依赖科学写作结构（背景、方法、结果）而对非结构化领域适用性不足。
*   **消融研究:** 文本分割消除了位置偏差（如背景部分因位置靠前被过度关注），结构化采样和加权求和策略显著提升性能，验证了各组件的有效性。
*   **实验设置合理性:** 实验覆盖多种任务格式（接近性、回归、查询、分类）和多个领域，数据规模较大（216k 论文生成 2.16M 三元组），对比模型包括 SciBERT、SPECTER、SciNCL 等主流方法，设计全面；但未讨论计算成本（如三个编码器的训练和推理开销）及小规模数据集表现。

## Further Thoughts

1. **引用意图作为结构化划分依据:** 将引用意图与科学写作结构对齐作为细粒度表示的划分依据，这种思路可推广到其他结构化文本（如法律文档、专利），利用其固有结构进行表示学习。
2. **自适应加权替代任务微调:** 通过简单权重搜索实现任务自适应，避免复杂微调，这种方法可应用于其他多向量表示场景，降低定制化成本。
3. **LLM 在数据预处理中的潜力:** 使用指令微调的 LLM 进行文本分割，解决标注数据不足问题，这种方法可进一步探索，例如在其他 NLP 任务中利用 LLM 进行特征提取或数据增强。