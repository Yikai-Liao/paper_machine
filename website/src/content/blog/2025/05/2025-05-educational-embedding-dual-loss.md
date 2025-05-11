---
title: "An Open-Source Dual-Loss Embedding Model for Semantic Retrieval in Higher Education"
pubDatetime: 2025-05-08T03:14:14+00:00
slug: "2025-05-educational-embedding-dual-loss"
type: "arxiv"
id: "2505.04916"
score: 0.5069112553378103
author: "grok-3-latest"
authors: ["Ramteja Sajja", "Yusuf Sermet", "Ibrahim Demir"]
tags: ["Semantic Retrieval", "Embedding Model", "Fine-Tuning", "Educational Technology", "Natural Language Processing"]
institution: ["University of Iowa", "Tulane University"]
description: "本文提出两种针对教育领域的开源嵌入模型，通过双损失微调策略显著提升语义检索性能，接近专有模型水平，为学术问答和检索系统提供了透明、低成本的解决方案。"
---

> **Summary:** 本文提出两种针对教育领域的开源嵌入模型，通过双损失微调策略显著提升语义检索性能，接近专有模型水平，为学术问答和检索系统提供了透明、低成本的解决方案。 

> **Keywords:** Semantic Retrieval, Embedding Model, Fine-Tuning, Educational Technology, Natural Language Processing

**Authors:** Ramteja Sajja, Yusuf Sermet, Ibrahim Demir

**Institution(s):** University of Iowa, Tulane University


## Problem Background

随着AI在教育领域的广泛应用，语义检索系统对学术内容的独特语言和结构特性（如课程大纲中的术语和隐式表达）适应不足成为一大挑战。
通用嵌入模型由于训练于广域网络数据，在教育问答和语义搜索任务中性能不佳，而专有模型（如OpenAI嵌入服务）存在透明性、成本和数据隐私问题，限制了其在教育机构中的应用。
论文旨在开发针对教育领域的开源嵌入模型，提升语义检索性能，同时解决专有模型的局限性。

## Method

*   **核心思想：** 基于开源架构 *all-MiniLM-L6-v2*，通过领域特定数据集和两种微调策略，开发针对教育语义检索的嵌入模型，提升对学术语言细微差异的捕捉能力。
*   **数据集构建：** 合成生成包含3197个句子对的数据集，通过手动标注和大型语言模型（LLM，如GPT-4）辅助，涵盖同义术语、改写问题和隐式-显式映射等教育语义现象，确保数据反映学术话语特征。
*   **微调策略：**
    *   **对比学习（MNRL）：** 使用 MultipleNegativesRankingLoss，通过批次内正负样本对比优化嵌入空间中语义相似句子的相对距离，适用于检索任务中对相似结果的排序需求。
    *   **双损失优化（Dual-Loss）：** 结合 MNRL 和 CosineSimilarityLoss，前者优化相对排序，后者通过显式标签（相似为1，不相似为0）直接监督相似度分数，提升嵌入空间的语义校准能力；采用双数据加载器（DataLoader）分别提供正样本对和正负样本对，交替优化两个损失函数。
*   **训练配置：** 两种策略分别设置不同学习率（MNRL为2e-5，Dual-Loss为1e-5）、批次大小（64）和优化器（AdamW），配合WarmupCosine学习率调度器，确保模型适应教育领域任务。
*   **关键创新：** 双损失策略通过结合相对和绝对监督，增强模型对教育语言中细微语义差异（如‘办公时间’与‘讲座时间’）的区分能力，同时保持开源架构的透明性和低成本。

## Experiment

*   **有效性：** 在28个大学课程大纲上的评估显示，两种微调模型均显著优于开源基线（如 all-MiniLM-L6-v2 准确率在69.64%-92.86%），其中双损失模型在课程信息、教师信息和助教信息类别中分别达到100%、88.10%和87.50%的准确率，接近OpenAI的 text-embedding-3-large（90.18%-100%）。
*   **优越性：** 双损失模型相比单一MNRL模型（84.52%-100%）在教师和助教信息类别中表现更优，证明了双损失策略在语义校准上的优势，同时大幅缩小与专有模型的性能差距。
*   **实验设置：** 评估涵盖多机构、多学科的课程大纲，问题设计考虑语言变体（如‘教授姓名’的不同表达），通过手动验证答案准确性，确保结果可靠性；但固定大小分块策略可能割裂语义内容，影响检索精度，为论文中提到的局限性。
*   **开销：** 微调和推理基于轻量级架构，适合教育应用中的低延迟需求，训练过程未使用GPU，显示出较低的计算成本。

## Further Thoughts

双损失优化策略（结合相对排序和绝对相似度监督）不仅在教育领域有效，也可能推广到其他需要高语义分辨的领域，如医疗或法律文档检索；此外，通过LLM辅助生成领域特定合成数据集的思路，为资源有限的领域提供了低成本、高效的模型适配方案，值得探索如何结合更复杂的生成技术（如多模态数据）进一步提升数据集质量；最后，论文强调开源模型在隐私和成本效益上的潜力，启发我们思考如何通过社区协作，加速开发更多领域特定的嵌入模型，以替代专有解决方案。