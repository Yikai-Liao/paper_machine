---
title: "Multilingual Information Retrieval with a Monolingual Knowledge Base"
pubDatetime: 2025-06-03T07:05:49+00:00
slug: "2025-06-multilingual-retrieval-embedding"
type: "arxiv"
id: "2506.02527"
score: 0.4580864065252425
author: "grok-3-latest"
authors: ["Anurag Beniwal", "Yingying Zhuang", "Aman Gupta"]
tags: ["Information Retrieval", "Text Embedding", "Contrastive Learning", "Multilingual Retrieval", "Synthetic Data"]
institution: ["Amazon"]
description: "本文提出了一种基于加权采样和对比学习的嵌入模型微调策略，利用单语言知识库实现多语言信息检索，显著提升低资源语言和代码切换场景下的检索性能。"
---

> **Summary:** 本文提出了一种基于加权采样和对比学习的嵌入模型微调策略，利用单语言知识库实现多语言信息检索，显著提升低资源语言和代码切换场景下的检索性能。 

> **Keywords:** Information Retrieval, Text Embedding, Contrastive Learning, Multilingual Retrieval, Synthetic Data

**Authors:** Anurag Beniwal, Yingying Zhuang, Aman Gupta

**Institution(s):** Amazon


## Problem Background

多语言信息检索在全球化商业场景中至关重要，但高资源语言（如英语）的知识库资源丰富，而低资源语言和代码切换（如 Hinglish）场景下的知识库构建成本高昂且资源稀缺。
论文旨在解决如何通过嵌入模型将不同语言的查询映射到共享向量空间，从而利用单语言知识库支持多语言查询检索，推动跨语言知识共享和对话系统（Conversation AI）的包容性发展。

## Method

*   **核心思想:** 提出一种嵌入模型微调策略，通过对比学习（Contrastive Learning）将多语言查询与单语言知识库对齐，确保语义相似的查询在向量空间中距离更近。
*   **具体实现:** 设计了 Algorithm 1，包含以下步骤：
    *   **数据准备:** 将单语言知识库（如英语）分为索引集和训练集，利用大语言模型（LLM）将训练集查询翻译到目标语言（如 Hinglish）。
    *   **正负样本生成:** 为每个查询生成正样本对（语义相似查询）和负样本对（语义不同查询），正样本通过共享标签匹配，负样本采用加权采样策略。
    *   **加权采样策略:** 负样本生成结合随机负样本和基于标签相似性加权采样的难负样本（Hard Negatives），以平衡全局嵌入空间优化和局部语义区分，保持正负样本比例为 1:3。
    *   **合成数据增强:** 针对低资源语言数据稀缺问题，利用 LLM 生成目标语言的合成正负样本对，进一步扩充训练数据。
*   **训练细节:** 使用 InfoNCE 对比损失函数，在预训练多语言嵌入模型（如 multilingual-e5-base）上进行微调，优化嵌入向量空间的对齐效果。
*   **关键优势:** 方法语言无关，适用于任何目标语言和代码切换场景，且不依赖多语言知识库构建。

## Experiment

*   **有效性:** 实验以 Hinglish 查询和英语知识库为案例，基于加权采样策略微调的嵌入模型在 Recall@3 上提升至 0.7410（相比随机负样本策略提升约 1.8%），在 MRR 上达到 0.6653，显著优于仅使用难负样本或最难负样本的策略（Recall@1 分别提升约 18% 和 36%）。
*   **全面性:** 实验对比了多种开源多语言嵌入模型（如 multilingual-e5-base），并通过消融研究验证了混合负样本策略和合成数据增强的贡献，数据量充足（35,000 英语记录 + 10,000 Hinglish 数据）。
*   **合理性:** 评价指标（Recall@k 和 MRR）符合信息检索任务需求，实验设置涵盖多种策略对比，验证了方法的鲁棒性；数据匿名化虽限制商业直接适用，但不影响科学结论。
*   **局限与开销:** 合成数据依赖 LLM 生成，可能会引入噪声，训练成本因微调和数据生成而有所增加，但整体效果仍具优势。

## Further Thoughts

加权采样策略平衡全局和局部优化的思想可扩展至跨模态或跨领域任务，如文本-图像检索或知识迁移；利用 LLM 生成合成数据缓解低资源问题的方法值得深入探索，例如结合领域特定数据生成更高质量样本；此外，语言无关性提示我们可以在更多复杂语言现象（如方言）或非语言数据对齐任务中测试类似框架。