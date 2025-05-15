---
title: "QUPID: Quantified Understanding for Enhanced Performance, Insights, and Decisions in Korean Search Engines"
pubDatetime: 2025-05-12T08:35:09+00:00
slug: "2025-05-heterogeneous-slm-ensemble"
type: "arxiv"
id: "2505.07345"
score: 0.4502472736400084
author: "grok-3-latest"
authors: ["Ohjoon Kwon", "Changsu Lee", "Jihye Back", "Lim Sun Suk", "Inho Kang", "Donghyeon Jeon"]
tags: ["LLM", "SLM", "Information Retrieval", "Ensemble Model", "Relevance Assessment"]
institution: ["Naver Corporation"]
description: "本文提出 QUPID 方法，通过异构集成生成式和嵌入式小型语言模型（SLMs），在相关性评估任务中显著提升精度并降低计算成本，为大规模搜索系统提供了高效可扩展的解决方案。"
---

> **Summary:** 本文提出 QUPID 方法，通过异构集成生成式和嵌入式小型语言模型（SLMs），在相关性评估任务中显著提升精度并降低计算成本，为大规模搜索系统提供了高效可扩展的解决方案。 

> **Keywords:** LLM, SLM, Information Retrieval, Ensemble Model, Relevance Assessment

**Authors:** Ohjoon Kwon, Changsu Lee, Jihye Back, Lim Sun Suk, Inho Kang, Donghyeon Jeon

**Institution(s):** Naver Corporation


## Problem Background

大型语言模型（LLMs）在信息检索（IR）中的相关性评估任务中表现出色，但其高计算成本、延迟问题和内存占用使其在处理每日数亿查询的实际搜索系统中难以部署，尤其是在多语言环境（如韩语）中表现不佳。
小型语言模型（SLMs）虽然成本较低，但单一 SLM 的表达能力和上下文理解能力不足，且现有 SLM 集成方法多采用同构模型，未能充分利用不同架构模型的互补优势。
因此，论文试图解决如何在保持高相关性评估精度的同时，显著降低计算成本和延迟，使其适用于大规模搜索系统。

## Method

*   **核心思想:** 提出 QUPID 方法，通过异构集成两个不同架构的小型语言模型（SLMs），即生成式 SLM 和嵌入式 SLM，结合两者的互补优势，提升相关性评估精度，同时降低计算成本。
*   **生成式 SLM (QUPID GEN):** 
    *   基于 token 概率进行上下文推理，模型经过细调后，针对每个查询-文档对输出特定标签 token（如 'Highly Relevant', 'Low Relevance', 'Not Relevant'）的概率分布。
    *   通过 softmax 函数将 log 概率转换为概率值，并结合标签权重计算最终相关性分数，擅长捕捉上下文语义和复杂推理。
*   **嵌入式 SLM (QUPID EMB):** 
    *   通过提取查询-文档对的 token 级隐藏状态嵌入（hidden state embeddings），应用均值池化（mean pooling）生成固定大小的表示。
    *   再通过线性变换和 softmax 函数将嵌入映射为相关性分数向量，擅长捕捉语义相似性和分布接近性。
*   **集成策略:** 
    *   通过加权平均方法结合两模型的输出分数，权重可通过验证集性能调优或启发式设定。
    *   这种异构集成确保了生成式推理和嵌入式相似性信号的平衡，提升了评估的鲁棒性和准确性。
*   **关键优势:** 不依赖大型模型，计算成本低，且通过架构多样性弥补了单一 SLM 的局限性，适用于高吞吐量的搜索环境。

## Experiment

*   **有效性:** QUPID 集成模型在 Cohen’s Kappa 上达到 0.646，远超代表性 LLMs（如 ChatGPT-4o 的 0.387），AUC 指标也显著优于基线，表明其相关性评估能力更接近人类判断。
*   **效率提升:** 推理延迟为 62ms，相比 LLMs（如 LLaMA3.3-70b 的 3258ms）快约 60 倍，计算成本大幅降低，适合大规模部署。
*   **实际应用:** 在搜索结果排名任务中，QUPID 提升了 nDCG@5 指标约 1.9%，并在多个用例（如低质量过滤、查询重写评估、片段质量评估）中展现了实用性。
*   **实验设置合理性:** 实验覆盖多种文档类型（片段式网页内容、用户生成内容、一般网页文档），数据集规模约 100 万查询-文档对，数据标注由专业团队完成，确保可靠性；对比基线包括多种 LLMs 和 SLM 集成方法，评估指标（如 Cohen’s Kappa, AUC）全面。
*   **局限性:** 实验主要针对韩语环境，未深入探讨模型在其他语言或多模态场景下的泛化能力。

## Further Thoughts

论文中异构模型集成的思路令人启发，传统集成多采用同构模型，而 QUPID 通过结合生成式和嵌入式 SLM 展示了架构多样性的潜力，这提示我们是否可以进一步扩展到更多类型模型（如基于图神经网络或多模态模型）以应对复杂搜索场景；此外，调整推理温度优化概率分布的策略也启发我们可以在推理阶段引入更多动态调整机制（如基于查询复杂度的自适应参数）来提升性能。