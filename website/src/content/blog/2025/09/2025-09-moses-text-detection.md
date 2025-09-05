---
title: "MoSEs: Uncertainty-Aware AI-Generated Text Detection via Mixture of Stylistics Experts with Conditional Thresholds"
pubDatetime: 2025-09-02T16:51:43+00:00
slug: "2025-09-moses-text-detection"
type: "arxiv"
id: "2509.02499"
score: 0.4895117459032098
author: "grok-3-latest"
authors: ["Junxi Wu", "Jinpeng Wang", "Zheng Liu", "Bin Chen", "Dongjian Hu", "Hao Wu", "Shu-Tao Xia"]
tags: ["LLM", "Text Detection", "Stylistic Modeling", "Uncertainty Quantification", "Dynamic Threshold"]
institution: ["Nankai University", "Tsinghua University", "Harbin Institute of Technology, Shenzhen", "Peng Cheng Laboratory", "Shenzhen ShenNong Information Technology Co., Ltd."]
description: "本文提出 MoSEs 框架，通过风格感知和动态阈值估计，显著提升了 AI 生成文本检测性能，尤其在低资源场景下。"
---

> **Summary:** 本文提出 MoSEs 框架，通过风格感知和动态阈值估计，显著提升了 AI 生成文本检测性能，尤其在低资源场景下。 

> **Keywords:** LLM, Text Detection, Stylistic Modeling, Uncertainty Quantification, Dynamic Threshold

**Authors:** Junxi Wu, Jinpeng Wang, Zheng Liu, Bin Chen, Dongjian Hu, Hao Wu, Shu-Tao Xia

**Institution(s):** Nankai University, Tsinghua University, Harbin Institute of Technology, Shenzhen, Peng Cheng Laboratory, Shenzhen ShenNong Information Technology Co., Ltd.


## Problem Background

随着大型语言模型（LLMs）的快速发展，其生成内容的潜在误用（如虚假新闻、学术不端）引发了广泛关注，亟需可靠的 AI 生成文本检测系统。
现有方法忽视了文本风格（Stylistics）建模，未能捕捉人类写作中由文化和职业背景塑造的特定习惯（Habitus），同时依赖静态阈值决策，缺乏对语言特性和不确定性的量化，导致检测性能受限。

## Method

*   **核心思想:** 提出 MoSEs（Mixture of Stylistics Experts）框架，通过风格感知和动态阈值估计，实现不确定性量化的 AI 生成文本检测。
*   **组件一 - Stylistics Reference Repository (SRR):** 构建一个多风格参考数据池，包含新闻、学术论文、对话等多种领域的数据，每条样本标注来源标签（人类/AI生成）和多维特征（如文本长度、n-gram 重复率、语义嵌入），用于捕捉跨领域文本分布差异。
*   **组件二 - Stylistics-Aware Router (SAR):** 基于预训练语言编码器构建语义潜在空间，通过原型聚类（Prototype-based Approximation）和 m-最近邻搜索，动态选择与输入文本风格最相似的参考样本组，而非简单分类到固定风格，减少计算开销并提升风格适配性。
*   **组件三 - Conditional Threshold Estimator (CTE):** 结合语言统计属性（如文本长度、词概率分布）和语义特征，动态估计分类阈值，支持不确定性量化；提供两种实现方式：逻辑回归（MoSEs-lr）用于线性建模，XGBoost（MoSEs-xg）捕捉特征间非线性交互，最终输出预测标签及置信度。
*   **关键创新:** 将风格建模与条件阈值估计结合，通过动态决策机制提升检测适应性和可靠性，尤其适用于风格多样和低资源场景。

## Experiment

*   **有效性:** MoSEs-xg 在主数据集（4 个风格，1800 参考样本+200 测试样本）上平均准确率提升 11.34%，在低资源数据集（200 参考样本+200 测试样本）上提升更显著，达到 39.15%（以 RoBERTa 为例），表明方法在数据稀缺时仍具高性能。
*   **优越性:** 相比静态阈值和最近邻投票方法，MoSEs 通过动态阈值调整显著提升检测性能，尤其在风格多样性高的场景下；McNemar 检验证实改进具有统计显著性。
*   **实验设置合理性:** 实验覆盖多种风格（新闻、学术、对话等）和资源条件，测试了不同判别模型（RoBERTa、Fast-DetectGPT、Lastde），并通过分布外（OOD）测试验证了跨风格和跨模型的泛化能力。
*   **不足与开销:** MoSEs-xg 推理时间（1.04ms）较静态阈值方法慢，计算开销可能限制实时应用；语义特征压缩（PCA）虽提升效率，但可能丢失深层信息。

## Further Thoughts

风格建模的思路可扩展至其他 NLP 任务，如作者身份识别或文本生成质量评估，通过更细粒度的风格分解（如情感、语调）进一步提升精度；动态阈值估计机制为不确定性量化提供了新视角，可应用于数据分布不均的分类任务；低资源场景下的显著性能提示通过迁移学习或元学习减少数据依赖，甚至实现零样本检测；此外，是否可结合多模态数据（如图像标题）扩展框架，应对更复杂的 AI 生成内容？