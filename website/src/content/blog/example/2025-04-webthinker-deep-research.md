---
title: "WebThinker: Empowering Large Reasoning Models with Deep Research Capability"
pubDatetime: 2025-05-01T18:25:19+08:00
slug: "2025-04-webthinker-deep-research"
score: 0.7797082916340089
author: "grok-3-mini-latest"
authors: ["Xiaoxi Li", "Jiajie Jin", "Guanting Dong", "Hongjin Qian", "Yutao Zhu", "Yongkang Wu", "Ji-Rong Wen", "Zhicheng Dou"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Renmin University of China", "BAAI", "Huawei Poisson Lab"]
description: "本文提出WebThinker框架，通过Deep Web Explorer和Autonomous Think-Search-and-Draft策略增强LRMs的网页搜索与报告生成能力，并利用RL-based训练优化工具交互，实现显著的复杂任务性能提升。"
---

> **Summary:** 本文提出WebThinker框架，通过Deep Web Explorer和Autonomous Think-Search-and-Draft策略增强LRMs的网页搜索与报告生成能力，并利用RL-based训练优化工具交互，实现显著的复杂任务性能提升。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning
> **Recommendation Score:** 0.7797082916340089

**Authors:** Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen, Zhicheng Dou
**Institution(s):** Renmin University of China, BAAI, Huawei Poisson Lab

## Problem Background

大型推理模型（LRMs）如OpenAI-o1和DeepSeek-R1展示了出色的长程推理能力，但它们依赖静态内部知识，这限制了它们在复杂知识密集型任务上的性能，例如处理需要整合多样网络信息的任务，以及生成全面的研究报告，从而无法有效应对现实世界的深度研究需求。

## Method

*   **核心思想:** WebThinker框架旨在增强LRMs的深度研究能力，使其能够在推理过程中自主搜索网页、导航页面并起草报告，实现推理、搜索和写作的无缝整合。
*   **如何实现:** 该框架包括三个关键组件：(1) Deep Web Explorer模块，允许LRMs动态搜索网页、点击交互元素（如链接）并提取信息；(2) Autonomous Think-Search-and-Draft策略，让LRMs实时交错进行推理、搜索和报告写作，使用专用工具如起草章节、检查报告和编辑内容；(3) 基于强化学习的训练策略，通过迭代在线Direct Preference Optimization (DPO)生成偏好数据，优化LRMs的工具使用能力，从而提升整体任务性能，而不需修改模型参数。
*   **关键:** 该方法不依赖预定义工作流，而是让LRMs自主决策和交互，确保在保持推理能力的同时，高效利用外部知识。

## Experiment

*   **有效性:** WebThinker在复杂推理基准（如GPQA、GAIA、WebWalkerQA和HLE）上显著提升性能，例如在GAIA上比基线方法高出8.5%以上，在HLE上达到22.3%的改进；在科学报告生成任务（如Glaive）上，WebThinker的报告在完整性、彻底性和连贯性方面均优于RAG基线和专有系统，如平均得分从7.9提升到8.0。
*   **优越性:** 与直接推理或标准RAG相比，WebThinker展示了更强的泛化能力，尤其在需要深度网页探索的任务中；消融实验证明，Deep Web Explorer和RL训练是关键因素，去除它们会导致性能下降15%以上；实验设置全面合理，包括多种数据集、基线对比和迭代训练，确保结果可靠。
*   **开销:** 实验使用开源模型如QwQ-32B，训练涉及4节点8 NVIDIA H100 GPU，生成过程控制在81920 tokens以内，平衡了性能提升和计算成本。

## Further Thoughts

论文启发性想法包括：LRMs可以通过工具增强外部知识访问，这可能扩展到多模态环境或更高级的工具学习机制；此外，RL训练优化工具使用的策略可启发未来模型在动态交互场景中的自适应改进，例如结合更多外部数据源来提升泛化能力。