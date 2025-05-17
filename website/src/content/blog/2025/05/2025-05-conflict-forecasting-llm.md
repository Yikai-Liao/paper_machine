---
title: "Do Large Language Models Know Conflict? Investigating Parametric vs. Non-Parametric Knowledge of LLMs for Conflict Forecasting"
pubDatetime: 2025-05-14T23:24:22+00:00
slug: "2025-05-conflict-forecasting-llm"
type: "arxiv"
id: "2505.09852"
score: 0.6023169199760546
author: "grok-3-latest"
authors: ["Apollinaire Poli Nemkova", "Sarath Chandra Lingareddy", "Sagnik Ray Choudhury", "Mark V. Albert"]
tags: ["LLM", "Conflict Forecasting", "Parametric Knowledge", "Non-Parametric Knowledge", "Retrieval-Augmented Generation"]
institution: ["University of North Texas, USA"]
description: "本文通过对比大型语言模型在冲突预测中的参数和非参数知识表现，证明检索增强生成（RAG）能显著提升预测性能，为 AI 辅助预警系统提供了实证指导。"
---

> **Summary:** 本文通过对比大型语言模型在冲突预测中的参数和非参数知识表现，证明检索增强生成（RAG）能显著提升预测性能，为 AI 辅助预警系统提供了实证指导。 

> **Keywords:** LLM, Conflict Forecasting, Parametric Knowledge, Non-Parametric Knowledge, Retrieval-Augmented Generation

**Authors:** Apollinaire Poli Nemkova, Sarath Chandra Lingareddy, Sagnik Ray Choudhury, Mark V. Albert

**Institution(s):** University of North Texas, USA


## Problem Background

冲突预测对于人道主义预警系统、资源分配和政策制定至关重要，但传统方法依赖统计模型和人工特征工程，缺乏泛化能力。
本文探讨大型语言模型（LLMs）是否能通过其预训练权重中的参数知识（parametric knowledge）预测暴力冲突的升级和伤亡人数，以及是否可以通过非参数方法（如检索增强生成，RAG）引入外部上下文来提升预测性能。

## Method

*   **实验设计核心思想:** 对比 LLMs 在冲突预测任务中的参数知识和非参数知识表现，评估其在高风险领域的适用性。
*   **具体实现方式:**
    *   **参数预测（Parametric Forecasting）:** 仅依赖 LLMs 的预训练知识（零样本设置），通过提示模型预测给定国家未来一个月的冲突趋势（分类为升级、降级、稳定冲突或和平）和伤亡人数（回归任务）。
    *   **非参数预测（Retrieval-Augmented Generation, RAG）:** 通过 RAG 管道为模型提供外部上下文，包括过去三个月的新闻摘要（通过 FAISS 检索和 GPT-3.5 总结）、GDELT 数据（文章语气和 Goldstein 评分）以及 ACLED 的周度伤亡数据，然后预测下一个月的冲突趋势和伤亡人数。
    *   **模型选择:** 使用 GPT-4（通过 OpenAI API）和 LLaMA-2-13B-chat（通过 Hugging Face Transformers）进行实验，温度参数设为 0.2 以控制生成随机性。
    *   **数据与任务:** 数据覆盖 2020-2024 年非洲之角和中东地区，任务包括分类（冲突趋势标签）和回归（伤亡人数预测），并进一步从伤亡预测推导分类标签。
*   **评价框架:** 使用准确率、精确率、召回率和 F1 分数（micro, macro, weighted）评估分类任务，使用平均绝对误差（MAE）评估回归任务。

## Experiment

*   **参数预测效果:** GPT-4 在参数设置下表现较好，尤其在从伤亡人数推导冲突类别任务中（如索马里准确率达 0.83），表明其预训练知识编码了一些广义冲突模式；但在细粒度分类任务中表现较差（macro-F1 分数普遍低于 0.22）。LLaMA-2 表现明显较弱，参数知识不足以支持复杂预测。
*   **非参数预测效果（RAG）:** 引入外部上下文后，GPT-4 性能显著提升，尤其在分类任务（如以色列 F1-macro 从 0.16 升至 0.33）和伤亡预测中（以色列 MAE 从 218.48 降至 93.54）；LLaMA-2 提升有限，部分地区甚至表现下降（如埃塞俄比亚），显示其整合上下文能力较弱。
*   **实验设置分析:** 实验覆盖多个冲突频发地区和五年时间跨度，数据来源（ACLED, GDELT）权威，任务设计全面（分类+回归）；但样本量较小（每个国家仅 59 个月度数据点），类别分布不平衡（稳定冲突和和平占主导），可能影响评估可靠性。

## Further Thoughts

论文中 RAG 在提升冲突预测性能方面的作用令人启发，特别是在高风险领域如何处理参数知识与外部上下文的冲突（knowledge conflict）值得进一步探索。是否可以通过更精细的检索机制（如多语言新闻源）或动态调整上下文权重来优化 RAG 效果？此外，LLaMA-2 在特定场景下表现接近 GPT-4，提示开源模型在资源受限环境中的潜力，是否可以通过领域特定微调进一步缩小性能差距？