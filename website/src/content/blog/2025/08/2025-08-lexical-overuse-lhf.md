---
title: "Word Overuse and Alignment in Large Language Models: The Influence of Learning from Human Feedback"
pubDatetime: 2025-08-03T21:45:37+00:00
slug: "2025-08-lexical-overuse-lhf"
type: "arxiv"
id: "2508.01930"
score: 0.8457909226549587
author: "grok-3-latest"
authors: ["Tom S. Juzek", "Zina B. Ward"]
tags: ["LLM", "Alignment", "Preference Learning", "Lexical Overuse", "Human Feedback"]
institution: ["Florida State University"]
description: "本文通过识别 LHF 诱导的词汇偏好并实验验证人类偏好，提供了证据表明学习人类反馈（LHF）是大型语言模型词汇过度使用的重要原因，同时揭示了模型对齐中的文化和任务设计问题。"
---

> **Summary:** 本文通过识别 LHF 诱导的词汇偏好并实验验证人类偏好，提供了证据表明学习人类反馈（LHF）是大型语言模型词汇过度使用的重要原因，同时揭示了模型对齐中的文化和任务设计问题。 

> **Keywords:** LLM, Alignment, Preference Learning, Lexical Overuse, Human Feedback

**Authors:** Tom S. Juzek, Zina B. Ward

**Institution(s):** Florida State University


## Problem Background

大型语言模型（LLMs）常过度使用某些词汇（如‘delve’和‘intricate’），这种现象虽已被广泛观察，但其原因尚未明确。
本文探究‘学习人类反馈’（Learning from Human Feedback, LHF，包括 RLHF 和 DPO）是否是导致词汇过度使用的关键因素，并讨论这种现象是否构成模型对齐（Alignment）中的‘错位’（Misalignment），即模型输出与目标用户群体的语言期望不符。

## Method

*   **核心思想：** 研究 LHF 是否导致 LLMs 的词汇过度使用，通过比较未训练和训练后模型的输出差异，并实验验证人类是否偏好特定词汇。
*   **具体步骤：**
    *   **词汇偏好识别：** 使用 Meta 的 Llama 模型（Llama 3.2-3B Base 和 Llama 3.2-3B Instruct），生成基于 PubMed 摘要的语料（约 450 万词），比较未经过 LHF 训练的 Base 模型和经过 LHF 训练的 Instruct 模型在词汇使用上的差异，通过卡方检验等统计方法识别显著增加的词汇（如‘nuanced’增加 8342%）。
    *   **人类偏好实验：** 设计文本对，分别包含高 LHF 评分词汇和低 LHF 评分词汇，招募 400 名参与者（主要来自全球南方）进行偏好评估，使用‘LHF-Score’量化词汇的潜在 LHF 影响，并通过卡方检验和混合线性回归分析结果。
*   **技术细节：** 实验控制了文本长度等变量，使用 Python 和 spaCy 进行数据处理和词性标注，确保结果的可比性。

## Experiment

*   **有效性：** 词汇识别阶段成功识别出多个与文献中 LLM 过度使用词汇高度重合的词，人类偏好实验显示参与者对高 LHF 评分词汇文本有显著偏好（52.4% vs 47.6%，p < 0.01），支持 LHF 导致词汇过度使用的假设。
*   **合理性：** 实验设置较为全面，控制了文本长度等变量，并通过‘gotcha’项和响应时间阈值确保数据质量，但排除率较高（46.8%），可能影响结果代表性。
*   **局限性：** 某些词汇（如‘nuanced’）表现出反向偏好（偏好率仅 46.6%），提示过度使用词汇可能因用户反感而产生复杂影响；参与者主要来自全球南方，样本代表性有限。

## Further Thoughts

词汇过度使用可能反映了语言变迁的加速，与 LHF 工作者的年龄和地域背景（如全球南方）有关，这提示 AI 模型不仅是技术工具，也是文化和语言影响的传播者；此外，LHF 任务设计可能导致人类评估者将某些词汇作为质量代理，表明模型对齐需结合任务设计和人类行为心理学研究，或许可以通过多元化 LHF 数据集或优化任务设计减少词汇偏见。