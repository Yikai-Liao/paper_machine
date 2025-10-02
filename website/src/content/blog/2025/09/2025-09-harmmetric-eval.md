---
title: "HarmMetric Eval: Benchmarking Metrics and Judges for LLM Harmfulness Assessment"
pubDatetime: 2025-09-29T07:34:01+00:00
slug: "2025-09-harmmetric-eval"
type: "arxiv"
id: "2509.24384"
score: 0.8305374947331977
author: "grok-3-latest"
authors: ["Langqi Yang", "Tianhang Zheng", "Kedong Xiu", "Yixuan Chen", "Di Wang", "Puning Zhao", "Zhan Qin", "Kui Ren"]
tags: ["LLM", "Harmfulness Assessment", "Jailbreak Attacks", "Evaluation Metrics", "Benchmarking"]
institution: ["The State Key Laboratory of Blockchain and Data Security, Zhejiang University", "Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security"]
description: "本文提出HarmMetric Eval基准测试框架，系统评估有害性指标和评判模型的有效性，并揭示传统指标如METEOR和ROUGE-1在某些场景下优于LLM评判模型的意外发现。"
---

> **Summary:** 本文提出HarmMetric Eval基准测试框架，系统评估有害性指标和评判模型的有效性，并揭示传统指标如METEOR和ROUGE-1在某些场景下优于LLM评判模型的意外发现。 

> **Keywords:** LLM, Harmfulness Assessment, Jailbreak Attacks, Evaluation Metrics, Benchmarking

**Authors:** Langqi Yang, Tianhang Zheng, Kedong Xiu, Yixuan Chen, Di Wang, Puning Zhao, Zhan Qin, Kui Ren

**Institution(s):** The State Key Laboratory of Blockchain and Data Security, Zhejiang University, Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security


## Problem Background

大型语言模型（LLMs）通过对齐技术确保输出符合人类价值观，但越狱攻击（jailbreak attacks）能够绕过安全机制，诱导模型生成有害内容。
为了评估越狱攻击的效果，研究界开发了多种有害性评估指标和评判模型，然而这些指标的有效性和可靠性缺乏系统性验证，导致越狱相关研究的结论可信度受到质疑。
本文旨在解决这一关键问题：如何系统性地评估有害性指标和评判模型的有效性，并揭示其潜在的优劣势。

## Method

*   **核心思想:** 提出一个名为HarmMetric Eval的基准测试框架，系统性地评估有害性指标和评判模型的有效性，通过标准化的数据集和评分机制揭示不同方法的优劣。
*   **有害性定义:** 定义了有害性响应的三个核心标准：不安全（Unsafe，指响应内容危险、毒性或非法）、相关性（Relevant，指响应直接针对提示词意图）、有用性（Useful，指响应提供逻辑合理且有效的帮助）。这三个标准确保评估覆盖真实攻击场景中的各种响应类型。
*   **数据集构建:** 构建了一个包含238个有害提示词的高质量数据集，每个提示词搭配14种响应（包括4种有害响应、3种安全响应、3种无关响应、2种无用响应），总计超过3300个响应样本，确保评估的多样性和细粒度。
*   **评分机制:** 设计了一种基于自我比较的评分机制（self-comparison-based scoring mechanism），通过整体有效性评分和细粒度评分，兼容不同输出格式和评分范围的指标，标准化评估过程。
*   **评估对象:** 评估了近20种现有有害性指标和评判模型，包括基于LLM的评判模型（如Llama Guard系列、GPT-4o模板）、有害性分类器（如RoBERTa、BERT-base）、字符串匹配方法（如GCG、AutoDAN）以及传统基于参考的NLP指标（如METEOR、ROUGE-1、BLEU）。

## Experiment

*   **整体效果:** 实验结果显示，现有有害性指标的整体有效性得分最高仅为0.634（满分1.0），表明当前指标在可靠性上仍有较大提升空间。
*   **方法对比:** 令人惊讶的是，传统基于参考的指标METEOR（得分0.634）和ROUGE-1（得分0.563）在整体有效性上超过了广泛使用的基于LLM的评判模型（如Llama Guard系列最高得分0.456，GPT-4o模板最高得分0.523）。
*   **细粒度分析:** LLM评判模型在区分安全与不安全响应方面表现较好，但在处理无用响应（如模糊肯定）时表现不佳；字符串匹配方法表现最差（最高得分仅0.008），尤其在处理重定向和无关响应时；METEOR和ROUGE-1在评估无关响应和无用响应时表现出色。
*   **实验设置合理性:** 实验覆盖了多种类型的指标和响应类别，数据集设计合理，能够模拟真实攻击场景；但数据集规模（238个提示词）相对较小，可能限制评估的泛化性。
*   **硬件支持:** 实验使用四块NVIDIA RTX A6000 GPU（每块48GB显存）完成，计算资源充足。

## Further Thoughts

论文揭示传统基于参考的NLP指标（如METEOR和ROUGE-1）在有害性评估中可能被低估，其基于词语相似性的方法在语义捕捉上表现出意外优势，启发未来研究可以探索将传统指标与LLM评判模型结合，构建更可靠的评估机制；此外，有害性三标准（Unsafe, Relevant, Useful）为评估设计提供了清晰的理论框架，可在其他安全相关研究中借鉴。