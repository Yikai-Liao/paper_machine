---
title: "Are LLMs Enough for Hyperpartisan, Fake, Polarized and Harmful Content Detection? Evaluating In-Context Learning vs. Fine-Tuning"
pubDatetime: 2025-09-09T14:01:15+00:00
slug: "2025-09-llm-misinformation-detection"
type: "arxiv"
id: "2509.07768"
score: 0.5913233429096122
author: "grok-3-latest"
authors: ["Michele Joshua Maggini", "Dhia Merzougui", "Rabiraj Bandyopadhyay", "Gaël Dias", "Fabrice Maurel", "Pablo Gamallo"]
tags: ["LLM", "Fine-Tuning", "In-Context Learning", "Classification", "Multilingual"]
institution: ["Universidade de Santiago de Compostela", "Normandie Univ", "GESIS Leibniz Institute for the Social Sciences"]
description: "本文通过全面基准测试，揭示了微调（Fine-Tuning）在误信息检测任务中相较上下文学习（In-Context Learning）的显著优势，并探索了提示策略在多语言、多任务场景下的局限与潜力。"
---

> **Summary:** 本文通过全面基准测试，揭示了微调（Fine-Tuning）在误信息检测任务中相较上下文学习（In-Context Learning）的显著优势，并探索了提示策略在多语言、多任务场景下的局限与潜力。 

> **Keywords:** LLM, Fine-Tuning, In-Context Learning, Classification, Multilingual

**Authors:** Michele Joshua Maggini, Dhia Merzougui, Rabiraj Bandyopadhyay, Gaël Dias, Fabrice Maurel, Pablo Gamallo

**Institution(s):** Universidade de Santiago de Compostela, Normandie Univ, GESIS Leibniz Institute for the Social Sciences


## Problem Background

在线平台上充斥着虚假新闻（Fake News）、极端党派内容（Hyperpartisan）、政治偏见（Political Bias）和有害内容（Harmful Content），对公共话语和民主完整性构成威胁。
大型语言模型（LLMs）被认为是解决这些问题的潜在工具，但缺乏对其在不同模型、使用方法和多语言场景下性能的系统性基准测试，论文旨在评估 LLMs 在检测上述内容时的适应性，重点对比微调（Fine-Tuning）和上下文学习（In-Context Learning）的表现。

## Method

*   **微调（Fine-Tuning, FT）**：通过在特定任务数据集上进一步训练预训练模型（包括编码器模型如 RoBERTa 和解码器模型如 LlaMA3.1-8B），以适应分类任务。实验采用参数高效微调技术（如 LoRA），针对不同任务（如 Hyperpartisan、Fake News、Political Bias、Harmful Content）优化模型参数，确保模型能够深度适应任务需求。
*   **上下文学习（In-Context Learning, ICL）**：不修改模型参数，仅通过提示（Prompt）引导模型完成任务，具体策略包括：
    *   **零样本（Zero-Shot）**：提供通用或特定任务描述的提示，无需示例，测试模型内部知识的应用能力。
    *   **少样本（Few-Shot）**：提供少量标注示例，测试随机选择与基于确定点过程（Determinantal Point Process, DPP）的多样性选择示例的效果，探索示例选择对模型泛化能力的影响。
    *   **代码书（Codebook）**：提供结构化的任务定义、分类规则和示例，增强模型的规则推理能力，特别适用于需要明确分类标准的任务。
    *   **思维链（Chain-of-Thought, CoT）**：引导模型逐步推理，将任务分解为子任务（如情感分析、修辞偏见检测），以提高分类的可解释性和准确性。
*   **实验设计**：方法覆盖编码器和解码器两种架构，测试不同模型规模和预训练数据的影响，同时在多语言（英语、西班牙语、葡萄牙语、阿拉伯语、保加利亚语）和多任务场景下进行实验，确保结果的广泛适用性。

## Experiment

*   **微调（FT）效果**：FT 整体表现优于 ICL，在33个测试场景中有28个表现最佳，解码器模型（如 LlaMA3.1-8B）在需要事实知识的任务（如 Fake News 和 Political Bias 检测）上表现更好（例如 LlaMA3.1-8B 在 Fake News 检测的 F1 得分达0.907），而编码器模型（如 RoBERTa）在语言风格导向的任务（如 Hyperpartisan 和 Harmful Content 检测）上更优（例如 RoBERTa-large 在 Hyperpartisan 检测的 F1 得分达0.857）。
*   **上下文学习（ICL）效果**：ICL 表现普遍不如 FT，零样本提示从通用到特定的改进有限（例如 LLaMA3.1-8B-Instruct 在 Hyperpartisan 任务的 F1 得分从0.678提升至0.738）；代码书方法在 Harmful Content 和部分 Fake News 任务上表现较好（例如 Mistral 在 C1B 数据集的 F1 得分达0.864）；少样本学习中 DPP 选择的多样性示例有时降低分类方差，但未显著提升性能；CoT 在大多数任务上表现不佳，仅在 FNN 数据集上有所提升。
*   **实验设置合理性**：实验覆盖10个数据集、5种语言、二分类和多分类任务，设置较为全面，数据集选择考虑了不同领域（新闻、推特）和时间跨度，增强了结果普适性，但部分数据集较老旧，可能无法完全反映当前误信息动态，计算资源和模型规模限制也可能影响结果全面性。

## Further Thoughts

代码书方法通过结构化规则和示例增强模型推理能力，启发我们在需要高可解释性的任务（如法律文本分析）中探索类似策略；DPP 示例选择降低分类方差，提示在数据稀缺场景下优化示例选择的潜力；任务类型与模型架构的适配性差异表明未来可根据任务特性定制模型选择，而非单纯追求规模；多语言性能波动揭示低资源语言支持不足，未来可聚焦多语言预训练数据多样性或开发专用模型。