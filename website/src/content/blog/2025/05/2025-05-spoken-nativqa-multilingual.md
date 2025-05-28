---
title: "SpokenNativQA: Multilingual Everyday Spoken Queries for LLMs"
pubDatetime: 2025-05-25T14:22:18+00:00
slug: "2025-05-spoken-nativqa-multilingual"
type: "arxiv"
id: "2505.19163"
score: 0.6617697091176726
author: "grok-3-latest"
authors: ["Firoj Alam", "Md Arid Hasan", "Shammur Absar Chowdhury"]
tags: ["LLM", "Spoken Queries", "Multilingual Data", "Speech Recognition", "Question Answering"]
institution: ["Qatar Computing Research Institute", "University of New Brunswick"]
description: "本文提出 *SpokenNativQA*，一个多语言、真实口语的日常问答数据集，通过基准测试揭示 ASR 错误对 LLMs 性能的影响，并验证端到端多模态模型在减少错误传播方面的潜力。"
---

> **Summary:** 本文提出 *SpokenNativQA*，一个多语言、真实口语的日常问答数据集，通过基准测试揭示 ASR 错误对 LLMs 性能的影响，并验证端到端多模态模型在减少错误传播方面的潜力。 

> **Keywords:** LLM, Spoken Queries, Multilingual Data, Speech Recognition, Question Answering

**Authors:** Firoj Alam, Md Arid Hasan, Shammur Absar Chowdhury

**Institution(s):** Qatar Computing Research Institute, University of New Brunswick


## Problem Background

大型语言模型（LLMs）在处理多语言口语查询（Spoken Queries）方面的能力尚未被充分探索，现有口语问答（SQA）数据集多为英语且多为合成数据，缺乏真实、多语言、日常场景下的口语数据；本文旨在通过构建一个多语言、真实口语的问答数据集 *SpokenNativQA*，评估 LLMs 在真实口语场景中的表现，解决 ASR 错误传播和口语特有现象（如犹豫、口音）对问答性能的影响问题。

## Method

* **数据集构建**：基于 MultiNativQA 数据集，选取阿拉伯语和英语（卡塔尔地区）的测试集（分别包含 988 和 2322 个问答对），招募阿拉伯语为母语、英语为第二语言的发言者，在无特定录音环境限制下录制问题音频，模拟真实场景；每个问题由 10 位发言者录制，最终形成约 33,000 个样本、30 小时音频数据，覆盖 18 个日常话题。
* **基准测试方法**：采用级联方法（Cascaded Approach），即先使用自动语音识别（ASR）系统（如 Google、Azure、Whisper、Fanar）将音频转录为文本，再输入到 LLMs（如 GPT-4o、Fanar、ALLaM）进行问答；同时测试直接处理音频的多模态模型（如 GPT-4o-audio）；使用零样本学习（Zero-Shot Learning）设置，通过自定义提示（Prompt）优化模型输出。
* **评估指标**：采用基于语义相似度的 BERTScore（F1 分数）作为评估指标，利用语言特定的预训练模型（如 AraBERT 和 bert-base-uncased）提取上下文嵌入，衡量问答性能。
* **核心目标**：通过真实多语言口语数据，揭示 ASR 错误对下游问答任务的影响，并对比级联方法与端到端多模态模型的性能差异。

## Experiment

* **ASR 性能**：在阿拉伯语中，Google 的卡塔尔地区 ASR 表现最佳（WER 5.85），而在英语中 Whisper 表现最佳（WER 10.58），可能与发言者语言背景（阿拉伯语为 L1，英语为 L2）导致的口音多样性有关。
* **SQA 性能**：无 ASR 设置（直接使用标准文本问题）下性能最高（阿拉伯语 F1 0.536，英语 F1 0.619）；引入 ASR 后性能下降，但 Whisper 在英语中表现最接近无 ASR 设置；GPT-4o-audio（直接处理音频）表现优异，阿拉伯语 F1 0.55，英语 F1 0.62，甚至超越部分无 ASR 设置。
* **实验设置合理性**：实验覆盖多种 ASR 系统和 LLMs，考虑了多语言和文化对齐场景，设置较为全面；但测试集规模较小（阿拉伯语 988 样本，英语 2322 样本），可能限制结果泛化性，且未深入探讨话题分布对性能的影响。
* **显著性**：实验表明 ASR 错误显著影响 SQA 性能，端到端多模态模型在减少错误传播方面有明显优势，验证了研究问题的重要性。

## Further Thoughts

真实口语数据的重要性启发我们进一步扩展到更多低资源语言和方言，构建更具包容性的数据集；端到端多模态模型优于级联方法的表现提示未来可聚焦直接从音频提取语义信息，探索预训练音频-语言模型在多语言 SQA 中的潜力；文化对齐数据设计是否可引入更多上下文（如用户背景、对话历史）以增强日常查询理解；ASR 错误对不同语言影响的差异是否可通过针对性微调缓解？