---
title: "Voila: Voice-Language Foundation Models for Real-Time Autonomous Interaction and Voice Role-Play"
pubDatetime: 2025-05-05T15:05:01+00:00
slug: "2025-05-voila-voice-autonomous"
type: "arxiv"
id: "2505.02707"
score: 0.6798594726714795
author: "grok-3-latest"
authors: ["Yemin Shi", "Yu Shu", "Siwei Dong", "Guangyi Liu", "Jaward Sesay", "Jingwen Li", "Zhiting Hu"]
tags: ["Voice AI", "End-to-End Model", "Full-Duplex Interaction", "Speech Tokenization", "Multimodal Alignment"]
institution: ["Maitrix.org", "UC San Diego", "MBZUAI"]
description: "本文提出 `Voila`，一个语音-语言基础模型家族，通过端到端架构、分层多尺度Transformer和文本-音频交错对齐，实现低延迟、自主的全双工语音交互，并支持高效语音定制和多任务处理，显著提升人机交互自然性。"
---

> **Summary:** 本文提出 `Voila`，一个语音-语言基础模型家族，通过端到端架构、分层多尺度Transformer和文本-音频交错对齐，实现低延迟、自主的全双工语音交互，并支持高效语音定制和多任务处理，显著提升人机交互自然性。 

> **Keywords:** Voice AI, End-to-End Model, Full-Duplex Interaction, Speech Tokenization, Multimodal Alignment

**Authors:** Yemin Shi, Yu Shu, Siwei Dong, Guangyi Liu, Jaward Sesay, Jingwen Li, Zhiting Hu

**Institution(s):** Maitrix.org, UC San Diego, MBZUAI


## Problem Background

当前语音AI系统在实现自主、实时和情感表达的人机交互方面存在局限性，传统系统依赖模块化管道，反应式交互模式导致对话僵硬，无法主动参与或模拟人类对话的动态性；此外，高延迟、语音细微差别（如语调、情感）丢失以及缺乏全双工交互能力是主要问题。本文旨在构建一个语音-语言基础模型，以低延迟、保留丰富语音细节的方式实现自然、自主的实时交互，并支持多种语音任务。

## Method

* **核心架构：端到端设计**：摒弃传统管道系统（ASR-LLM-TTS），直接处理音频输入和输出，避免多模块延迟，保留语音细节如语调和情感。
* **分层多尺度Transformer**：结合预训练的大型语言模型（LLM）骨干和音频Transformer，LLM 负责语义信息处理，音频Transformer 专注于声学信息生成，确保语义理解和语音生成的高质量。
* **语音分词器（Voila-Tokenizer）**：通过残差向量量化（RVQ）将连续音频信号转化为离散语义和声学令牌，首层令牌聚焦语义，后三层捕捉声学细节，提升模型训练效率。
* **文本-音频交错对齐**：采用交错排列策略，将文本语义单元与对应音频片段紧密对齐，提升生成语音的表达力和同步性，优于现有松散耦合方法。
* **全双工交互能力**：`Voila-autonomous` 支持同时监听、推理和回应，模拟人类对话中的自然交互行为，如打断和回馈。
* **语音定制化与角色扮演**：通过文本指令定义角色特性，并利用参考语音嵌入（最短10秒音频样本）快速定制新语音，预构建超过百万种语音，支持个性化交互。

## Experiment

* **整体性能**：在自建的 `Voila Benchmark` 上，`Voila` 准确率达 30.56%，显著优于 SpeechGPT (13.29%) 和 Moshi (11.45%)，尤其在数学和编程领域表现突出，体现其文本-音频对齐对 LLM 推理能力的有效利用。
* **任务特定表现**：在 ASR 任务中，`Voila` 词错率（WER）为 4.8%（未用 LibriSpeech 训练数据），优于 Moshi (5.7%)，加入训练数据后达 2.7%，与最佳模型持平；在 TTS 任务中，WER 为 3.2%，优于 Moshi (4.7%) 和 Vall-E (5.9%)。
* **延迟优势**：响应延迟仅 195 毫秒，低于人类平均响应时间（约 300 毫秒），证明其实时交互能力。
* **实验设置**：实验覆盖对话、ASR、TTS 多种任务，基准测试包含多个领域（如数学、历史、编程），设计全面且合理；但未深入探讨多语言表现差异，可能为未来改进方向。

## Further Thoughts

端到端架构与全双工交互的结合为多模态自主交互提供了新思路，可扩展至视觉-语音融合场景；语音定制化机制（短音频样本快速生成个性化语音）启发个性化AI助手的构建，未来可结合情感识别实现自适应交互；统一模型支持多任务（如 ASR、TTS、翻译）的设计，提示我们探索更通用多模态基础模型的可能性，减少任务特定调整。