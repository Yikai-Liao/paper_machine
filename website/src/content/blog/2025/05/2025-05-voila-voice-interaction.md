---
title: "Voila: Voice-Language Foundation Models for Real-Time Autonomous Interaction and Voice Role-Play"
pubDatetime: 2025-05-05T15:05:01+00:00
slug: "2025-05-voila-voice-interaction"
type: "arxiv"
id: "2505.02707"
score: 0.7447861716048989
author: "grok-3-latest"
authors: ["Yemin Shi", "Yu Shu", "Siwei Dong", "Guangyi Liu", "Jaward Sesay", "Jingwen Li", "Zhiting Hu"]
tags: ["Voice AI", "LLM", "End-to-End Modeling", "Full-Duplex Interaction", "Speech Customization"]
institution: ["Maitrix.org", "UC San Diego", "MBZUAI"]
description: "本文提出 `Voila`，一个端到端、低延迟的语音-语言基础模型家族，通过分层建模和全双工交互设计，显著提升语音AI的自然性和自主性。"
---

> **Summary:** 本文提出 `Voila`，一个端到端、低延迟的语音-语言基础模型家族，通过分层建模和全双工交互设计，显著提升语音AI的自然性和自主性。 

> **Keywords:** Voice AI, LLM, End-to-End Modeling, Full-Duplex Interaction, Speech Customization

**Authors:** Yemin Shi, Yu Shu, Siwei Dong, Guangyi Liu, Jaward Sesay, Jingwen Li, Zhiting Hu

**Institution(s):** Maitrix.org, UC San Diego, MBZUAI


## Problem Background

当前语音AI系统在实现自然、实时和自主交互方面存在显著不足：传统模块化管道系统（如 Siri、Alexa）延迟高、丢失语音情感和语调细节，且交互模式僵硬（turn-based）；即使基于大型语言模型（LLM）的改进系统，也无法完全模拟人类对话中的动态性和情感共鸣。
论文旨在解决如何构建一个低延迟、保留语音细微特征、支持全双工实时交互的语音-语言模型，并实现自主交互能力（即AI能主动参与对话，而非被动响应）。

## Method

*   **核心思想：** 提出 `Voila`，一个语音-语言基础模型家族，通过端到端架构和全双工设计，实现低延迟、自然且自主的语音交互，同时保留语音的语调、情感等细微特征。
*   **具体实现：**
    *   **端到端架构：** 摒弃传统管道式设计（ASR-LLM-TTS），直接处理音频输入并生成音频输出，避免中间文本转换导致的延迟和信息丢失。
    *   **分层多尺度Transformer：** 结合LLM骨干网络（处理语义信息）和音频Transformer（处理声学信息），实现语义与声学的解耦和高效建模，确保推理能力与语音质量兼顾。
    *   **语音分词器（Voila-Tokenizer）：** 将连续音频信号转化为离散语义和声学token，语义token关注语言内容，声学token保留音色、语调等细节，支持跨模态训练。
    *   **文本-音频交错对齐：** 通过交错排列文本和音频token，确保语义与声学信息的细粒度对齐，提升生成语音的自然度和同步性。
    *   **全双工交互设计：** `Voila-autonomous` 支持同时监听和回应，模拟人类对话中的打断、反馈等动态行为，增强交互的自主性和自然性。
    *   **语音定制化：** 通过语音嵌入（voice embedding）和文本指令，支持快速定制新语音（最短10秒音频样本）和角色扮演，预构建超过百万种语音。
*   **关键优势：** 整合LLM的推理能力与语音建模能力，响应延迟低至195毫秒，同时支持多任务（如ASR、TTS）统一建模。

## Experiment

*   **有效性：** 在 `Voila` Benchmark 上，`Voila` 平均准确率为 30.56%，显著优于 SpeechGPT (13.29%) 和 Moshi (11.45%)，尤其在数学和编程领域表现突出，证明其文本-音频对齐策略有效利用了LLM推理能力；在ASR任务中，词错误率（WER）为 4.8%（未使用LibriSpeech训练数据时），优于 Moshi (5.7%)，加入训练数据后达 2.7%，与最佳模型持平；在TTS任务中，WER 为 3.2%，优于 Moshi (4.7%) 和 Vall-E (5.9%)。
*   **延迟表现：** 响应延迟仅195毫秒，低于人类平均响应时间（约300毫秒），在实时交互上取得突破。
*   **实验设置合理性：** 实验覆盖对话、ASR、TTS等多任务场景，基准测试数据来源广泛（MMLU、MATH等），通过第三方工具（如 Whisper、GPT-4o）评估结果，增强可信度；但全双工交互的具体评估指标（如打断检测准确率）未详细讨论，存在一定局限。

## Further Thoughts

论文启发我们思考如何将端到端与全双工交互机制推广到其他多模态任务（如视觉-语言交互），以实现更自然的交互体验；语义与声学解耦的思路可应用于视频生成中内容与风格的分离建模；此外，快速语音定制化方法提示我们探索个性化AI助手设计，通过少量用户数据快速适配模型行为和风格。