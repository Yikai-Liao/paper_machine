---
title: "Voila: Voice-Language Foundation Models for Real-Time Autonomous Interaction and Voice Role-Play"
pubDatetime: 2025-05-05T15:05:01+00:00
slug: "2025-05-voila-voice-interaction"
type: "arxiv"
id: "2505.02707"
score: 0.6798594726714795
author: "grok-3-latest"
authors: ["Yemin Shi", "Yu Shu", "Siwei Dong", "Guangyi Liu", "Jaward Sesay", "Jingwen Li", "Zhiting Hu"]
tags: ["LLM", "Voice Interaction", "End-to-End Model", "Full-Duplex", "Text-Audio Alignment"]
institution: ["Maitrix.org", "UC San Diego", "MBZUAI"]
description: "本文提出 `Voila` 语音-语言基础模型家族，通过端到端全双工架构、文本-音频交错对齐和语音定制技术，实现低延迟、自主、自然的语音交互，显著提升人机交互体验。"
---

> **Summary:** 本文提出 `Voila` 语音-语言基础模型家族，通过端到端全双工架构、文本-音频交错对齐和语音定制技术，实现低延迟、自主、自然的语音交互，显著提升人机交互体验。 

> **Keywords:** LLM, Voice Interaction, End-to-End Model, Full-Duplex, Text-Audio Alignment

**Authors:** Yemin Shi, Yu Shu, Siwei Dong, Guangyi Liu, Jaward Sesay, Jingwen Li, Zhiting Hu

**Institution(s):** Maitrix.org, UC San Diego, MBZUAI


## Problem Background

当前语音AI系统多为反应式交互，依赖管道式架构，导致高延迟、语音细节丢失和机械的交互模式，无法实现类似人类对话的实时、自主、情感丰富的交互体验。
论文旨在通过语音-语言基础模型克服这些限制，使AI从被动工具转变为主动、情感化的日常伙伴。

## Method

*   **核心创新:** 提出 `Voila` 模型家族，包括 `Voila-e2e`（端到端语音对话）和 `Voila-autonomous`（全双工自主交互），采用端到端架构，摒弃传统管道系统，结合层次化多尺度Transformer实现低延迟、自然的语音交互。
*   **语音分词器（Voice Tokenizer）:** 将连续音频信号转化为离散的语义和声学令牌，通过改进的残差向量量化（RVQ）方法，第一层令牌聚焦语义信息，其余层捕捉声学细节（如音色、语调），便于LLM处理音频数据。
*   **文本-音频对齐（Text-Audio Alignment）:** 通过多任务训练（包括ASR、TTS和指令跟随）和交错对齐策略，确保文本和音频令牌在语义单元上的紧密对应，提升生成语音的表达性和同步性，优于现有松散耦合方法。
*   **层次化多尺度Transformer:** 包括语音-语言LLM骨干（处理语义信息，继承预训练LLM的推理能力）和音频Transformer（生成声学令牌），实现语义与声学的分离建模和高效整合。
*   **全双工设计:** `Voila-autonomous` 通过双流输入（用户音频流和自身音频流）实现同时听、说和推理，支持实时自主交互，如打断、回馈等自然对话行为。
*   **语音定制:** 引入语音嵌入（Voice Embedding）机制，支持从短至10秒的音频样本快速定制新语音，并预构建超过百万种语音，结合文本指令定义角色特性，实现个性化交互。
*   **统一模型:** 支持多种语音任务（ASR、TTS、对话）而无需任务特定调整，训练于多语言数据，支持英语、汉语等六种语言。

## Experiment

*   **有效性:** 在 `Voila Benchmark` 上，`Voila` 准确率达30.56%，显著优于 SpeechGPT (13.29%) 和 Moshi (11.45%)，尤其在数学和编程领域表现突出，显示其文本-音频对齐策略有效利用了LLM推理能力。
*   **ASR与TTS性能:** 在 LibriSpeech test-clean 数据集上，ASR词错误率（WER）为4.8%（未用训练数据）和2.7%（使用训练数据），与最先进模型相当；TTS WER 为3.2%和2.8%，优于 Moshi (4.7%) 和 Vall-E (5.9%)，语音生成质量更高。
*   **延迟表现:** 响应时间低至195毫秒，低于人类平均响应时间，远优于传统管道系统（数秒），实现实时交互。
*   **实验设置合理性:** 构建 `Voila Benchmark` 覆盖多领域任务，结合标准数据集（如 LibriSpeech）测试ASR和TTS，与开源模型对比，实验设计全面；但未深入探讨复杂情感交互或多语言环境表现，缺乏大规模用户测试，实际应用效果有待验证。

## Further Thoughts

1. 语音与语言的深度融合启发我们在其他多模态任务（如视觉-语言）中探索高效对齐机制，确保信息完整性。
2. 全双工自主交互理念提示AI可从被动响应转向主动参与，未来可基于用户情感和上下文设计更复杂的主动行为。
3. 百万语音定制机制表明个性化是AI发展方向，可结合用户行为数据动态调整交互风格，提升体验。
4. 统一模型支持多任务的思路启发设计通用AI架构，减少任务特定训练成本，适应多样化场景。