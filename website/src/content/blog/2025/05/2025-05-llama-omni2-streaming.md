---
title: "LLaMA-Omni2: LLM-based Real-time Spoken Chatbot with Autoregressive Streaming Speech Synthesis"
pubDatetime: 2025-05-05T12:53:09+00:00
slug: "2025-05-llama-omni2-streaming"
type: "arxiv"
id: "2505.02625"
score: 0.7340302809807175
author: "grok-3-latest"
authors: ["Qingkai Fang", "Yan Zhou", "Shoutao Guo", "Shaolei Zhang", "Yang Feng"]
tags: ["LLM", "Speech Synthesis", "Streaming Generation", "Modular Architecture", "Real-Time Interaction"]
institution: ["Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences (ICT/CAS)", "Key Laboratory of AI Safety, Chinese Academy of Sciences", "University of Chinese Academy of Sciences, Beijing, China"]
description: "LLaMA-Omni 2 通过模块化设计和自回归流式语音生成技术，在低成本训练下实现了实时、高质量的语音交互，显著优于现有 SpeechLM 模型。"
---

> **Summary:** LLaMA-Omni 2 通过模块化设计和自回归流式语音生成技术，在低成本训练下实现了实时、高质量的语音交互，显著优于现有 SpeechLM 模型。 

> **Keywords:** LLM, Speech Synthesis, Streaming Generation, Modular Architecture, Real-Time Interaction

**Authors:** Qingkai Fang, Yan Zhou, Shoutao Guo, Shaolei Zhang, Yang Feng

**Institution(s):** Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences (ICT/CAS), Key Laboratory of AI Safety, Chinese Academy of Sciences, University of Chinese Academy of Sciences, Beijing, China


## Problem Background

语音作为人机交互的重要接口，能够显著提升交互效率和用户体验，但传统级联系统（ASR + LLM + TTS）存在错误累积、响应延迟高和难以捕捉副语言信息等问题。
端到端语音语言模型（SpeechLMs）虽被提出以解决这些问题，但原生 SpeechLMs 需要大规模语音数据预训练，成本高且可能遗忘文本能力，而模块化 SpeechLMs 在实时性和语音自然性上仍有不足。
本文旨在通过模块化设计和流式语音生成技术，平衡实时性、自然性和智能性，实现高质量的实时语音交互。

## Method

*   **核心架构:** LLaMA-Omni 2 基于 Qwen2.5 系列模型（参数规模 0.5B 到 14B），采用模块化 SpeechLM 设计，集成了语音编码器和自回归流式语音解码器。
*   **语音理解模块:** 使用 Whisper-large-v3 编码器将输入语音转换为表征序列，通过适配器（包括降采样模块和前馈网络）将表征输入到 LLM，确保语音信息有效传递。
*   **流式语音生成模块:** 
    *   采用自回归文本转语音语言模型（基于 Qwen2.5-0.5B 初始化），通过扩展词汇表支持语音 token 生成。
    *   引入门控融合机制，结合 LLM 的隐藏状态和文本嵌入，确保生成的语音 token 既考虑上下文信息又与文本内容对齐。
    *   使用‘读-写’策略（Read-R-Write-W），按预定比例（如 R=3, W=10）交替读取 LLM 输出和生成语音 token，实现文本与语音同步流式生成。
    *   最终通过因果流匹配模型（受 CosyVoice 2 启发）和 HiFi-GAN 声码器将语音 token 转为波形。
*   **训练策略:** 采用两阶段训练，仅使用 200K 多轮语音对话数据：第一阶段分别训练语音到文本和文本到语音模块，第二阶段训练语音到语音生成能力，冻结部分模块以降低成本。
*   **关键创新:** 通过流式生成和模块化设计，避免大规模预训练需求，同时显著降低响应延迟并提升语音自然性。

## Experiment

*   **有效性:** LLaMA-Omni 2 在语音问答（SpokenQA）和语音指令跟随任务上显著优于基线模型（如 GLM-4-Voice 和 LLaMA-Omni），例如在 Web Questions 数据集上，LLaMA-Omni2-7B 的语音到语音（S2S）准确率仅下降 3.2%（相比 S2T），而 GLM-4-Voice 下降 16.3%，表明其语音生成能力更强。
*   **模型规模影响:** 随着 LLM 参数规模从 0.5B 到 14B 增加，性能持续提升，LLaMA-Omni2-14B 在所有任务上表现最佳，证明方法能有效利用 LLM 能力。
*   **延迟与质量:** 响应延迟约 600ms，满足实时交互需求；语音自然性（UTMOS 评分）显著优于基线，得益于 CosyVoice 2 的流匹配模型。
*   **消融实验:** 验证了门控融合模块、TTS 预训练和读写策略的重要性，例如 R=3, W=10 的设置在性能、延迟和语音质量间取得最佳平衡。
*   **实验设置合理性:** 实验涵盖不同模型规模、数据量和架构设计的对比，评估了 S2T 和 S2S 两种场景，数据可信且支持结论。

## Further Thoughts

模块化 SpeechLM 的低成本训练策略（仅 200K 数据）启发我们探索更高效的数据合成或迁移学习方法，以进一步降低资源需求；
流式语音生成的‘读-写’策略可扩展到其他多模态任务（如视频与文本同步生成），实现更广义的实时交互；
未来引入条件生成机制（如基于输入语音的情感标签）可能增强语音风格表达力，值得进一步研究。