---
title: "LLaMA-Omni2: LLM-based Real-time Spoken Chatbot with Autoregressive Streaming Speech Synthesis"
pubDatetime: 2025-05-05T12:53:09+00:00
slug: "2025-05-llama-omni2-streaming"
type: "arxiv"
id: "2505.02625"
score: 0.7701593910334862
author: "grok-3-latest"
authors: ["Qingkai Fang", "Yan Zhou", "Shoutao Guo", "Shaolei Zhang", "Yang Feng"]
tags: ["LLM", "Speech Synthesis", "Streaming Generation", "Modular Design", "Real-Time Interaction"]
institution: ["Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences (ICT/CAS)", "Key Laboratory of AI Safety, Chinese Academy of Sciences", "University of Chinese Academy of Sciences, Beijing, China"]
description: "LLaMA-Omni 2 通过模块化设计和自回归流式语音生成技术，显著提升了实时语音交互的智能性、自然性和低延迟表现，超越了现有 SpeechLM 模型。"
---

> **Summary:** LLaMA-Omni 2 通过模块化设计和自回归流式语音生成技术，显著提升了实时语音交互的智能性、自然性和低延迟表现，超越了现有 SpeechLM 模型。 

> **Keywords:** LLM, Speech Synthesis, Streaming Generation, Modular Design, Real-Time Interaction

**Authors:** Qingkai Fang, Yan Zhou, Shoutao Guo, Shaolei Zhang, Yang Feng

**Institution(s):** Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences (ICT/CAS), Key Laboratory of AI Safety, Chinese Academy of Sciences, University of Chinese Academy of Sciences, Beijing, China


## Problem Background

语音作为人机交互的重要接口，能够显著提升交互效率和用户体验，但传统级联系统（ASR + LLM + TTS）存在错误累积、响应延迟高和难以捕捉副语言信息等问题。
端到端语音语言模型（SpeechLMs）虽被提出以解决这些问题，但原生 SpeechLMs 需大规模数据预训练，成本高且可能遗忘文本能力，而模块化 SpeechLMs 在实时性和语音自然性上仍有不足。
本文旨在通过模块化设计和高效训练策略，构建一个低延迟、高质量的实时语音交互系统。

## Method

*   **核心架构:** LLaMA-Omni 2 基于 Qwen2.5 系列模型（参数规模 0.5B 至 14B），采用模块化 SpeechLM 设计，集成语音编码器和自回归流式语音解码器，实现端到端语音交互。
*   **语音理解模块:** 使用 Whisper-large-v3 编码器将输入语音转换为表征序列，通过适配器（包括降采样模块和前馈网络）将表征映射到 LLM 输入空间，确保语音信息有效传递。
*   **流式语音生成模块:** 包括两个核心组件：
    *   自回归文本转语音语言模型（基于 Qwen2.5-0.5B 初始化），将 LLM 输出（文本 token 和隐藏状态）转换为语音 token，采用‘读-写’策略（Read-R, Write-W）实现流式生成，即每读取 R 个文本 token，生成 W 个语音 token。
    *   因果流匹配模型（受 CosyVoice 2 启发），将语音 token 转为梅尔频谱图，并通过 HiFi-GAN 声码器生成最终波形，支持分块流式合成。
*   **门控融合机制:** 在文本转语音过程中，通过门控机制自适应融合 LLM 的隐藏状态和文本嵌入，确保语音生成与文本内容一致，同时保留上下文信息。
*   **训练策略:** 采用两阶段训练，仅使用 200K 多轮语音对话数据：
    *   第一阶段分别训练语音到文本和文本到语音模块。
    *   第二阶段训练语音到语音生成能力，冻结部分模块以降低计算成本。
*   **关键创新:** 通过‘读-写’策略和流式合成技术，实现文本与语音同步生成，有效控制响应延迟，同时保持语音自然性和智能性。

## Experiment

*   **有效性:** 在语音问答（SpokenQA）和语音指令跟随任务中，LLaMA-Omni 2 显著优于基线模型，如 LLaMA-Omni2-7B 在 Llama Questions 数据集上的准确率（S2T: 70.3%, S2S: 60.7%）高于 GLM-4-Voice（S2T: 64.7%, S2S: 50.7%），且 S2T 与 S2S 性能差距较小，显示出语音生成能力的提升。
*   **模型扩展性:** 随着 LLM 参数规模从 0.5B 到 14B 增加，性能持续提升，LLaMA-Omni2-14B 在所有任务中表现最佳，表明方法对模型规模的适应性强。
*   **延迟与自然性:** 响应延迟控制在约 600ms（LLaMA-Omni2-7B 为 582.91ms），远低于 GLM-4-Voice（1562.81ms），满足实时交互需求；语音自然性（UTMOS 评分）也显著优于基线，得益于流式合成和高质量声码器。
*   **消融实验:** 验证了门控融合模块、TTS 预训练和读写策略的重要性，例如移除门控融合后 ASR-WER 从 3.26 升至 4.89，表明各组件对性能的贡献。
*   **实验设置合理性:** 实验涵盖了不同模型规模、训练数据量和超参数（如 R 和 W）的对比，设置较为全面，但未充分探讨极端低资源设备上的表现，可能存在一定局限性。

## Further Thoughts

模块化 SpeechLM 的高效训练策略令人印象深刻，仅用 200K 数据即可超越依赖大规模数据的模型，这提示我们可以在数据受限场景下通过优化模块设计和训练流程实现高性能。
流式语音生成的‘读-写’策略提供了一种平衡延迟与质量的新思路，可推广至其他实时多模态任务，如视频生成或实时翻译。
门控融合机制的自适应融合方法，或许能应用于其他跨模态任务，提升模型对上下文和内容的综合理解能力。
未来可探索基于用户语音风格（如情感、语速）的自适应生成，通过引入多样化数据或条件控制机制，进一步提升交互的个性化体验。