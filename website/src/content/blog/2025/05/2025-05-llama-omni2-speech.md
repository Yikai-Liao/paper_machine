---
title: "LLaMA-Omni2: LLM-based Real-time Spoken Chatbot with Autoregressive Streaming Speech Synthesis"
pubDatetime: 2025-05-05T12:53:09+00:00
slug: "2025-05-llama-omni2-speech"
type: "arxiv"
id: "2505.02625"
score: 0.7340302809807175
author: "grok-3-latest"
authors: ["Qingkai Fang", "Yan Zhou", "Shoutao Guo", "Shaolei Zhang", "Yang Feng"]
tags: ["LLM", "Speech Synthesis", "Streaming Generation", "Modular Design", "Real-time Interaction"]
institution: ["Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences", "Key Laboratory of AI Safety, Chinese Academy of Sciences", "University of Chinese Academy of Sciences"]
description: "LLaMA-Omni 2 通过模块化设计和自回归流式语音生成，以较低成本实现高质量端到端语音交互，显著超越依赖大规模数据的基线模型。"
---

> **Summary:** LLaMA-Omni 2 通过模块化设计和自回归流式语音生成，以较低成本实现高质量端到端语音交互，显著超越依赖大规模数据的基线模型。 

> **Keywords:** LLM, Speech Synthesis, Streaming Generation, Modular Design, Real-time Interaction

**Authors:** Qingkai Fang, Yan Zhou, Shoutao Guo, Shaolei Zhang, Yang Feng

**Institution(s):** Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences, Key Laboratory of AI Safety, Chinese Academy of Sciences, University of Chinese Academy of Sciences


## Problem Background

传统的语音聊天机器人依赖级联管道（自动语音识别、语言模型和文本转语音），存在误差累积、响应延迟高和难以捕捉副语言信息的问题。
论文旨在通过端到端的语音语言模型（SpeechLM），以较低成本实现实时、智能和自然的语音交互，解决传统方法的局限性。

## Method

*   **核心架构:** LLaMA-Omni 2 是一种模块化语音语言模型，基于 Qwen2.5 系列模型（参数规模从 0.5B 到 14B），通过集成语音编码器和自回归流式语音解码器实现端到端语音交互。
*   **语音理解模块:** 采用 Whisper-large-v3 编码器将输入语音转换为表示序列，并通过包含降采样模块和前馈网络的适配器将表示输入到语言模型中，确保语义信息的有效传递。
*   **流式语音生成:** 受 CosyVoice 2 启发，使用自回归文本转语音语言模型（基于 Qwen2.5-0.5B 初始化）生成语音 token，通过‘读-写’（Read-Write）策略实现流式生成，即在语言模型输出文本的同时生成语音 token；随后，语音 token 通过块感知因果流匹配模型转换为 mel 频谱图，并由 HiFi-GAN 声码器生成最终波形。
*   **训练策略:** 采用两阶段训练，第一阶段分别优化语音到文本和文本到语音模块，第二阶段训练语音到语音生成能力，重点优化门控融合模块以结合语言模型的隐藏状态和文本嵌入，提高语音生成的上下文一致性；训练仅使用 200K 多轮语音对话样本，显著降低数据需求。
*   **创新点:** 模块化设计充分利用现有语言模型能力，通过小规模微调实现语音交互；流式生成策略通过调整读写比例平衡延迟和语音质量。

## Experiment

*   **性能表现:** LLaMA-Omni 2 在语音问答（SpokenQA）和语音指令跟随任务上表现出色，例如 LLaMA-Omni2-7B 在 Web Questions 数据集上的准确率（S2T: 34.5%, S2S: 31.3%）显著优于 GLM-4-Voice（S2T: 32.2%, S2S: 15.9%），且 S2T 到 S2S 的性能下降较小，显示出强大的语音生成能力。
*   **延迟与自然性:** 响应延迟约为 600ms（LLaMA-Omni2-7B 为 582.91ms），满足实时交互需求，且远低于 GLM-4-Voice（1562.81ms）；语音自然性（UTMOS 评分）显著优于基线，得益于 CosyVoice 2 的流匹配模型。
*   **数据效率:** 仅用 200K 样本训练即可超越依赖数百万小时数据的 GLM-4-Voice，显示出极高的数据效率；消融研究表明多轮对话数据比单轮数据更有效，且 200K 样本已接近性能饱和。
*   **实验设置合理性:** 实验覆盖了多个参数规模（0.5B 到 14B），并在不同数据集上测试了 S2T 和 S2S 性能，设置全面；消融研究验证了门控融合模块、TTS 预训练和读写策略的必要性，例如门控融合模块显著降低 ASR-WER（从 4.89 降至 3.26）。

## Further Thoughts

模块化 SpeechLM 的设计启发我们可以在其他预训练模型上添加新模态模块，探索跨模态能力的低成本扩展；流式生成的读写策略可推广到其他实时任务，为延迟与质量的权衡提供新思路；少量多轮对话数据的高效利用提示我们在资源受限场景中注重数据上下文性和多样性，可能比单纯增加数据量更有效。