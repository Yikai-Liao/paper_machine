---
title: "VITA-Audio: Fast Interleaved Cross-Modal Token Generation for Efficient Large Speech-Language Model"
pubDatetime: 2025-05-06T17:59:53+00:00
slug: "2025-05-vita-audio-realtime"
type: "arxiv"
id: "2505.03739"
score: 0.5698956615368417
author: "grok-3-latest"
authors: ["Zuwei Long", "Yunhang Shen", "Chaoyou Fu", "Heting Gao", "Lijiang Li", "Peixian Chen", "Mengdan Zhang", "Hang Shao", "Jian Li", "Jinlong Peng", "Haoyu Cao", "Ke Li", "Rongrong Ji", "Xing Sun"]
tags: ["LLM", "Cross-Modal Mapping", "Speech Generation", "Token Prediction", "Real-Time Interaction"]
institution: ["Tencent Youtu Lab", "Nanjing University", "Xiamen University"]
description: "VITA-Audio通过轻量级MCTP模块和四阶段训练策略，在端到端语音模型中实现首次音频token零延迟生成，显著提升推理速度并在ASR、TTS、SQA任务上达到开源模型最优性能，为实时语音交互设定了新标准。"
---

> **Summary:** VITA-Audio通过轻量级MCTP模块和四阶段训练策略，在端到端语音模型中实现首次音频token零延迟生成，显著提升推理速度并在ASR、TTS、SQA任务上达到开源模型最优性能，为实时语音交互设定了新标准。 

> **Keywords:** LLM, Cross-Modal Mapping, Speech Generation, Token Prediction, Real-Time Interaction

**Authors:** Zuwei Long, Yunhang Shen, Chaoyou Fu, Heting Gao, Lijiang Li, Peixian Chen, Mengdan Zhang, Hang Shao, Jian Li, Jinlong Peng, Haoyu Cao, Ke Li, Rongrong Ji, Xing Sun

**Institution(s):** Tencent Youtu Lab, Nanjing University, Xiamen University


## Problem Background

随着语音交互成为人机交互的主要形式，实时语音系统因其自然性和副语言特征（如语调、情感）的丰富性而备受关注。
然而，传统级联架构（ASR+LLM+TTS）存在累积延迟、语义信息丢失和错误传播的问题，而现有端到端语音模型在流式生成场景中首次音频token生成的高延迟仍是一个显著瓶颈，限制了实时交互的实现。

## Method

*   **核心思想:** 提出VITA-Audio模型，通过轻量级的Multiple Cross-modal Token Prediction (MCTP)模块，利用LLM隐藏状态中的上下文信息，直接映射生成多个音频token，实现首次音频token的零延迟生成。
*   **模块设计:** MCTP模块基于文本与音频token的单调对齐特性，采用轻量级Transformer块结构，从LLM隐藏状态和文本嵌入中预测音频token。每个模块在单次前向传播中生成10个音频token，并通过级联预测架构（后续模块利用前模块的隐藏状态和输出序列）减少误差累积。
*   **训练策略:** 采用四阶段渐进式训练：1) 音频-文本对齐，通过大规模语音预训练扩展LLM的音频建模能力；2) 单个MCTP模块训练，预测单个后续token；3) 多个MCTP模块训练，扩展到多token预测；4) 监督微调，基于语音问答数据优化语音到语音对话能力，同时保持少量其他数据以稳定训练。
*   **推理模式:** 提供四种推理模式（Turbo, Boost, Balance, Vanilla），通过调整主模型与MCTP模块生成token的比例，适应不同速度和质量需求场景。
*   **关键优势:** 不依赖LLM的复杂语义建模，MCTP模块计算开销极低（单次前向仅占LLM骨干的11%时间），并通过共享嵌入层和输出头无缝集成到LLM自回归过程中。

## Experiment

*   **推理速度提升:** 在7B参数规模下，VITA-Audio-Turbo模式实现约5倍推理加速（从Vanilla模式的53.89秒减少到11.83秒，token生成速率从76提升到346每秒），在0.5B到72B模型规模上均保持类似加速效果。
*   **延迟降低:** 首次音频token生成时间从236ms（Vanilla模式）缩短到53ms（Turbo模式），显著减少感知延迟，满足实时交互需求。
*   **任务性能:** 在SQA任务中，VITA-Audio-Plus-Vanilla在S→T和S→S设置下均优于同规模开源模型，性能下降仅9%，表明文本-语音对齐质量高；在TTS任务中，VITA-Audio-Turbo在Seed-TTS和LibriTTS基准上接近或优于其他开源模型；在ASR任务中性能略逊但保持稳定。
*   **实验设置合理性:** 实验覆盖ASR、TTS、SQA等多任务，数据集包括多语言多领域开源数据，增强结果可信度和可复现性；不同推理模式的对比实验充分验证了方法在速度与质量权衡上的灵活性。

## Further Thoughts

VITA-Audio利用隐藏状态的上下文信息通过轻量级模块实现跨模态映射的思路非常具有启发性，这种方法不仅适用于语音生成，还可能扩展到其他多模态任务（如文本到图像或视频生成），只要模态间存在强对齐关系；此外，四阶段渐进式训练策略为复杂多任务模型优化提供了有效路径，值得借鉴；进一步思考，是否可以通过动态调整MCTP模块数量或结合自适应推理技术，根据任务复杂度和设备资源实现更高效的部署？