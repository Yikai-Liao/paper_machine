---
title: "Lightweight End-to-end Text-to-speech Synthesis for low resource on-device applications"
pubDatetime: 2025-05-12T16:10:15+00:00
slug: "2025-05-lightweight-e2e-tts"
type: "arxiv"
id: "2505.07701"
score: 0.5600033442010555
author: "grok-3-latest"
authors: ["Biel Tura Vecino", "Adam Gabryś", "Daniel Mątwicki", "Andrzej Pomirski", "Tom Iddon", "Marius Cotescu", "Jaime Lorenzo-Trueba"]
tags: ["TTS", "End-to-End", "Lightweight Model", "Speech Synthesis", "Low Resource"]
institution: ["Alexa AI"]
description: "本文提出了一种轻量级端到端文本转语音模型（LE2E），通过联合训练声学模型和声码器，在低资源设备上实现了高质量实时语音合成，参数量减少90%且速度提升10倍。"
---

> **Summary:** 本文提出了一种轻量级端到端文本转语音模型（LE2E），通过联合训练声学模型和声码器，在低资源设备上实现了高质量实时语音合成，参数量减少90%且速度提升10倍。 

> **Keywords:** TTS, End-to-End, Lightweight Model, Speech Synthesis, Low Resource

**Authors:** Biel Tura Vecino, Adam Gabryś, Daniel Mątwicki, Andrzej Pomirski, Tom Iddon, Marius Cotescu, Jaime Lorenzo-Trueba

**Institution(s):** Alexa AI


## Problem Background

随着智能设备和物联网的普及，离线、设备端实时文本转语音（TTS）系统的需求日益增加，特别是在无网络连接或隐私敏感场景下。
然而，当前端到端（E2E）TTS模型计算复杂、内存占用大，不适合低资源环境，而传统两阶段方法（声学模型和声码器分开训练）存在特征不匹配问题，导致合成质量下降。
论文旨在开发一种轻量级、高质量的E2E-TTS模型，解决低资源设备上的实时语音合成难题。

## Method

*   **核心思想:** 提出轻量级端到端TTS模型（LE2E），通过联合训练声学模型和声码器，避免传统两阶段方法的特征不匹配问题，同时减少计算和内存需求，适合设备端应用。
*   **模型架构:** 
    *   **生成器（Generator）:** 包括声学潜在模型（基于LightSpeech）和声码器（基于Multi-Band MelGAN）。声学模型将文本输入（音素和位置嵌入）转化为潜在声学表示，通过文本编码器、方差适配器（预测时长和音高）和声学解码器完成处理；声码器则将潜在表示上采样为波形信号，使用伪正交镜像滤波器（PQMF）合成最终波形。
    *   **判别器（Discriminators）:** 采用BigVGAN提出的多周期判别器（MPD）和多分辨率判别器（MRD），通过不同分辨率和周期结构的分析，指导生成器合成更自然的波形，减少感知伪影。
*   **训练目标:** 结合多种损失函数优化模型，包括：
    *   **时长损失（Duration Loss）:** 使用均方误差（MSE）优化音素时长预测。
    *   **音高损失（Pitch Loss）:** 通过交叉熵损失进行音高密度建模，而非传统回归任务，提升预测稳定性。
    *   **对抗损失（GAN Loss）:** 采用最小二乘损失进行对抗训练，提升生成波形的真实性。
    *   **特征匹配损失（Feature Matching Loss）:** 计算判别器中间特征的L1距离，增强生成与目标的相似性。
    *   **重建损失（Reconstruction Losses）:** 包括多分辨率短时傅里叶变换（STFT）损失和mel频谱图损失，进一步提升波形质量。
*   **关键创新:** 端到端联合训练简化了流程，消除了两阶段训练的复杂性；轻量化设计使模型参数量减少90%，实时因子提升10倍；改进的损失函数和判别器设计进一步提升了语音质量。

## Experiment

*   **数据集与设置:** 在LJSpeech数据集（约24小时单说话人英语语音）上进行实验，数据集分为训练（12,900样本）、验证和测试（各100样本），评估指标包括主观均值意见分数（MOS）和客观指标（如条件Fréchet语音距离cFSD、音高误差F0 RMSE、说话人嵌入相似度XWLM）。
*   **效果对比:** LE2E模型MOS为3.79，与VITS（3.81）相当，略低于JETS（4.01），但参数量仅为3.71M（比JETS减少90%），实时因子（RTF）为0.0084（比JETS快10倍）。
*   **方法提升:** 相比同架构的两阶段方法（LightSpeech + MB-MelGAN cascade），LE2E在MOS（3.79 vs 3.73）和客观指标上均有提升，同时简化了训练流程（无需额外微调）。
*   **合理性与局限:** 实验设置全面，指标多样，结果可信；但实验仅限于单说话人、单语言场景，缺乏多场景验证，可能影响泛化性结论。

## Further Thoughts

端到端联合训练的思路启发了我对减少中间表示（如mel-spectrogram）在其他语音任务（如语音识别或转换）中应用的思考，是否能同样提升效率和质量？
此外，轻量化设计在低资源设备上的成功应用让我联想到动态模型裁剪策略，例如根据设备性能在推理时调整模型复杂度，以进一步优化用户体验。
最后，改进的判别器和损失函数设计对生成质量的提升显著，这提示在其他生成任务中可以探索更复杂的对抗训练策略，甚至结合跨模态数据（如文本与语音的联合建模）来增强模型性能。