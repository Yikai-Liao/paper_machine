---
title: "XY-Tokenizer: Mitigating the Semantic-Acoustic Conflict in Low-Bitrate Speech Codecs"
pubDatetime: 2025-06-29T16:51:50+00:00
slug: "2025-06-xy-tokenizer-conflict-mitigation"
type: "arxiv"
id: "2506.23325"
score: 0.6188688998841849
author: "grok-3-latest"
authors: ["Yitian Gong", "Luozhijie Jin", "Ruifan Deng", "Dong Zhang", "Xin Zhang", "Qinyuan Cheng", "Zhaoye Fei", "Shimin Li", "Xipeng Qiu"]
tags: ["LLM", "Speech Codec", "Semantic Alignment", "Acoustic Reconstruction", "Multi-Task Learning"]
institution: ["Fudan University", "Shanghai Innovation Institute"]
description: "本文提出 XY-Tokenizer，一种低比特率语音编解码器，通过双塔架构和多阶段多任务学习缓解语义-声学冲突，在 1kbps 下实现与最先进模型相当的语义对齐和音频重建性能。"
---

> **Summary:** 本文提出 XY-Tokenizer，一种低比特率语音编解码器，通过双塔架构和多阶段多任务学习缓解语义-声学冲突，在 1kbps 下实现与最先进模型相当的语义对齐和音频重建性能。 

> **Keywords:** LLM, Speech Codec, Semantic Alignment, Acoustic Reconstruction, Multi-Task Learning

**Authors:** Yitian Gong, Luozhijie Jin, Ruifan Deng, Dong Zhang, Xin Zhang, Qinyuan Cheng, Zhaoye Fei, Shimin Li, Xipeng Qiu

**Institution(s):** Fudan University, Shanghai Innovation Institute


## Problem Background

语音编解码器在语音大语言模型（Speech LLMs）中起到关键作用，需在低比特率下同时保留语义信息和声学信息以支持文本对齐和高保真音频重建。然而，现有编解码器在语义对齐和声学保真之间存在冲突，尤其在低比特率（如 1kbps）场景下表现明显，论文旨在设计一种新型编解码器以平衡这两者。

## Method

* **核心思想**：提出 XY-Tokenizer，一种低比特率语音编解码器，通过双塔架构和多阶段多任务学习缓解语义-声学冲突。
* **架构设计**：
  - **编码器**：包含语义通道和声学通道两个并行分支，分别处理 Mel 频谱图输入，语义通道基于预训练 Whisper 模型（参数固定）提取语言特征，声学通道（参数可训练）捕获副语言信息，两者输出拼接后进一步处理。
  - **量化器**：采用残差向量量化（RVQ），8 层，每层码本大小 1024，时间分辨率 12.5Hz，总比特率 1kbps。
  - **解码器**：同样分为语义和声学分支，语义解码器基于 Qwen2.5 的 LLM 生成文本转录，声学解码器通过 Vocos 模型重建 16kHz 音频波形。
* **训练策略**：
  - **预训练阶段**：通过多任务学习同时优化语义对齐（基于 LLM 的自动语音识别任务，使用交叉熵损失）和粗粒度声学重建（多尺度 Mel 频谱图重建损失），并加入量化承诺损失确保量化有效性。
  - **后训练阶段**：采用生成对抗网络（GAN）框架，固定编码器和量化器，丢弃语义解码器，引入多周期、多尺度判别器优化细粒度声学特征，提升感知质量，损失包括重建损失、特征匹配损失和对抗损失。
* **关键创新**：通过减少语义和声学任务间的共享参数（仅在 RVQ 模块共享）缓解任务冲突，利用预训练模型降低训练复杂度，分阶段优化确保性能平衡。

## Experiment

* **有效性**：XY-Tokenizer 在 1kbps 比特率下，语义性能（WER 0.13）接近 Baichuan Audio Tokenizer（0.10），优于 SpeechTokenizer（0.34）和 Mimi-8（0.28）；声学性能（SIM 0.83）接近 BigCodec（0.84），优于大多数低比特率基线，表明其在两任务上均表现优异。
* **综合分析**：相比专注于单一任务的模型（如 BigCodec 仅优化声学），XY-Tokenizer 实现了更好的语义-声学平衡，优于基于语义蒸馏的模型（如 SpeechTokenizer），验证了其缓解任务冲突的能力。
* **实验设置**：使用 Emilia 数据集（101k 小时音频）训练，评估在 LibriSpeech 上进行，指标全面（WER, SIM, STOI, PESQ），基线选择合理，覆盖语义和声学模型；消融实验验证了减少共享参数和固定 LLM 的重要性；不足之处在于未测试更低比特率或多语言/噪声环境下的泛化性。
* **计算开销**：预训练阶段使用 32 个 H100 GPU，50 万步，后训练阶段单 GPU 25 万步，显示较高计算需求，但与同类研究相当。

## Further Thoughts

XY-Tokenizer 的双塔架构和减少共享参数的思路启发我们可以在其他多模态任务中应用类似策略以缓解任务冲突；多阶段训练（先粗粒度后细粒度优化）对复杂模型训练有借鉴意义；此外，利用预训练模型初始化组件的做法可推广到其他资源受限场景。发散性思考：是否可以通过动态调整任务权重或自适应比特率分配机制，根据应用需求（如语音识别或音频合成优先）进一步优化性能？