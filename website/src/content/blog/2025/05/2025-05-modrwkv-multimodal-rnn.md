---
title: "ModRWKV: Transformer Multimodality in Linear Time"
pubDatetime: 2025-05-20T15:34:36+00:00
slug: "2025-05-modrwkv-multimodal-rnn"
type: "arxiv"
id: "2505.14505"
score: 0.5553319460897546
author: "grok-3-latest"
authors: ["Jiale Kang", "Ziyin Yue", "Qingyu Yin", "Jiang Rui", "Weile Li", "Zening Lu", "Zhouran Ji"]
tags: ["Multimodal Learning", "RNN Architecture", "Linear Complexity", "Encoder Design", "Cross-Modal Fusion"]
institution: ["RWKVOS", "Zhejiang University", "The Hong Kong University of Science and Technology"]
description: "本文提出 ModRWKV 框架，基于 RWKV7 的 RNN 架构，通过可切换的模态编码器实现多模态理解，在性能和计算效率之间取得平衡，展现了线性模型在多模态大语言模型领域的潜力。"
---

> **Summary:** 本文提出 ModRWKV 框架，基于 RWKV7 的 RNN 架构，通过可切换的模态编码器实现多模态理解，在性能和计算效率之间取得平衡，展现了线性模型在多模态大语言模型领域的潜力。 

> **Keywords:** Multimodal Learning, RNN Architecture, Linear Complexity, Encoder Design, Cross-Modal Fusion

**Authors:** Jiale Kang, Ziyin Yue, Qingyu Yin, Jiang Rui, Weile Li, Zening Lu, Zhouran Ji

**Institution(s):** RWKVOS, Zhejiang University, The Hong Kong University of Science and Technology


## Problem Background

当前多模态研究主要基于具有二次复杂度的 Transformer 架构的大型语言模型（LLMs），而线性复杂度模型（如 RNNs）由于推理成本低具有潜力，但其应用多局限于纯文本领域，存在关键空白：如何将线性模型扩展到多模态大语言模型（MLLMs）领域，以在性能和计算效率之间取得平衡。

## Method

* **核心思想**：提出 ModRWKV 框架，基于 RWKV7（一种现代 RNN 架构）作为语言模型主干，通过动态适应的异构模态编码器实现多源信息融合，探索线性模型在多模态任务中的潜力。
* **多模态编码器设计**：为不同模态选择合适的编码器，例如视觉编码器使用 CLIP 和 SigLIP2，音频编码器使用 WavLM 和 Whisper，时间序列编码器使用 WaveNet 和 Timer，将原始数据转化为特征嵌入，与 RWKV7 对齐。
* **适配器设计**：通过轻量级适配器（如单层 MLP）实现模态间的维度对齐，减少参数量，迫使 RWKV7 主干承担大部分跨模态推理任务。
* **序列压缩**：针对多模态数据生成的长序列问题，采用 1D 卷积（Conv1D）进行序列压缩，显著降低计算开销，同时尽量保持性能。
* **训练范式**：采用分阶段训练策略，第一阶段冻结编码器和 RWKV 模型，仅训练适配器；第二阶段解冻适配器和 RWKV 参数进行联合训练；利用 RWKV7 预训练权重初始化，加速多模态训练。

## Experiment

* **有效性**：ModRWKV 在视觉任务（如 VQA-v2、ScienceQA、MMMU）中表现出色，3B 参数模型优于同规模 VL-Mamba-2.8B，甚至在某些任务上与更大的 LLaVA-1.5-7B 相当或更优，例如 MMMU 得分达 38.7%；在音频任务中，LibriSpeech WER 低至 2.43%，Aishell-1 CER 低至 5.08%；在时间序列任务中，适配器缩放因子为 4x 时 MSE 指标在多个数据集上最优。
* **实验设置合理性**：实验覆盖视觉、音频、时间序列等多模态任务，评估了不同编码器、模型规模（0.4B 到 3B）和消融研究（如编码器选择、序列压缩效果），例如 SigLIP2 编码器显著优于 CLIP。
* **计算效率**：通过 1D 卷积压缩序列长度，性能略降的同时显著提升推理速度，例如序列压缩 50% 时 ScienceQA 准确率提升 4.6%。
* **总结**：ModRWKV 在性能和效率之间取得较好平衡，实验数据支持其作为 Transformer 替代方案的可行性。

## Further Thoughts

线性复杂度模型（如 RNNs）在多模态任务中的潜力挑战了 Transformer 主导范式，启发思考是否可在更多场景利用 RNN 低成本优势；预训练权重选择（如 g1 模型在 ScienceQA 提升 28%）提示针对多模态任务设计特定预训练策略的重要性；适配器缩放因子和序列压缩策略对性能-效率权衡的探索，是否可进一步动态调整以适应不同任务需求？