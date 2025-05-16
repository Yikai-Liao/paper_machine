---
title: "Flash-VL 2B: Optimizing Vision-Language Model Performance for Ultra-Low Latency and High Throughput"
pubDatetime: 2025-05-14T15:45:17+00:00
slug: "2025-05-flash-vl-optimization"
type: "arxiv"
id: "2505.09498"
score: 0.37994182493347545
author: "grok-3-latest"
authors: ["Bo Zhang", "Shuo Li", "Runhe Tian", "Yang Yang", "Jixin Tang", "Jinhao Zhou", "Lin Ma"]
tags: ["VLM", "Low Latency", "Token Compression", "Semantic Consistency", "Multimodal Training"]
institution: ["Meituan"]
description: "本文提出 Flash-VL 2B，通过创新的隐式语义拼接、令牌压缩和多阶段训练策略，在保持高性能的同时显著提升视觉-语言模型的推理速度和吞吐量。"
---

> **Summary:** 本文提出 Flash-VL 2B，通过创新的隐式语义拼接、令牌压缩和多阶段训练策略，在保持高性能的同时显著提升视觉-语言模型的推理速度和吞吐量。 

> **Keywords:** VLM, Low Latency, Token Compression, Semantic Consistency, Multimodal Training

**Authors:** Bo Zhang, Shuo Li, Runhe Tian, Yang Yang, Jixin Tang, Jinhao Zhou, Lin Ma

**Institution(s):** Meituan


## Problem Background

视觉-语言模型（VLMs）在多模态任务中表现出色，但高延迟和高计算需求限制了其在实时应用和资源受限环境中的部署。
论文旨在解决性能与效率之间的权衡问题，提出 Flash-VL 2B，目标是实现超低延迟和高吞吐量，同时保持竞争性的精度。

## Method

*   **架构设计**：采用 ViT-Adapter-LLM 架构，选择 SigLIP2-so400m-patch16-512 作为视觉编码器（固定分辨率，减少计算量），Qwen-2.5-1.5B-Instruct 作为轻量化语言模型，通过轻量级适配器连接两者。
*   **令牌压缩**：通过像素重排（Pixel Shuffling）将视觉令牌数量从 1024 个压缩至 256 个，有效降低计算负担，同时附加层归一化和线性层以对齐视觉与文本特征。
*   **图像处理创新**：提出隐式语义拼接（Implicit Semantic Stitching, ISS），通过提取图像瓦片边界共识特征，确保动态裁剪中的语义连续性，避免信息丢失，同时控制重叠率（约 12.5%）以减少重复信息对性能的负面影响。
*   **数据与训练策略**：基于高质量开源多模态数据集（如 InfinityMM），采用多阶段训练管道，包括预训练、微调和直接偏好优化（DPO），逐步提升模型能力，训练参数根据阶段动态调整（如学习率从 1e-3 衰减至 0）。

## Experiment

*   **性能表现**：Flash-VL 2B 在 11 个标准 VLM 基准测试上的平均性能达到 64.8%（Flash-VL-2B d-ISS），超越同规模模型 InternVL2.5-2B（63.6%），尤其在 MathVista 和 OCRBench 上分别提升 2.1% 和 43 点。
*   **速度提升**：Flash-VL-2B s 吞吐量达到 60.73 令牌/秒，显著高于 Qwen2-VL-2B（39.07 令牌/秒），在时间到首令牌（TTFT）和每输出令牌时间（TPOT）上也表现出最佳平均延迟。
*   **消融研究**：ISS 策略带来显著改进（平均提升 0.8%），DPO 优化在多任务上提升明显（如 OCRBench 从 696 到 764）；视觉编码器和适配器设计的对比实验验证了 SigLIP2 和像素重排适配器的优越性。
*   **实验设置**：实验覆盖静态和动态分辨率场景，使用 11 个多模态基准测试，数据量和任务类型较为全面，但动态分辨率在 SigLIP2 上表现不佳，原因未深入探讨。

## Further Thoughts

隐式语义拼接（ISS）通过边界特征提取保持语义一致性的思路，可推广至视频或多图像任务中，通过时间或空间维度的语义连接进一步优化效率；此外，轻量化模型结合多阶段训练和偏好优化的策略，启发我们探索知识蒸馏或动态训练目标调整，以在特定任务上进一步提升小规模模型性能。