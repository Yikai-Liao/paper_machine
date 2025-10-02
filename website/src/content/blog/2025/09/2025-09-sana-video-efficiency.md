---
title: "SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer"
pubDatetime: 2025-09-29T12:28:09+00:00
slug: "2025-09-sana-video-efficiency"
type: "arxiv"
id: "2509.24695"
score: 0.7312388685150549
author: "grok-3-latest"
authors: ["Junsong Chen", "Yuyang Zhao", "Jincheng Yu", "Ruihang Chu", "Junyu Chen", "Shuai Yang", "Xianbang Wang", "Yicheng Pan", "Daquan Zhou", "Huan Ling", "Haozhe Liu", "Hongwei Yi", "Hao Zhang", "Muyang Li", "Yukang Chen", "Han Cai", "Sanja Fidler", "Ping Luo", "Song Han", "Enze Xie"]
tags: ["Diffusion Model", "Linear Attention", "Video Generation", "Autoregressive Training", "Memory Efficiency"]
institution: ["NVIDIA", "Hong Kong University (HKU)", "Massachusetts Institute of Technology (MIT)", "Tsinghua University (THU)", "Peking University (PKU)", "King Abdullah University of Science and Technology (KAUST)"]
description: ""
---

> **Summary:**  

> **Keywords:** Diffusion Model, Linear Attention, Video Generation, Autoregressive Training, Memory Efficiency

**Authors:** Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, Haozhe Liu, Hongwei Yi, Hao Zhang, Muyang Li, Yukang Chen, Han Cai, Sanja Fidler, Ping Luo, Song Han, Enze Xie

**Institution(s):** NVIDIA, Hong Kong University (HKU), Massachusetts Institute of Technology (MIT), Tsinghua University (THU), Peking University (PKU), King Abdullah University of Science and Technology (KAUST)


## Problem Background

视频生成领域因高计算复杂性和内存限制面临挑战，尤其是在高分辨率（如720p）和长视频（超过10秒）生成中，传统模型训练成本高昂、生成速度慢，难以在云端和边缘设备上广泛应用。
论文旨在开发一个高效、高质量的视频生成器，解决计算效率和长视频生成难题，使高分辨率视频生成更具可访问性和实用性。

## Method

*   **核心思想:** 设计一个小型扩散模型 SANA-Video，通过线性注意力机制和固定内存 KV 缓存，实现高效的高分辨率和长视频生成，同时保持输出质量。
*   **Linear DiT（线性扩散变换器）:** 基于 SANA 架构，将传统自注意力机制替换为线性注意力，计算复杂度从 O(N²) 降至 O(N)，显著提升高分辨率视频生成效率；引入旋转位置编码（RoPE）增强长上下文建模能力，通过调整 RoPE 应用顺序（在 ReLU 激活后）确保训练稳定性；设计时空混合前馈网络（Mix-FFN），加入一维时间卷积以聚合时间特征，提升运动一致性。
*   **Block Linear Attention with Constant-Memory KV Cache:** 针对长视频生成，利用线性注意力的累积特性，设计块状自回归方法，将 KV 缓存简化为固定内存状态（仅存储注意力状态和键的累积和），避免内存随序列长度增长，支持分钟级长视频生成；提出两阶段自回归训练范式，包括单调递增 SNR 采样和改进的自强制（Self-Forcing）训练，减少曝光偏差，提升长视频质量。
*   **高效训练与数据过滤:** 从预训练文本到图像（T2I）模型继续训练，采用多阶段训练策略（从低分辨率到高分辨率），结合严格数据过滤（如基于运动质量和审美评分的筛选），大幅降低训练成本；设计深度压缩视频自编码器（DCAE-V），提升高分辨率视频生成效率。
*   **关键点:** 方法不依赖大规模模型参数，通过架构优化和训练策略创新实现效率与性能的平衡，支持边缘设备部署（如 RTX 5090 GPU）。

## Experiment

*   **有效性:** SANA-Video 在 H100 GPU 上生成5秒720p视频仅需36秒，比 Wan2.1-1.3B 快16倍，比 Wan2.1-14B 快53倍；在 RTX 5090 GPU 上通过 NVFP4 量化后，生成时间从71秒缩短至29秒，加速2.4倍。
*   **性能表现:** 在 VBench 基准测试中，文本到视频（T2V）总分为83.71，图像到视频（I2V）总分为88.02，与大型模型如 Open-Sora-2.0（14B）和 Wan2.1-14B 相当，尤其在语义一致性上表现最佳（T2V 为81.35，I2V 为96.40）；支持分钟级长视频生成，性能与 Self-Forcing 等方法相当或更优。
*   **实验设置合理性:** 实验覆盖 T2V、I2V 多种任务，测试不同分辨率（480p、720p）和视频长度（5秒至1分钟），设置全面；消融实验验证了线性注意力、RoPE、单调递增 SNR 采样等设计的贡献，如线性注意力在720p下实现4倍加速，RoPE 显著降低训练损失。
*   **开销与权衡:** 主要开销在于训练初期的数据过滤和多阶段训练，但整体训练成本极低（64个 H100 GPU 训练12天，仅为 MovieGen 的1%），推理时内存需求固定，适合长视频生成和边缘设备部署。

## Further Thoughts

线性注意力结合固定内存 KV 缓存的设计为处理长序列任务提供了新思路，未来可推广至长文本或音频生成领域；块状自回归训练和单调递增 SNR 采样策略在减少曝光偏差方面的成功，或许能启发其他自回归模型的训练优化；高效数据过滤和多阶段训练范式大幅降低资源需求，未来可探索基于生成质量的动态数据选择机制，进一步提升训练效率。