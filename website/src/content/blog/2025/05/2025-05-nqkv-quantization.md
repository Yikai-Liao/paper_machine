---
title: "NQKV: A KV Cache Quantization Scheme Based on Normal Distribution Characteristics"
pubDatetime: 2025-05-22T04:23:19+00:00
slug: "2025-05-nqkv-quantization"
type: "arxiv"
id: "2505.16210"
score: 0.7932237502809838
author: "grok-3-latest"
authors: ["Zhihang Cai", "Xingjun Zhang", "Zhendong Tan", "Zheng Wei"]
tags: ["LLM", "Quantization", "KV Cache", "Memory Optimization", "Inference"]
institution: ["Xi’an Jiaotong University"]
description: "本文提出 NQKV 算法，基于 KV Cache 元素的正态分布特性进行分块 4 位量化，显著减少内存占用并支持更大批次大小和更长上下文，同时保持模型精度和提升吞吐量。"
---

> **Summary:** 本文提出 NQKV 算法，基于 KV Cache 元素的正态分布特性进行分块 4 位量化，显著减少内存占用并支持更大批次大小和更长上下文，同时保持模型精度和提升吞吐量。 

> **Keywords:** LLM, Quantization, KV Cache, Memory Optimization, Inference

**Authors:** Zhihang Cai, Xingjun Zhang, Zhendong Tan, Zheng Wei

**Institution(s):** Xi’an Jiaotong University


## Problem Background

大型语言模型（LLMs）在推理过程中，注意力机制的键值对缓存（KV Cache）随着批次大小和上下文长度的增加，内存占用急剧上升，成为部署的主要瓶颈。
例如，OPT-175B 模型在批次大小为 64、序列长度为 8192 时，KV Cache 占用高达 83.78% 的 GPU 内存（2.3TB），是模型参数的 7 倍。
现有量化方法（如 8 位量化）在进一步降低到 4 位时会导致显著精度下降，因此需要在不牺牲模型输出的前提下大幅减少 KV Cache 内存占用。

## Method

*   **核心思想:** 基于 KV Cache 元素在 token 和 block 维度上符合正态分布的特性，提出 NQKV 算法，通过分块量化（per-block quantization）和正态分布存储数据类型（如 Normal Float, NF4）来最小化量化误差，将 KV Cache 量化到 4 位。
*   **具体实现:** 
    *   **预填充阶段（Prefill Phase）:** 将输入提示生成的键（Key）和值（Value）按 token 维度划分为固定大小的块（block size 为 256），对每个块进行 4 位 NF4 量化，存储量化后的索引（indices）到 KV Cache 中，而非直接存储浮点数，以节省内存。
    *   **解码阶段（Decoding Phase）:** 对新生成的键和值同样进行分块 NF4 量化，追加到 KV Cache 末尾；随后通过查表方式反量化（dequantize）为 16 位浮点数，用于注意力机制计算，适应 KV Cache 的流式特性。
    *   **优化策略:** 引入填充（padding）技术，在计算时对 KV Cache 的 token 维度进行填充（确保尺寸为 16 的倍数），以满足 GPU 硬件要求并利用高效矩阵乘法内核（如 Cutlass GEMM），避免直接填充带来的额外存储和计算开销。
*   **关键优势:** 不需要对模型进行重新训练或微调，仅在推理时对 KV Cache 进行量化操作，与其他权重或激活量化方法正交，可组合使用。

## Experiment

*   **精度表现:** NQKV 将 KV Cache 量化到 4 位后，对模型精度的影响极小，例如在 OPT-1.3B 上平均精度下降仅 0.7%，在更大规模的 OPT-6.7B 和 OPT-13B 上影响更小，甚至在某些零样本任务（如与 SmoothQuant 结合时）略有提升，显示出在大模型上的鲁棒性。
*   **内存节省:** NQKV 显著减少内存占用，使 OPT-6.7B 模型支持 4 倍批次大小或 2.5 倍序列长度（相比 FP16 模型），对于 OPT-30B，在大批量和长序列场景下比 SmoothQuant 额外节省 60%-80% 内存。
*   **吞吐量提升:** 启用 KV Cache 后，NQKV 显著提升吞吐量，例如在 OPT-30B 上比不使用 KV Cache 的 SmoothQuant 快 9.3 倍，但由于反量化操作，吞吐量略低于 SmoothQuant 的 KV Cache 版本（损失小于 20%）。
*   **实验设置合理性:** 实验覆盖了 OPT 模型家族（125M 到 30B）和多个零样本任务（PIQA, WinoGrande 等），在 Nvidia A100 GPU（80GB）上测试，与基线方法（如 SmoothQuant）对比，设置较为全面；但未涉及多硬件平台或不同数据集的泛化性测试，可能有一定局限性。

## Further Thoughts

基于数据分布特性设计量化方案的思路非常具有启发性，NQKV 利用 KV Cache 元素符合正态分布的特性，选择贴近分布的存储数据类型（如 NF4）来减少量化误差，这种方法可以推广到其他深度学习模型的中间表示（如激活值、梯度）上，通过分析其分布特性定制量化策略；此外，分块量化（per-block quantization）将误差限制在局部范围内的思想，也可能在其他内存优化或误差控制场景中应用。