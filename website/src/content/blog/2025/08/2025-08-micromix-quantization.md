---
title: "MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models"
pubDatetime: 2025-08-04T12:22:39+00:00
slug: "2025-08-micromix-quantization"
type: "arxiv"
id: "2508.02343"
score: 0.6395670974476352
author: "grok-3-latest"
authors: ["Wenyuan Liu", "Haoqian Meng", "Yilun Luo", "Peng Zhang", "Xindian Ma"]
tags: ["LLM", "Mixed Precision", "Quantization", "Hardware Acceleration", "Inference Optimization"]
institution: ["College of Intelligence and Computing, Tianjin University, Tianjin, China"]
description: "MicroMix 提出了一种基于 Microscaling 格式的混合精度量化算法和内核，通过自适应精度分配和硬件优化，在 Blackwell 架构上显著提升大型语言模型推理效率并保持高精度。"
---

> **Summary:** MicroMix 提出了一种基于 Microscaling 格式的混合精度量化算法和内核，通过自适应精度分配和硬件优化，在 Blackwell 架构上显著提升大型语言模型推理效率并保持高精度。 

> **Keywords:** LLM, Mixed Precision, Quantization, Hardware Acceleration, Inference Optimization

**Authors:** Wenyuan Liu, Haoqian Meng, Yilun Luo, Peng Zhang, Xindian Ma

**Institution(s):** College of Intelligence and Computing, Tianjin University, Tianjin, China


## Problem Background

大型语言模型（LLMs）推理需要巨大的计算和能耗资源，量化技术通过低精度矩阵替换高精度矩阵显著提升了推理速度。
然而，现有基于 INT4 的量化方法无法充分利用 NVIDIA Blackwell 架构的 FP4 Tensor Cores，因数据格式不匹配和内核效率瓶颈（如反量化操作耗时长）导致性能受限；同时，混合精度量化在 Microscaling (MX) 格式上的应用因缺乏自适应分配而精度下降。
MicroMix 旨在解决这些问题，通过设计基于 MX 格式的混合精度量化算法和内核，提升效率并减少精度损失。

## Method

*   **核心思想:** 提出 MicroMix，一种基于 Microscaling (MX) 数据格式（MXFP4, MXFP6, MXFP8）的混合精度量化框架，通过自适应精度分配和高效内核设计，在 Blackwell 架构上实现高效推理，同时控制量化误差。
*   **自适应精度分配:** 
    *   将激活和权重通道分为三组，分别对应 MXFP4, MXFP6, MXFP8 格式。
    *   定义量化阈值（Quantization Threshold），确保低精度格式（如 MXFP4, MXFP6）的量化误差不超过高精度格式（如 INT8）的误差上界。
    *   根据激活通道的绝对均值排序，优先为重要通道（通常是具有较大均值的通道）分配高精度格式，适应不同层的激活分布。
    *   利用校准数据集（如 WikiText2）离线预计算通道分配比例，避免在线计算的运行时开销。
*   **高效内核设计:** 
    *   设计支持多种 MX 格式混合的矩阵乘法内核，利用 CUTLASS GEMM 和 Blackwell Tensor Cores 的 MMA 指令，将反量化操作深度融合到计算过程中，减少额外开销。
    *   采用融合重排序和量化操作（Reorder-and-Quantize），将激活和权重通道重排序以实现规则内存访问，解决混合精度带来的性能下降问题；激活动态重排序，权重离线预处理。
    *   在 Transformer 块中集成 MicroMix 内核，替换所有线性层，并在 LayerNorm 后执行单次重排序和量化操作以共享激活矩阵，进一步提升效率。
*   **关键点:** 方法兼顾精度与效率，通过误差控制确保模型性能，通过硬件优化提升计算速度，且支持灵活的精度比例调整。

## Experiment

*   **精度表现:** 在 Llama3.1-8B 和 Qwen2.5-32B 模型上，MicroMix 在零样本任务中保留了 FP16 精度的 95% 以上，在 MMLU（五样本任务）上保留了 90% 以上，困惑度劣化控制在 20% 以内，优于基线方法（如 Atom, QuaRot, AMXFP4）；在代码生成和数学推理任务中，精度下降小于 3%，部分场景甚至优于 FP16。
*   **效率提升:** 在 RTX 5070Ti 和 RTX 5090 GPU 上，MicroMix 内核计算速度比 TensorRT-FP8 快 8%-46%，Transformer 块预填充速度提升 6%-29%，端到端吞吐量提升 3.5%-9.7%，峰值内存使用减少约 20%，效率提升显著。
*   **实验设置合理性:** 实验覆盖多种模型（Llama, Qwen 系列）、任务（零样本、少样本、代码、数学）、硬件（消费级和服务器级 GPU）以及批次大小和序列长度，设置全面；消融研究验证了 MX 格式组合（如 E4M3+E3M2）和校准数据集选择的鲁棒性。
*   **总结:** MicroMix 在精度和效率之间取得良好平衡，实验数据支持其方法有效性，尤其在 Blackwell 架构上得益于 FP4 Tensor Cores 的支持。

## Further Thoughts

MicroMix 的自适应精度分配思路启发我们可以在模型剪枝或知识蒸馏中根据层或任务重要性动态分配资源；其硬件-算法协同设计提示未来应更深度结合硬件特性（如 AMD 或 TPU 架构）进行优化；此外，量化阈值的定义方法可能适用于其他量化格式或混合精度训练场景，值得探索其理论基础和泛化性。