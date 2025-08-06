---
title: "MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models"
pubDatetime: 2025-08-04T12:22:39+00:00
slug: "2025-08-micromix-quantization"
type: "arxiv"
id: "2508.02343"
score: 0.6395670974476352
author: "grok-3-latest"
authors: ["Wenyuan Liu", "Haoqian Meng", "Yilun Luo", "Peng Zhang", "Xindian Ma"]
tags: ["LLM", "Mixed Precision", "Quantization", "Hardware Optimization", "Inference Efficiency"]
institution: ["College of Intelligence and Computing, Tianjin University, Tianjin, China"]
description: "MicroMix 提出了一种基于 Microscaling 格式的混合精度量化框架，通过自适应精度分配和高效内核设计，在 Blackwell 架构上显著提升大型语言模型推理效率和内存利用率，同时保持高精度。"
---

> **Summary:** MicroMix 提出了一种基于 Microscaling 格式的混合精度量化框架，通过自适应精度分配和高效内核设计，在 Blackwell 架构上显著提升大型语言模型推理效率和内存利用率，同时保持高精度。 

> **Keywords:** LLM, Mixed Precision, Quantization, Hardware Optimization, Inference Efficiency

**Authors:** Wenyuan Liu, Haoqian Meng, Yilun Luo, Peng Zhang, Xindian Ma

**Institution(s):** College of Intelligence and Computing, Tianjin University, Tianjin, China


## Problem Background

大型语言模型（LLMs）推理面临巨大的计算和能耗成本，量化技术通过低精度矩阵替换高精度矩阵显著提升了推理速度。
然而，现有 INT4 量化内核无法充分利用 NVIDIA Blackwell 架构的 FP4 Tensor Cores 的性能潜力，且混合精度量化方法在应用于 Microscaling (MX) 格式时，因缺乏对激活分布多样性的自适应分配，导致精度下降。
论文旨在解决这些瓶颈，提出一种高效的混合精度量化框架，提升计算效率并保持模型精度。

## Method

*   **核心思想：** 提出 MicroMix，一个基于 Microscaling (MX) 数据格式（MXFP4、MXFP6、MXFP8）的混合精度量化框架，通过算法与内核协同设计，利用 Blackwell 架构的 FP4 Tensor Cores 提升效率，同时通过自适应精度分配减少量化误差。
*   **自适应精度分配：** 定义量化阈值（以 INT8 量化误差为上界），将激活和权重通道分为三组（MXFP4、MXFP6、MXFP8），对量化误差较大的通道分配更高精度，确保误差控制在可接受范围内。
*   **离线校准与通道排序：** 利用校准数据集（如 WikiText2）离线确定各精度通道的比例，并根据通道的绝对均值排序，优先为重要通道分配高精度，减少在线计算开销。
*   **高效内核设计：** 设计支持多种 MX 格式的矩阵乘法（GEMM）内核，利用 CUTLASS 库优化计算，融合去量化和重排序操作以减少内存访问不规则带来的性能损失，同时支持 BFloat16 输出。
*   **Transformer 集成：** 将 MicroMix 内核集成到 Transformer 块中，结合 FlashInfer 优化注意力计算和 KV Cache 操作，进一步提升整体效率。
*   **关键优势：** 充分利用硬件特性，避免 INT 量化中去量化到 CUDA Cores 的开销，实现动态精度比例的灵活调整。

## Experiment

*   **精度表现：** 在零样本任务中，MicroMix 保留了 FP16 精度的 95% 以上，在 MMLU（五样本）任务中保留了 90% 以上，困惑度劣化控制在 20% 以内，优于或媲美 Atom、QuaRot 等基线方法，在代码生成和数学推理任务中精度下降小于 3%。
*   **效率提升：** 相比 TensorRT-FP8，MicroMix 在内核计算速度上提升 8%-46%（RTX 5070Ti 笔记本）和 16%-46%（RTX 5090 服务器）；Transformer 块预填充速度提升 6%-29%；端到端吞吐量提升 3.5%-9.7%，峰值内存使用减少约 20%。
*   **实验设置合理性：** 实验覆盖多种模型（Llama、Qwen 系列）、任务（零样本、少样本、语言建模、代码生成、数学推理）和硬件环境，校准数据集选择对结果影响小于 1%，消融研究验证了 MXFP8 (E4M3) 和 MXFP6 (E3M2) 组合的优越性，整体设计全面且结论可信。

## Further Thoughts

MicroMix 启发我们关注硬件-算法协同设计的重要性，未来可针对不同 GPU 架构定制量化策略；其自适应量化阈值方法可扩展至其他领域（如图像处理），或探索任务优先级的动态阈值调整；离线校准的成功应用提示我们可通过更大规模校准数据集或跨模型共享校准结果提升泛化性；此外，通道级精度分配还可进一步细化至 token 级或上下文相关控制，以应对更复杂的激活分布。