---
title: "FlashOmni: A Unified Sparse Attention Engine for Diffusion Transformers"
pubDatetime: 2025-09-29T18:57:14+00:00
slug: "2025-09-flashomni-sparse-attention"
type: "arxiv"
id: "2509.25401"
score: 0.565611794265094
author: "grok-3-latest"
authors: ["Liang Qiao", "Yue Dai", "Yeqi Huang", "Hongyu Kan", "Jun Shi", "Hong An"]
tags: ["Diffusion Transformer", "Sparse Attention", "Feature Caching", "Acceleration", "Visual Generation"]
institution: ["University of Science and Technology of China", "University of Edinburgh", "University of Virginia"]
description: "本文提出 FlashOmni，一个统一的稀疏注意力引擎，通过多粒度稀疏策略和通用内核设计显著提升扩散变换器的推理效率，同时保持视觉生成质量。"
---

> **Summary:** 本文提出 FlashOmni，一个统一的稀疏注意力引擎，通过多粒度稀疏策略和通用内核设计显著提升扩散变换器的推理效率，同时保持视觉生成质量。 

> **Keywords:** Diffusion Transformer, Sparse Attention, Feature Caching, Acceleration, Visual Generation

**Authors:** Liang Qiao, Yue Dai, Yeqi Huang, Hongyu Kan, Jun Shi, Hong An

**Institution(s):** University of Science and Technology of China, University of Edinburgh, University of Virginia


## Problem Background

扩散变换器（Diffusion Transformers, DiTs）在高保真视觉生成任务（如图像和视频生成）中表现出色，但其注意力机制的高计算复杂度限制了推理效率，尤其是在高分辨率图像和长视频生成场景中。
现有的稀疏加速方法（如特征缓存和块稀疏跳跃）因稀疏粒度不一致、设计空间碎片化以及缺乏通用内核支持，导致难以跨任务复用和高效部署。
本文旨在解决这些问题，提供一个统一的框架以支持多种稀疏策略的高效执行，减少工程开销并提升 DiTs 的可扩展性。

## Method

*   **核心思想:** 提出 FlashOmni，一个统一的稀疏注意力引擎，通过‘Update-Dispatch’范式整合多粒度稀疏策略，在不牺牲生成质量的前提下加速 DiTs 推理。
*   **具体实现:**
    *   **统一稀疏符号（Sparse Symbols）:** 使用 8 位稀疏符号（Sc 和 Ss）统一表示多种稀疏策略，包括特征缓存（Feature Caching）和块稀疏跳跃（Block-Sparse Skipping）。这些符号基于注意力图的压缩表示生成，通过评估视觉-文本贡献和指导度量，决定哪些块进行缓存或跳跃计算。
    *   **通用稀疏注意力内核:** 设计一个通用的注意力内核，运行时解码稀疏符号，支持任意稀疏模式的执行。内核通过协作线程数组（CTA）根据符号选择计算模式（按需计算或缓存重用），并融合元素级操作以减少开销。
    *   **优化稀疏矩阵乘法（Sparse GEMMs）:** 针对注意力模块的线性层，设计 GEMM-Q 和 GEMM-O 操作，利用稀疏符号跳过冗余计算。GEMM-Q 在空间轴上根据符号决定是否计算查询向量；GEMM-O 在约简轴上通过缓存偏置（Bias）减少重复计算，并优化存储逻辑以降低内存消耗。
*   **关键特点:** 不需要重新训练模型，支持任意 DiT 架构，通过动态稀疏策略平衡效率和质量，尤其在早期去噪步骤保持高密度计算以确保跨模态一致性。

## Experiment

*   **有效性:** FlashOmni 在 FLUX.1 和 HunyuanVideo 等模型上表现出显著加速效果。注意力计算和 GEMM-Q 操作实现了接近线性的加速（与稀疏比例 1:1 匹配）；GEMM-O 操作实现了 2.5× 到 3.8× 的加速（最高接近理论极限的 87.5%）。端到端测试中，Hunyuan 模型在 46% 稀疏度下实现了约 1.5× 的加速。
*   **质量保持:** 通过 PSNR、LPIPS、SSIM 等指标验证，FlashOmni 在高稀疏度下仍保持了与全注意力计算相当的视觉生成质量，优于基线方法（如 SpargeAttn、DiTFastAttnV2、TaylorSeer）。
*   **对比优势:** 与块稀疏跳跃和特征缓存方法相比，FlashOmni 在相同稀疏度下提供更高的质量和效率，尤其在多模态融合（如视觉-文本交互）中表现更稳定。
*   **实验设置合理性:** 实验覆盖了图像生成（COCO-2017 数据集）和视频生成（VBench 基准）任务，评估指标全面（包括质量和效率），在 NVIDIA A100 GPU 上测试了多种稀疏配置，验证了方法的普适性和鲁棒性。
*   **开销分析:** 主要额外开销来自稀疏符号的解码操作，尤其在 GEMM-O 的约简轴上多次解码导致部分效率损失，但整体仍接近理论加速上限。

## Further Thoughts

FlashOmni 的稀疏符号设计提供了一种抽象方式，将不同稀疏策略统一到一个框架中，这种思想不仅适用于 DiTs，也可能推广到其他 Transformer 架构（如大型语言模型 LLMs），以实现跨领域的计算优化。
此外，‘Update-Dispatch’范式启发了一种分离策略更新与执行阶段的思路，可在其他生成模型或实时推理场景中应用，以减少频繁更新的计算负担。
另一个值得关注的点是动态稀疏策略，即在早期去噪步骤保持高密度计算以确保质量，这种基于任务阶段的自适应稀疏性设计可能为其他生成模型的优化提供新思路。