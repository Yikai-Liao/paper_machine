---
title: "Pack-PTQ: Advancing Post-training Quantization of Neural Networks by Pack-wise Reconstruction"
pubDatetime: 2025-05-01T02:53:46+00:00
slug: "2025-05-pack-wise-quantization"
type: "arxiv"
id: "2505.00259"
score: 0.48420622060609747
author: "grok-3-latest"
authors: ["Changjun Li", "Runqing Jiang", "Zhuo Song", "Pengpeng Yu", "Ye Zhang", "Yulan Guo"]
tags: ["Neural Network", "Quantization", "Post-Training", "Mixed Precision", "Cross-Block Dependency"]
institution: ["Shenzhen Campus, Sun Yat-sen University", "Aviation University of Air Force"]
description: "本文提出 Pack-PTQ 方法，通过 Hessian-guided 打包机制和包级混合精度量化策略，显著提升低比特后训练量化的性能，同时捕捉跨块依赖性。"
---

> **Summary:** 本文提出 Pack-PTQ 方法，通过 Hessian-guided 打包机制和包级混合精度量化策略，显著提升低比特后训练量化的性能，同时捕捉跨块依赖性。 

> **Keywords:** Neural Network, Quantization, Post-Training, Mixed Precision, Cross-Block Dependency

**Authors:** Changjun Li, Runqing Jiang, Zhuo Song, Pengpeng Yu, Ye Zhang, Yulan Guo

**Institution(s):** Shenzhen Campus, Sun Yat-sen University, Aviation University of Air Force


## Problem Background

神经网络在计算机视觉任务中取得了显著进展，但其高计算和内存需求使得在资源受限的边缘设备上部署成为挑战。
后训练量化（Post-Training Quantization, PTQ）作为一种无需端到端重训练的模型压缩方法受到关注，但传统 PTQ 方法多采用块级重建（block-wise reconstruction），忽略了块间依赖性（cross-block dependency），在低比特量化（如 3 比特）时精度下降明显。
本文旨在通过改进重建粒度和量化策略，解决低比特量化下的性能损失问题。

## Method

*   **核心思想:** 提出 Pack-PTQ 方法，通过将网络划分为非重叠的‘包’（packs）作为重建的基本单位，捕捉块间依赖性，并结合混合精度策略优化量化效果。
*   **Hessian-guided Adaptive Packing Mechanism:** 
    *   使用 Hessian 矩阵计算每个块的重要性分数（importance score），评估其对前序模块的影响。
    *   基于重要性分数，将连续块聚类为非重叠的包（packs），使得包内块高度相关，包间依赖性较弱。
    *   这种打包方式通过联合优化包内块，推导出更准确的量化参数。
*   **Pack-based Mixed-Precision Quantization:** 
    *   针对每个包对量化的不同敏感性（sensitivity），分配不同的比特宽度（bit-width）。
    *   通过优化问题形式化比特分配，确保在内存约束下最大化模型精度，敏感性高的包分配高比特，敏感性低的包进行更激进的低比特量化。
    *   使用包级重建损失（pack-wise reconstruction loss）对量化模型进行优化，适应不同大小的包，学习跨块依赖性。
*   **关键优势:** 不需要端到端重训练，仅依赖小规模校准数据集，同时通过数学工具（Hessian 矩阵）精确捕捉网络内部依赖关系，提升低比特量化性能。

## Experiment

*   **有效性:** Pack-PTQ 在低比特量化（如 W3/A3）下显著优于现有方法，例如在 ImageNet 数据集上，MobileNetV2 的 W3/A3 量化精度比第二好的 Genie 高出 8.15%（无混合精度）和 12.86%（有混合精度）；在 Vision Transformer 模型上，精度提升尤为明显，如 ViT-B 的 W3/A3 精度从 RepQ-ViT 的 0.14% 提升至 64.83%（有混合精度）。
*   **全面性与合理性:** 实验覆盖 2D 图像分类（ImageNet）和 3D 点云分类（ModelNet40）任务，涉及多种网络架构（CNNs、Vision Transformers、PointNet），验证了方法的普适性；消融实验表明 Hessian-guided 打包机制和混合精度策略各自贡献显著，联合使用效果最佳。
*   **特殊表现:** 在点云任务中，Pack-PTQ 在 W3/A3 设置下甚至超过全精度模型，显示出对某些任务的特殊适应性，可能与点云网络的精度冗余有关。
*   **局限性:** 对于复杂模型（如 Swin-S），计算开销较大，执行时间较长（6.82 小时），训练效率有待优化。

## Further Thoughts

Pack-PTQ 的‘包’概念可以扩展到其他模型压缩技术（如网络剪枝或知识蒸馏），通过捕捉跨层依赖性优化压缩策略；Hessian 矩阵评估重要性是否可结合其他指标（如梯度范数）进一步提高打包准确性；混合精度策略是否可动态调整，例如在推理时根据输入数据特性实时分配比特宽度；对于边缘设备，是否可设计轻量级打包机制，减少 Hessian 计算开销同时保留跨块依赖性捕捉能力。