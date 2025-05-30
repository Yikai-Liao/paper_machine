---
title: "Pioneering 4-Bit FP Quantization for Diffusion Models: Mixup-Sign Quantization and Timestep-Aware Fine-Tuning"
pubDatetime: 2025-05-27T13:40:47+00:00
slug: "2025-05-fp-quantization-diffusion"
type: "arxiv"
id: "2505.21591"
score: 0.5961387415081051
author: "grok-3-latest"
authors: ["Maosen Zhao", "Pengtao Chen", "Chong Yu", "Yan Wen", "Xudong Tan", "Tao Chen"]
tags: ["Diffusion Models", "Model Quantization", "Floating-Point Quantization", "Fine-Tuning", "Image Generation"]
institution: ["Fudan University"]
description: "本文提出 MSFP 框架、TALoRA 细调和 DFA 损失对齐，首次实现扩散模型 4 位 FP 量化，显著提升低位量化性能，为资源受限环境部署提供可行方案。"
---

> **Summary:** 本文提出 MSFP 框架、TALoRA 细调和 DFA 损失对齐，首次实现扩散模型 4 位 FP 量化，显著提升低位量化性能，为资源受限环境部署提供可行方案。 

> **Keywords:** Diffusion Models, Model Quantization, Floating-Point Quantization, Fine-Tuning, Image Generation

**Authors:** Maosen Zhao, Pengtao Chen, Chong Yu, Yan Wen, Xudong Tan, Tao Chen

**Institution(s):** Fudan University


## Problem Background

扩散模型（Diffusion Models, DMs）在图像生成中表现出色，但其高计算和内存需求限制了在资源受限设备上的部署。
模型量化通过降低权重和激活值位宽来减少资源占用，但现有方法（如整数量化 INT 和后训练量化 PTQ）在 4 位量化时性能下降明显，尤其难以处理激活值分布的复杂性和去噪过程的时间步长依赖性。
本文探索浮点量化（FP Quantization）在扩散模型中的低位应用，旨在实现 4 位量化同时保持接近全精度模型的性能。

## Method

*   **Mixup-Sign Floating-Point Quantization (MSFP):** 针对激活值分布的不对称性（由非线性激活函数 SiLU 引起），提出混合签名浮点量化策略。
    *   对于正常激活分布层（Normal-Activation-Distribution Layers, NALs），采用传统有符号 FP 量化，确保对称分布的有效表示。
    *   对于异常激活分布层（Anomalous-Activation-Distribution Layers, AALs），引入无符号 FP 量化并添加零点（zero point），以适配不对称分布，减少子零区域的精度损失。
    *   通过搜索-based 初始化确定量化参数（如格式、偏置和零点），在量化前后分布间最小化 MSE。
*   **Timestep-Aware LoRA (TALoRA):** 针对去噪过程中的时间步长依赖性，提出多 LoRA 模块结合时间步长感知路由器的细调方法。
    *   引入多个 LoRA 模块，每个模块针对不同时间步长的去噪特性进行优化。
    *   使用共享路由器（由时间嵌入层和 MLP 组成），根据时间步长动态选择合适的 LoRA 模块，适应从轮廓到细节的去噪任务需求。
    *   通过 STE 方法将路由器输出转换为 0/1 概率，确保高效分配。
*   **Denoising-Factor Loss Alignment (DFA):** 解决传统损失函数与量化误差在不同时间步长上的不匹配问题。
    *   引入去噪因子（denoising factor, γ_t），根据时间步长调整预测噪声的影响权重。
    *   修改损失函数为 γ_t 乘以传统 MSE 损失，使其与实际量化性能差距对齐，提升细调效果。

## Experiment

*   **有效性:** 在 4 位量化下，本文方法在多个数据集（如 CIFAR-10, CelebA, LSUN, ImageNet）和模型（DDIM, LDM）上显著优于现有 PTQ 和细调方法（如 EfficientDM, QuEST）。例如，在 CIFAR-10 上，FID 仅比全精度模型差 1.84，而其他方法差距更大；在 ImageNet 条件生成任务中，sFID 提升 7.00。
*   **稳定性:** 在 6 位量化下，性能几乎与全精度模型一致，表明方法在不同位宽下均有效。
*   **实验设置合理性:** 实验涵盖无条件和条件生成任务，数据集多样，采样方法包括 DDIM, PLMS 和 DPM-Solver，验证了方法的普适性。消融实验表明 MSFP, TALoRA 和 DFA 各模块均有贡献，组合效果最佳。
*   **局限性:** 论文未深入讨论方法在极低资源设备上的实际部署效果（如推理延迟和内存占用），这可能是未来改进方向。

## Further Thoughts

MSFP 的分布适应性量化策略启发我们探索自适应量化参数学习，通过动态调整量化方式进一步减少误差；TALoRA 的时间步长依赖性建模可推广至其他序列生成任务（如视频生成），利用多任务细调提升性能；DFA 的损失对齐思路提示在其他量化任务中引入任务特异性因子，避免优化偏差。