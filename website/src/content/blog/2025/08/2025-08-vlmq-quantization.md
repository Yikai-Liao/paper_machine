---
title: "VLMQ: Efficient Post-Training Quantization for Large Vision-Language Models via Hessian Augmentation"
pubDatetime: 2025-08-05T11:57:03+00:00
slug: "2025-08-vlmq-quantization"
type: "arxiv"
id: "2508.03351"
score: 0.6786421509575404
author: "grok-3-latest"
authors: ["Yufei Xue", "Yushi Huang", "Jiawei Shao", "Jun Zhang"]
tags: ["VLM", "Quantization", "Hessian Matrix", "Post-Training", "Modality Imbalance"]
institution: ["Institute of Artificial Intelligence (TeleAI), China Telecom", "The Hong Kong University of Science and Technology"]
description: "VLMQ 提出了一种高效的后训练量化框架，通过重要性感知的 Hessian 增强策略，针对视觉-语言模型的模态不平衡问题优化低比特量化性能。"
---

> **Summary:** VLMQ 提出了一种高效的后训练量化框架，通过重要性感知的 Hessian 增强策略，针对视觉-语言模型的模态不平衡问题优化低比特量化性能。 

> **Keywords:** VLM, Quantization, Hessian Matrix, Post-Training, Modality Imbalance

**Authors:** Yufei Xue, Yushi Huang, Jiawei Shao, Jun Zhang

**Institution(s):** Institute of Artificial Intelligence (TeleAI), China Telecom, The Hong Kong University of Science and Technology


## Problem Background

大型视觉-语言模型（VLMs）因其巨大的模型规模和多模态输入特性，在资源受限环境中部署时面临内存和计算复杂度的挑战，而现有基于 Hessian 的后训练量化（PTQ）方法（如 GPTQ）由于忽略了 VLMs 中视觉 token 的冗余性和模态不平衡问题（即文本 token 少而视觉 token 多且冗余），导致量化后性能显著下降。
论文旨在解决如何在不进行昂贵微调或重训练的情况下，设计一种高效 PTQ 方法，针对 VLMs 的视觉过表达问题优化低比特量化性能。

## Method

*   **核心思想：** 提出 VLMQ（Vision-Language Model Quantization），一种基于重要性感知的后训练量化框架，通过增强 Hessian 矩阵来优化 VLMs 的量化过程，缓解视觉 token 冗余和模态不平衡的影响。
*   **重要性感知目标函数：** 引入一个对角矩阵 **G** 表示每个 token 的重要性权重，调整量化目标函数，使重要 token 获得更高权重，冗余 token 权重降低，从而避免 Hessian 矩阵偏向冗余视觉特征。
*   **增强 Hessian 矩阵构建：** 基于重要性权重 **G**，构建增强 Hessian 矩阵（**H̃ = XGX^T**），用于指导权重更新，同时保持与现有框架（如 GPTAQ）的兼容性，支持并行化更新和 Cholesky 分解等高效技巧。
*   **梯度驱动重要性因子计算：** 建立块级损失扰动与 token 级量化误差的理论联系，通过单次轻量级块级反向传播计算梯度驱动的重要性因子，具体方法是在注意力模块后设置断点，计算局部损失（block-wise MSE），并基于 Q/K/V/O 投影输出梯度的 L1 范数（或 L2 范数）定义 token 重要性。
*   **逐层量化与误差补偿：** 逐步对每个解码层进行量化，交替局部前向/反向传播和校准，动态捕捉误差传播，并在关键投影（如 o_proj）中利用增强 Hessian 进行误差补偿。

## Experiment

*   **有效性：** VLMQ 在 INT3 和 INT2g128 量化设置下显著优于基线方法（如 GPTQ 和 GPTAQ），例如在 Qwen2.5-VL-7B-Instruct-INT2g128 上平均精度提升 1.88%，在 MME-RealWorld (Chinese) 任务上提升高达 16.45%。
*   **低比特优越性：** 在超低比特量化（如 INT2g128）中，VLMQ 通过选择 GPTQ 作为基础算法并结合重要性感知策略，显著缓解了 GPTAQ 中因误差累积导致的性能下降问题。
*   **实验设置全面性：** 实验覆盖从 0.5B 到 32B 的多个 VLMs（如 LLaVA-OneVision, Qwen2-VL），在 8 个视觉-语言基准任务（如 TextVQA, ScienceQA）上评估了文本识别、视觉感知和推理能力，校准数据包含 512 个文本-图像对，确保模态平衡。
*   **合理性与局限：** 实验设置合理，提供了详细实现细节和消融研究，但小模型（如 0.5B）量化后性能差距较大，且评估主要聚焦图像-文本任务，未涉及视频理解等场景。
*   **开销：** VLMQ 额外开销较小，主要来自每层一次局部前向/反向传播，在 H100 GPU 上仅增加 1.2-6 分钟，相比基线方法几乎可忽略。

## Further Thoughts

VLMQ 的重要性感知策略启发我们可以在其他多模态模型（如语音-文本模型）中识别冗余特征并优化量化或剪枝；其在 o_proj 中利用增强 Hessian 进行误差补偿的思路可扩展为动态误差校正机制，应用于低比特量化或长序列处理；此外，模态不平衡问题可能不仅影响量化，也可通过重要性权重调整注意力机制的计算分配，优化多模态模型整体性能。