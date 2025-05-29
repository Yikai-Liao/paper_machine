---
title: "GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning"
pubDatetime: 2025-05-26T06:48:20+00:00
slug: "2025-05-gralora-fine-tuning"
type: "arxiv"
id: "2505.20355"
score: 0.668787674613904
author: "grok-3-latest"
authors: ["Yeonjoon Jung", "Daehyun Ahn", "Hyungjun Kim", "Taesu Kim", "Eunhyeok Park"]
tags: ["LLM", "Parameter Efficient Fine-Tuning", "Low-Rank Adaptation", "Gradient Dynamics", "Model Expressivity"]
institution: ["SqueezeBits", "POSTECH"]
description: "GraLoRA 通过细粒度低秩适配器克服 LoRA 的梯度纠缠问题，在不增加计算成本的情况下显著提升参数高效微调性能，尤其在高秩设置下接近全参数微调效果。"
---

> **Summary:** GraLoRA 通过细粒度低秩适配器克服 LoRA 的梯度纠缠问题，在不增加计算成本的情况下显著提升参数高效微调性能，尤其在高秩设置下接近全参数微调效果。 

> **Keywords:** LLM, Parameter Efficient Fine-Tuning, Low-Rank Adaptation, Gradient Dynamics, Model Expressivity

**Authors:** Yeonjoon Jung, Daehyun Ahn, Hyungjun Kim, Taesu Kim, Eunhyeok Park

**Institution(s):** SqueezeBits, POSTECH


## Problem Background

大型生成模型的微调中，LoRA 作为一种参数高效微调（PEFT）方法因其简单性和高效性而广受欢迎，但其存在显著局限：当增加秩（rank）以提升表达能力时，性能反而停滞或下降，无法接近全参数微调（FFT）的效果。
作者通过理论分析发现，LoRA 的结构瓶颈导致梯度纠缠（gradient entanglement），即某些异常输入通道（outlier channels）过度主导梯度更新，压制其他通道的贡献，进而限制了模型在高秩下的性能。

## Method

*   **核心思想:** 提出 GraLoRA（Granular Low-Rank Adaptation），通过将权重矩阵划分为多个子块（sub-blocks），并为每个子块配备独立的低秩适配器（low-rank adapter），实现细粒度的参数更新，以解决 LoRA 的梯度纠缠问题，增强表达能力并接近 FFT 的梯度动态。
*   **具体实现:** 
    *   将权重矩阵划分为 k×k 个子块，每个子块的输入和输出维度为原始 LoRA 的 1/k，每个子块的低秩适配器秩为 r/k，但整体表达能力从 r 提升至 kr。
    *   每个子块独立更新梯度，限制异常输入通道的影响范围，仅影响相关子块的适配器，从而减少全局梯度失真。
    *   计算和参数量与 LoRA 基本一致，保持高效性；推理时可将适配器合并到原始权重矩阵中，无额外开销。
    *   提出 Hybrid GraLoRA，在低秩场景下结合 LoRA 和 GraLoRA 的优势，通过共享部分秩给 LoRA 提升子块表达能力。
*   **关键优势:** 增强了模型对复杂特征的表达能力，尤其在高秩下避免性能下降；对异常输入更鲁棒，梯度更新更接近 FFT 的局部性。

## Experiment

*   **有效性:** 在代码生成任务（HumanEval+）中，GraLoRA 在 LLaMA3.1-8B 模型上所有秩下均优于 LoRA、MoRA 和 RaSA，尤其在秩 128 时 Pass@1 提升 8.5%，Pass@5 和 Pass@10 分别提升 6.9% 和 5.1%；在常识推理任务中，GraLoRA 在多个模型（如 Qwen2.5-1.5B 到 LLaMA3.1-70B）上平均准确率提升 0.9%-1.1%，在 24 个任务中 20 个取得最佳结果。
*   **实验设置合理性:** 实验覆盖了不同任务（代码生成、常识推理）、模型规模（1.5B 到 70B）和秩设置（16 到 128），并通过消融实验验证了 k 值和 Hybrid GraLoRA 比例的影响，设置全面且具有代表性。
*   **开销分析:** 计算开销与 LoRA 相当（FLOPs 甚至略低），训练时内存开销略增（因中间表示维度扩大 k 倍），但通过梯度检查点技术可有效缓解，实际影响较小。

## Further Thoughts

GraLoRA 的细粒度分区策略启发了对 PEFT 方法的新思考：是否可以通过自适应分区机制，根据输入特征或任务特性动态调整子块大小和秩分配，以进一步提升性能？此外，GraLoRA 的思想是否可扩展至视觉变换器或多模态模型的微调，解决类似梯度纠缠问题？另一个方向是结合稀疏性，利用任务相关性激活特定子块，减少不必要的计算和内存开销。