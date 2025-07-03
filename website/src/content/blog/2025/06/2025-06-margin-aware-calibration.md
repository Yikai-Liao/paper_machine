---
title: "Sample Margin-Aware Recalibration of Temperature Scaling"
pubDatetime: 2025-06-30T03:35:05+00:00
slug: "2025-06-margin-aware-calibration"
type: "arxiv"
id: "2506.23492"
score: 0.5218601545870628
author: "grok-3-latest"
authors: ["Haolan Guo", "Linwei Tao", "Haoyang Luo", "Minjing Dong", "Chang Xu"]
tags: ["Neural Network", "Calibration", "Uncertainty Quantification", "Temperature Scaling", "Post-Training"]
institution: ["University of Sydney", "City University of Hong Kong"]
description: "本文提出 SMART 方法，利用 logit gap 作为校准指标，通过轻量级样本级温度调整显著提升神经网络校准性能，同时保持数据和计算效率。"
---

> **Summary:** 本文提出 SMART 方法，利用 logit gap 作为校准指标，通过轻量级样本级温度调整显著提升神经网络校准性能，同时保持数据和计算效率。 

> **Keywords:** Neural Network, Calibration, Uncertainty Quantification, Temperature Scaling, Post-Training

**Authors:** Haolan Guo, Linwei Tao, Haoyang Luo, Minjing Dong, Chang Xu

**Institution(s):** University of Sydney, City University of Hong Kong


## Problem Background

深度神经网络在预测准确性显著提升的同时，普遍存在过自信（overconfidence）问题，这在安全关键场景（如医疗诊断、自动驾驶）中带来风险。
现有的后处理校准方法面临两难：全局方法（如温度缩放 TS）对所有样本统一调整，引入高偏差；而更复杂的参数化方法由于高维输入噪声和验证数据不足，存在高方差问题。
论文旨在设计一种轻量级、数据高效的校准方法，针对每个样本特性进行精确温度调整，降低校准误差，同时保持模型预测不变。

## Method

*   **核心思想:** 提出 SMART（Sample Margin-Aware Recalibration of Temperature），利用最大和次大 logits 之间的差值（logit gap）作为校准指标，针对每个样本动态调整温度，以实现精确校准。
*   **具体实现:** 
    *   **Logit Gap 指标:** Logit gap 是一个标量信号，量化了模型在决策边界上的不确定性。相比高维 logits 或嵌入特征，logit gap 去除了噪声，且不改变模型预测结果。论文通过理论分析证明，logit gap 能紧密界定最优温度调整范围。
    *   **轻量级温度回归:** 使用一个单隐藏层多层感知机（MLP，仅 49 个参数）根据 logit gap 预测每个样本的温度值，避免了过参数化问题，确保计算效率。
    *   **SoftECE 目标函数:** 引入软分箱期望校准误差（SoftECE）作为优化目标，通过平滑分箱策略平衡偏差和方差，使得即使验证数据极少（低至 50 个样本），也能稳定更新温度参数。
*   **关键优势:** 作为后处理方法，SMART 不需重新训练原始模型，仅通过 logits 调整实现校准，保持预测不变，同时克服了全局方法的粗粒度局限和复杂参数化方法的高方差问题。

## Experiment

*   **有效性:** SMART 在多个数据集（CIFAR-10, CIFAR-100, ImageNet 及其变体）和模型架构（CNN 如 ResNet-50, Transformer 如 ViT-B/16）上显著降低校准误差（如 ECE 和 AdaECE），优于基线方法（TS, PTS, CTS, Spline）。例如，在 CIFAR-100 上，SMART 将 ResNet-50 的 ECE 从 17.53% 降至 1.37%。
*   **数据效率:** 在验证数据极少（低至 ImageNet 验证集的 0.001%）时，SMART 仍保持稳定性能，而 PTS 等方法 ECE 显著上升。
*   **鲁棒性:** 在分布偏移场景（如 ImageNet-C, ImageNet-LT）下，SMART 表现出色，尤其在 Transformer 架构上，TS 等全局方法甚至恶化校准，而 SMART 始终保持正向改进。
*   **计算效率:** SMART 运行时间（23.03 秒）远低于 CTS（5457.55 秒），参数量仅 49 个，展现高效性。
*   **实验设置合理性:** 实验覆盖多种数据集、模型架构和分布偏移场景，评估指标全面（ECE, AdaECE），数据支持结论显著且合理。

## Further Thoughts

Logit gap 作为一个简洁的标量不确定性指标，不仅适用于校准问题，还可能在异常检测或主动学习中发挥作用，值得探索其在多任务学习中的潜力；
SoftECE 的软分箱策略平衡偏差和方差，可推广至数据稀疏场景下的其他优化问题，如强化学习中的奖励估计；
SMART 证明少参数模型在特定任务中可能更有效，启发我们在 AI 设计中优先考虑任务特异性信号而非盲目增加复杂度。