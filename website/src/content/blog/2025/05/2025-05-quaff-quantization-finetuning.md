---
title: "Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis"
pubDatetime: 2025-05-20T07:19:36+00:00
slug: "2025-05-quaff-quantization-finetuning"
type: "arxiv"
id: "2505.14742"
score: 0.668070116562311
author: "grok-3-latest"
authors: ["Hong Huang", "Dapeng Wu"]
tags: ["LLM", "Quantization", "Fine-Tuning", "Outlier Suppression", "Efficiency"]
institution: ["City University of Hong Kong"]
description: "本文提出 Quaff 框架，基于异常空间稳定性假设（OSSH），通过目标动量缩放解耦权重和激活值量化，显著提升大型语言模型在资源受限设备上的微调效率和性能。"
---

> **Summary:** 本文提出 Quaff 框架，基于异常空间稳定性假设（OSSH），通过目标动量缩放解耦权重和激活值量化，显著提升大型语言模型在资源受限设备上的微调效率和性能。 

> **Keywords:** LLM, Quantization, Fine-Tuning, Outlier Suppression, Efficiency

**Authors:** Hong Huang, Dapeng Wu

**Institution(s):** City University of Hong Kong


## Problem Background

大型语言模型（LLMs）在资源受限的个人设备上部署时，面临任务特定微调带来的高计算和内存需求挑战。
现有量化方法（如权重-激活量化 WAQ）虽能降低资源开销，但因激活值异常（outliers）导致量化误差增大，性能下降，尤其在微调过程中，静态缩放无法适应分布变化，动态缩放则需存储全精度权重，效率低下。
论文旨在解决这一三重权衡问题（效率、性能、部署性），实现高效低开销的量化微调。

## Method

*   **核心假设：异常空间稳定性假设（OSSH）**：在微调过程中，某些激活值异常通道的空间位置跨训练迭代保持稳定。
*   **框架设计：Quaff（Quantized Parameter-Efficient Fine-Tuning）**：
    *   **预处理阶段**：基于校准数据集预识别异常通道（outlier channels），对非异常通道权重进行低精度量化（如 INT8），仅保留少量异常通道权重为全精度（控制在总权重5%以内），减少存储开销。
    *   **微调阶段**：引入目标动量缩放机制（targeted momentum scaling），仅对预识别的异常通道动态计算缩放因子，通过结合历史缩放值和当前激活值分布（使用动量参数γ控制更新惯性），平滑分布波动，减少量化误差。
    *   **计算优化**：将矩阵计算分解为静态量化部分（硬件友好）和动态异常通道部分（小规模全精度计算），避免全局权重重新量化和存储全精度权重。
*   **关键创新**：通过解耦权重和激活值量化依赖，Quaff 避免了动态缩放的高开销，同时利用 OSSH 假设减少运行时检测异常的负担，兼顾效率与性能。

## Experiment

*   **性能与效率**：在 GPQA 推理基准上，Quaff 相比全精度微调实现了 1.73 倍延迟减少和 30% 内存节省，同时精度提升 0.6%；相比其他量化方法（如 SmoothQuant），在相同约束下精度提升 2.1%。
*   **部署性**：在消费级 GPU（如 RTX 2080 Super）上，Quaff 相比全精度方法加速 8.29 倍，证明其在资源受限设备上的适用性。
*   **实验设置**：实验覆盖了多种任务（推理、指令微调、长文本）、模型（LLaMA-2、Phi-3、OPT）和微调策略（LoRA、Prompt Tuning 等），数据集选择具有代表性（如 MMLU-Pro、Alpaca-Finance），并在消费级硬件上验证，设置全面合理。
*   **局限性**：未测试更大规模模型（>7B 参数）和更低精度（如 INT4），可能限制方法在极端场景下的表现评估。
*   **结论**：实验数据显著支持 Quaff 的高效性和性能优势，验证了 OSSH 假设的有效性。

## Further Thoughts

OSSH 假设揭示了激活值异常通道的空间稳定性，这可能不仅适用于微调，也可在推理或预训练中探索类似规律，用于设计更智能的量化策略；
动量缩放机制通过平滑分布波动减少噪声影响，这种思想可扩展到其他动态调整场景，如自适应优化算法；
Quaff 强调硬件友好性，启发我们在模型设计初期考虑硬件约束，探索与边缘设备深度耦合的量化方法。