---
title: "Scaling with Collapse: Efficient and Predictable Training of LLM Families"
pubDatetime: 2025-09-29T17:26:11+00:00
slug: "2025-09-scaling-collapse-llm"
type: "arxiv"
id: "2509.25087"
score: 0.8712696656494962
author: "grok-3-latest"
authors: ["Shane Bergsma", "Bin Claire Zhang", "Nolan Dey", "Shaheer Muhammad", "Gurpreet Gosal", "Joel Hestness"]
tags: ["LLM", "Scaling Laws", "Training Efficiency", "Hyperparameter Tuning", "Loss Curve Collapse"]
institution: ["Cerebras Systems"]
description: "本文通过固定 TPP 和优化 AdamW 时间尺度 τ，实现了大规模 LLM 训练损失曲线的坍缩，并开发了 Celerity 模型家族及相关工具（如提前停止和问题诊断），显著提升了训练的可预测性和计算效率。"
---

> **Summary:** 本文通过固定 TPP 和优化 AdamW 时间尺度 τ，实现了大规模 LLM 训练损失曲线的坍缩，并开发了 Celerity 模型家族及相关工具（如提前停止和问题诊断），显著提升了训练的可预测性和计算效率。 

> **Keywords:** LLM, Scaling Laws, Training Efficiency, Hyperparameter Tuning, Loss Curve Collapse

**Authors:** Shane Bergsma, Bin Claire Zhang, Nolan Dey, Shaheer Muhammad, Gurpreet Gosal, Joel Hestness

**Institution(s):** Cerebras Systems


## Problem Background

随着大型语言模型（LLM）规模的不断扩大，直接在大规模上进行实验变得成本高昂且不可行，因此需要在训练过程中预测关键指标（如损失曲线）并优化超参数设置。
论文从 Qiu 等人（2025）的发现出发，即训练损失曲线（Training Loss Curves, TLCs）在简单归一化后可以坍缩（collapse）到一条通用轨迹上，但这一现象是否适用于实际的大规模 LLM 训练（涉及宽度、深度、学习率、批大小和权重衰减的联合缩放）尚待验证。
核心问题是探索如何在实际训练中实现损失曲线的坍缩，并利用这一特性提高训练效率和可预测性，解决大规模训练中实验机会有限的困境。

## Method

*   **核心思想：** 通过特定的超参数缩放规则，确保不同规模模型的训练损失曲线（TLCs）在归一化后坍缩到一条通用轨迹上，从而提高训练的可预测性和计算效率。
*   **具体实现：**
    *   **超参数缩放规则：** 基于最大更新参数化（Maximal Update Parameterization, µP），在不同模型规模下联合缩放宽度、深度、学习率、批大小和权重衰减，确保训练动态的一致性。关键是固定每参数训练 token 数（Tokens-Per-Parameter, TPP）和 AdamW 优化器的时间尺度 τ（由学习率 η、权重衰减 λ 和总训练步数 T 决定，τ = B/(ηλD)）。
    *   **损失曲线归一化：** 通过将训练损失除以最终损失值（或通过早期对齐策略估计最终损失），实现不同规模模型损失曲线的坍缩，验证其在实际大规模训练中的普适性。
    *   **预测模型：** 开发一个参数化的预测模型（基于训练分数 t̂ 和学习率调度 η(t̂) 的幂律形式），用于预测归一化损失曲线，支持超参数调优中的提前停止策略。
    *   **Celerity 模型家族：** 引入 Celerity 作为首个在坍缩机制下训练的大规模 LLM 家族，验证理论在实际应用中的效果，涵盖从 300M 到 3.9B 参数规模，采用固定 TPP 和优化 τ 的训练策略。
*   **理论支持：** 使用噪声二次模型（Noisy Quadratic Model, NQM）分析 τ 对训练动态的影响，解释其如何通过控制偏差-方差权衡来塑造损失曲线。
*   **关键创新：** 不依赖于特定模型规模或架构，而是通过控制 TPP 和 τ 实现通用坍缩，同时提供诊断工具（如坍缩残差）和调优策略（如提前停止），直接应用于大规模训练实践。

## Experiment

*   **坍缩效果：** 在固定 TPP 和优化 τ 的条件下，Celerity 模型家族（参数规模从 300M 到 3.9B）的训练损失曲线实现了紧密的坍缩，尤其在 TPP=80 时效果最佳，验证了坍缩作为计算高效训练标志的理论。
*   **计算效率：** Celerity 模型在计算效率前沿上表现优异，与其他公开模型（如 BTLM）相比，在减少 75% 训练 FLOPs 的情况下达到了类似精度，表明坍缩机制有助于优化资源使用。
*   **诊断能力：** 坍缩残差（collapse residuals）能够提前检测训练问题，例如在 1.8B 模型训练中，数值不稳定性在训练进度 60% 时即被识别，而传统方法需等到 90% 才发现异常，显著缩短了问题定位时间。
*   **提前停止效果：** 在超参数调优中，提前停止策略非常有效，例如在 1.7B 和 3.3B 模型中，仅用 10%-30% 的训练数据即可预测最终损失并选择最佳设置，节省了大量计算资源。
*   **实验设置合理性：** 实验覆盖了多种 TPP（20、80、234）、学习率调度（恒定、线性衰减等）以及架构（密集模型和 MoE），设置较为全面，但论文也指出局限性，如未涉及多轮训练、极端 TPP 或重度数据课程调整等场景，可能影响坍缩的普适性。

## Further Thoughts

坍缩作为计算高效训练的‘签名’，未来可能扩展到验证损失或下游任务性能的坍缩（validation-collapse 或 downstream-collapse），为训练质量提供更全面的评估指标；
坍缩残差作为早期诊断工具的潜力，不仅限于数值问题，还可能用于检测数据分布偏移或训练过程中的其他异常，增强训练稳定性；
时间尺度 τ 作为控制训练动态的核心参数，其在其他优化器（如 Sophia）或不同训练阶段（如数据课程调整）中的作用值得进一步探索，可能揭示更广泛的优化规律；
Celerity 模型在计算效率和参数效率之间的权衡分析，启发我们思考如何通过调整 TPP 或架构设计（如 MoE）来优化训练-推理成本比，为未来超大规模模型训练提供策略参考。