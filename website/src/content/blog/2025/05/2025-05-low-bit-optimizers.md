---
title: "Pushing the Limits of Low-Bit Optimizers: A Focus on EMA Dynamics"
pubDatetime: 2025-05-01T06:47:45+00:00
slug: "2025-05-low-bit-optimizers"
type: "arxiv"
id: "2505.00347"
score: 0.5150351261842896
author: "grok-3-latest"
authors: ["Cong Xu", "Wenbin Liang", "Mo Yu", "Anan Liu", "Ke-Yue Zhang", "Lizhuang Ma", "Jianyong Wang", "Jun Wang", "Wei Zhang"]
tags: ["LLM", "Optimization", "Quantization", "Memory Efficiency", "Training"]
institution: ["East China Normal University", "Tencent Youtu Lab", "WeChat AI, Tencent", "Machine Learning Platform Department, Tencent TEG", "Tsinghua University", "Shanghai Innovation Institute"]
description: "本文提出 SOLO 框架，通过针对 EMA 更新特性的对数量化和动量调整，将优化器状态精度降低至 2 位或 3 位，同时保持接近全精度的训练性能，为资源受限环境下的 AI 研究提供可行解决方案。"
---

> **Summary:** 本文提出 SOLO 框架，通过针对 EMA 更新特性的对数量化和动量调整，将优化器状态精度降低至 2 位或 3 位，同时保持接近全精度的训练性能，为资源受限环境下的 AI 研究提供可行解决方案。 

> **Keywords:** LLM, Optimization, Quantization, Memory Efficiency, Training

**Authors:** Cong Xu, Wenbin Liang, Mo Yu, Anan Liu, Ke-Yue Zhang, Lizhuang Ma, Jianyong Wang, Jun Wang, Wei Zhang

**Institution(s):** East China Normal University, Tencent Youtu Lab, WeChat AI, Tencent, Machine Learning Platform Department, Tencent TEG, Tsinghua University, Shanghai Innovation Institute


## Problem Background

随着模型规模的快速增长，训练和微调大型模型（如大型语言模型 LLMs）的内存成本变得极为高昂，尤其是状态优化器（如 AdamW）需要维护大量辅助信息（通常是模型参数的两倍大小），导致计算资源成为研究瓶颈；论文旨在解决如何在大幅减少优化器状态内存占用的同时，尽量保持模型训练性能和收敛性，特别是在超低比特（如 2 位或 3 位）量化下避免现有方法的性能下降或训练失败问题。

## Method

* **核心思想**：提出 SOLO（Stateful Optimizers in ultra-Low bits）框架，通过针对指数移动平均（EMA）更新动态特性设计超低比特量化策略，解决内存占用问题，同时尽量维持优化器性能。
* **无符号 EMA 更新（自适应学习率估计）**：针对信号淹没（Signal Swamping）问题，即小信号被当前状态值淹没导致状态无法更新，SOLO 采用对数量化（Logarithmic Quantization），在零点附近分配更多量化级别，并结合特定随机取整机制（Stochastic Rounding），确保小信号有非零概率被采纳，同时降低量化方差；此外，通过对数基数的选择（基于 p-分位数）进一步优化量化分布。
* **有符号 EMA 更新（全局下降方向估计）**：针对量化误差导致梯度噪声方差激增的问题，SOLO 通过理论推导确定量化误差的上界，并提出基于精度的动量参数（β）调整策略，即降低 β 值以控制噪声在可接受范围内，确保收敛稳定性。
* **具体应用**：基于 AdamW 优化器，SOLO 设计了两种变体：4/2-bit AdamW（有符号状态 4 位，无符号状态 2 位）和 2-bit AdamW（两种状态均为 2 位）；采用分块量化（Block-wise Quantization）减少量化误差，并针对不同训练场景（如从头训练和微调）推荐不同的动量值。
* **关键特点**：方法与模型、任务和硬件无关，保持了通用性和灵活性；不需大幅调整超参数即可无缝应用于现有优化器。

## Experiment

* **有效性**：SOLO 在 4/2-bit 配置下性能接近甚至部分超过全精度优化器（如 LLaMA-7B 微调的 MMLU 指标，4/2-bit 为 41.03，32-bit 为 40.80），内存节省显著（训练 7B 模型节省约 45GB）；2-bit 配置虽有下降，但仍优于其他低比特基线（如 LLaVA-1.5 的 ScienceQA 得分，SOLO 2-bit 为 69.51，普通 2-bit 仅 28.93）。
* **优越性**：相比现有低比特优化器（如 8-bit 和 4-bit AdamW），SOLO 在超低比特下避免了训练崩溃，性能更稳定，尤其在从头训练和微调任务中表现出色。
* **合理性**：实验覆盖计算机视觉（CV）、语言处理（NLP）、推荐系统（RS）、大型语言模型（LLM）和视觉模型（LVM）多个领域，测试了从头训练和微调场景，数据集和模型选择全面（如 Swin-T, LLaMA-7B），评估指标多样（如 BLEU, ACC, AUC），重复运行确保可靠性；还分析了分块大小和动量参数的影响，验证了通用性。
* **不足**：未在超大规模模型（如 175B）预训练上验证效果，计算开销略增导致训练时间稍长（如 LLaMA-7B 微调，32-bit 为 8.7 小时，4/2-bit 为 9.1 小时）。

## Further Thoughts

SOLO 对 EMA 更新动态特性的系统性分析启发我们可以在其他优化算法中探索类似特性，设计更高效的低资源优化策略；对数量化思想可扩展至模型权重或激活值量化，优化小范围数据表示；动量调整策略提示在噪声敏感场景中动态调整优化参数以提高鲁棒性；未来可结合 SOLO 与分布式训练技术（如 ZeRO）进一步减少通信开销，或通过自适应量化级别根据训练阶段动态调整精度，平衡性能和内存。