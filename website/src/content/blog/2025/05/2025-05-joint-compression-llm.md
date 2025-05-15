---
title: "Semantic Retention and Extreme Compression in LLMs: Can We Have Both?"
pubDatetime: 2025-05-12T07:23:19+00:00
slug: "2025-05-joint-compression-llm"
type: "arxiv"
id: "2505.07289"
score: 0.7516461897194395
author: "grok-3-latest"
authors: ["Stanislas Laborde", "Martin Cousseau", "Antoun Yaacoub", "Lionel Prevost"]
tags: ["LLM", "Compression", "Pruning", "Quantization", "Semantic Retention"]
institution: ["Learning, Data and Robotics (LDR) ESIEA Lab, ESIEA, Paris, France"]
description: "本文通过理论框架和新型指标 SrCr，系统证明联合剪枝与量化在大型语言模型压缩中可实现优于单一方法的性能-压缩权衡，尤其在中等压缩率下提升显著。"
---

> **Summary:** 本文通过理论框架和新型指标 SrCr，系统证明联合剪枝与量化在大型语言模型压缩中可实现优于单一方法的性能-压缩权衡，尤其在中等压缩率下提升显著。 

> **Keywords:** LLM, Compression, Pruning, Quantization, Semantic Retention

**Authors:** Stanislas Laborde, Martin Cousseau, Antoun Yaacoub, Lionel Prevost

**Institution(s):** Learning, Data and Robotics (LDR) ESIEA Lab, ESIEA, Paris, France


## Problem Background

大型语言模型（LLMs）因参数规模激增（如从 GPT-3 的 1750 亿到 GPT-4 的万亿级）导致计算和内存成本高昂，部署在资源受限环境中面临巨大挑战。
现有压缩方法（如剪枝和量化）虽有进展，但单一方法在极致压缩时性能损失较大，尤其在复杂推理任务上，而联合压缩（结合剪枝和量化）的潜力尚未被充分探索，缺乏系统性框架和评估标准。
本文旨在研究联合压缩如何在极致压缩与语义能力保留之间取得更好的平衡。

## Method

*   **理论框架：** 提出‘理论压缩率’（Theoretical Compression Rate, TCr），基于剪枝比例和量化位宽计算信息压缩程度，作为公平比较不同压缩配置的基础，排除实现细节（如存储开销）的影响。
*   **顺序近似方法：** 由于真正联合优化（同时剪枝和量化）复杂性高，采用顺序应用（先剪枝后量化）作为近似，并通过误差分析（如 GPTQ 目标函数分解）验证其合理性，尤其在低稀疏度下。讨论了两种量化方式：Case A（量化所有权重）和 Case B（仅量化未剪枝权重），并通过理论和实验证明 Case A 在低稀疏度下是可行近似。
*   **语义保留评估指标：** 引入‘语义保留率’（Semantic Retention Rate, Sr）和‘语义保留压缩率’（Semantic Retention Compression Rate, SrCr），前者衡量压缩模型在任务上的性能保留，后者结合剪枝比例（平方根形式）和量化位宽（对数形式）量化压缩-性能权衡，提供可优化目标。
*   **实验配置：** 使用 SparseGPT（剪枝）和 GPTQ（量化）工具，在 LLaMA-3.1-8B 和 Mistral-7B-v0.3 模型上测试无结构剪枝、半结构化剪枝及不同位宽量化，探索单一与联合压缩的表现。

## Experiment

*   **有效性：** 实验表明联合压缩在特定配置下显著优于单一方法，例如在 81.25% TCr 下，25% 剪枝结合 4 位量化的语义保留率比纯 3 位量化提升约 20%，显示出更好的压缩-性能权衡。
*   **全面性：** 实验覆盖多个压缩点（75%、81.25%、87.5% TCr），在 MMLU-Pro、BBH 和 MATH 三个基准数据集上测试，任务类型多样（知识、推理、数学），设置合理。不同模型（LLaMA 和 Mistral）表现出架构差异对压缩的敏感性，增加了结果普适性。
*   **局限性：** 顺序近似可能未完全挖掘联合优化的潜力，实验仅限于 7-8B 规模模型，未涉及更大规模模型的压缩特性。半结构化剪枝（如 2:8 模式）虽表现良好，但部分模式尚未被硬件加速支持，实际部署效果待验证。

## Further Thoughts

联合压缩的协同效应提示可以进一步探索多方法联合优化（如结合低秩分解或知识蒸馏）；SrCr 指标的设计启发是否可将任务难度或模型架构特性纳入评估，以适应不同场景；半结构化剪枝结合量化的良好表现表明硬件感知压缩是未来方向，或许可以研究动态压缩策略，根据硬件资源实时调整压缩配置。