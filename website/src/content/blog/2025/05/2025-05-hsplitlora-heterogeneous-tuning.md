---
title: "HSplitLoRA: A Heterogeneous Split Parameter-Efficient Fine-Tuning Framework for Large Language Models"
pubDatetime: 2025-05-05T17:09:19+00:00
slug: "2025-05-hsplitlora-heterogeneous-tuning"
type: "arxiv"
id: "2505.02795"
score: 0.7800991386572494
author: "grok-3-latest"
authors: ["Zheng Lin", "Yuxin Zhang", "Zhe Chen", "Zihan Fang", "Xianhao Chen", "Praneeth Vepakomma", "Wei Ni", "Jun Luo", "Yue Gao"]
tags: ["LLM", "Parameter Efficient Tuning", "Split Learning", "Heterogeneous Computing", "Low-Rank Adaptation"]
institution: ["Fudan University", "University of Hong Kong", "Mohamed bin Zayed University of Artificial Intelligence", "Massachusetts Institute of Technology", "Data61, CSIRO", "University of New South Wales", "Nanyang Technological University"]
description: "HSplitLoRA提出了一种异构参数高效微调框架，通过分割学习和LoRA结合，动态调整重要权重、适配器秩和模型分割点，并在异构环境下显著提升大型语言模型的训练精度和收敛速度。"
---

> **Summary:** HSplitLoRA提出了一种异构参数高效微调框架，通过分割学习和LoRA结合，动态调整重要权重、适配器秩和模型分割点，并在异构环境下显著提升大型语言模型的训练精度和收敛速度。 

> **Keywords:** LLM, Parameter Efficient Tuning, Split Learning, Heterogeneous Computing, Low-Rank Adaptation

**Authors:** Zheng Lin, Yuxin Zhang, Zhe Chen, Zihan Fang, Xianhao Chen, Praneeth Vepakomma, Wei Ni, Jun Luo, Yue Gao

**Institution(s):** Fudan University, University of Hong Kong, Mohamed bin Zayed University of Artificial Intelligence, Massachusetts Institute of Technology, Data61, CSIRO, University of New South Wales, Nanyang Technological University


## Problem Background

大型语言模型（LLMs）因参数规模巨大，在资源受限的客户端设备上进行全参数微调不可行，尤其是在需要保护数据隐私的场景下。
传统的联邦学习（FL）虽能通过不共享原始数据保护隐私，但计算成本高昂，且在异构计算资源的客户端环境中，设备不可用问题（device unavailability）进一步恶化了训练性能。
论文旨在解决如何在异构计算资源的客户端设备上高效、安全地微调LLM，同时应对设备不可用和计算资源限制的挑战。

## Method

*   **核心思想:** 提出HSplitLoRA框架，结合分割学习（Split Learning, SL）和低秩适应（Low-Rank Adaptation, LoRA），通过在客户端和服务器之间分割模型计算负载，实现参数高效微调，同时针对异构计算环境设计动态调整策略。
*   **具体实现:**
    *   **重要权重识别（Important Weight Identification, IWI）：** 设计资源归一化梯度-权重乘积（RNGWP）指标，综合评估权重对训练性能的静态（权重大小）和动态（梯度敏感性）贡献以及计算成本，动态识别并优先微调重要权重。
    *   **自适应秩与模型分割配置（Adaptive Rank and Model Splitting Configuration, ARMSC）：** 根据客户端设备的异构计算预算，动态调整LoRA适配器的分解秩（rank）和模型分割点（split point），以平衡客户端与服务器的计算负载并优化训练性能。调整策略基于全局权重重要性变化触发，确保稳定性。
    *   **无噪声适配器聚合（Noise-free Adapter Aggregation, NAA）：** 通过矩阵拼接方式聚合异构LoRA适配器，避免传统数值平均方法引入的噪声，确保数学一致性并支持异构适配器的高效聚合。
*   **关键点:** 不修改预训练模型，仅通过LoRA适配器更新少量参数，结合SL减少客户端计算负担，并通过动态配置适应异构环境。

## Experiment

*   **有效性:** 在LLaMA-2-7B和GPT-2-L模型上，HSplitLoRA在同构和异构环境下均表现出色，尤其在异构环境下，困惑度（PPL）显著低于基准方法（如在LLaMA-2-7B上比HetLoRA低0.47，比SplitLoRA低0.15）。
*   **优越性:** 收敛速度明显快于其他方法（如在LLaMA-2-7B同构环境下比FT快1.4倍，比HetLoRA快1.6倍），并在自然语言生成任务的多种指标（BLEU, NIST, METEOR, ROUGE-L, CIDEr）上全面优于基准。
*   **开销:** 通过模型分割和LoRA适配器，显著减少客户端可训练参数数量（如在LLaMA-2-7B上远低于HetLoRA），降低计算和通信开销。
*   **全面性与合理性:** 实验涵盖同构和异构场景，客户端GPU内存预算分布合理（异构环境下LLaMA-2-7B为5-20GB），通过多指标验证性能，设计较为全面，但未深入探讨网络延迟影响。

## Further Thoughts

HSplitLoRA提出的自适应配置策略启发了我，动态调整LoRA秩和模型分割点的思路不仅适用于分割学习，还可能扩展到联邦学习中的个性化模型优化；此外，RNGWP指标的多维度评估方法（权重、梯度、计算成本）可应用于模型剪枝或知识蒸馏，优先保留高贡献低成本组件；无噪声适配器聚合的矩阵拼接方法也可能推广到其他参数高效微调技术（如Adapter），以解决异构聚合问题。