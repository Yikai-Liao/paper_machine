---
title: "HSplitLoRA: A Heterogeneous Split Parameter-Efficient Fine-Tuning Framework for Large Language Models"
pubDatetime: 2025-05-05T17:09:19+00:00
slug: "2025-05-heterogeneous-split-lora"
type: "arxiv"
id: "2505.02795"
score: 0.7670504264819312
author: "grok-3-latest"
authors: ["Zheng Lin", "Yuxin Zhang", "Zhe Chen", "Zihan Fang", "Xianhao Chen", "Praneeth Vepakomma", "Wei Ni", "Jun Luo", "Yue Gao"]
tags: ["LLM", "Parameter-Efficient Fine-Tuning", "Split Learning", "Heterogeneous Computing", "Adaptation"]
institution: ["Fudan University", "University of Hong Kong", "Mohamed bin Zayed University of Artificial Intelligence", "Massachusetts Institute of Technology", "CSIRO", "University of New South Wales", "Nanyang Technological University"]
description: "HSplitLoRA 提出了一种异构参数高效微调框架，通过重要权重识别、自适应秩与模型分割配置及无噪声适配器聚合，显著提升了大型语言模型在异构环境下的微调性能和效率。"
---

> **Summary:** HSplitLoRA 提出了一种异构参数高效微调框架，通过重要权重识别、自适应秩与模型分割配置及无噪声适配器聚合，显著提升了大型语言模型在异构环境下的微调性能和效率。 

> **Keywords:** LLM, Parameter-Efficient Fine-Tuning, Split Learning, Heterogeneous Computing, Adaptation

**Authors:** Zheng Lin, Yuxin Zhang, Zhe Chen, Zihan Fang, Xianhao Chen, Praneeth Vepakomma, Wei Ni, Jun Luo, Yue Gao

**Institution(s):** Fudan University, University of Hong Kong, Mohamed bin Zayed University of Artificial Intelligence, Massachusetts Institute of Technology, CSIRO, University of New South Wales, Nanyang Technological University


## Problem Background

大型语言模型（LLMs）在各种下游任务中表现出色，但由于参数规模巨大，在资源受限的客户端设备上进行微调面临巨大挑战。
传统的联邦学习（FL）方法虽然通过本地训练保护数据隐私，但计算成本高，尤其是在异构计算环境中，设备不可用性问题严重影响训练效率。
此外，现有方法难以适应客户端设备的异构计算资源，导致性能下降和资源利用不足，亟需一种高效的分布式微调框架。

## Method

*   **核心思想:** 提出 HSplitLoRA 框架，基于分割学习（Split Learning, SL）和低秩适配（LoRA）微调，通过动态调整和优化策略，在异构客户端设备上实现高效的 LLMs 微调。
*   **重要权重识别（Important Weight Identification, IWI）:** 设计了一种资源归一化的梯度-权重乘积（RNGWP）指标，综合权重大小、梯度敏感性和计算成本，动态评估每个可训练权重的训练贡献；通过平衡当前重要性和历史重要性（通过加权平均和动态平衡参数），优先对重要权重配置 LoRA 适配器进行微调，从而在资源受限环境下提升效率。
*   **自适应秩和模型分割配置（Adaptive Rank and Model Splitting Configuration, ARMSC）:** 根据客户端设备的异构计算预算，动态调整 LoRA 适配器的分解秩（控制可训练参数数量）和模型分割点（决定客户端与服务器间的计算负载分配）；利用全局权重重要性指标选择最优配置，并通过阈值控制调整频率，避免频繁调整导致的不稳定，同时适应资源波动。
*   **无噪声适配器聚合（Noise-free Adapter Aggregation, NAA）:** 针对异构 LoRA 适配器的聚合问题，提出矩阵拼接方法，将不同客户端的低秩分解矩阵沿秩维度拼接后计算增量更新，避免传统数值平均引入的噪声，确保聚合的数学一致性和准确性。

## Experiment

*   **有效性:** 实验在 LLaMA-2-7B 和 GPT-2-L 模型上，使用 E2E 数据集进行自然语言生成任务，HSplitLoRA 在同构和异构设置下均显著优于基准方法（包括全参数微调 FT、集中式 LoRA CenLoRA、异构联邦 LoRA HetLoRA 和分割 LoRA SplitLoRA），在异构环境中 PPL 降低约 0.47（对比 HetLoRA），多指标（如 BLEU、NIST）提升明显。
*   **收敛速度:** HSplitLoRA 收敛时间最短，在异构设置下比 HetLoRA 快约 1.7 倍，得益于重要权重优先微调和自适应配置策略。
*   **实验设置合理性:** 实验涵盖同构和异构两种场景，客户端数量为 5，GPU 内存分布（5-20GB）模拟真实边缘设备环境，模型和数据集选择具有代表性；消融实验进一步验证了各组件（IWI、ARMSC、NAA）的独立贡献，设计全面合理，但泛化性需在更多数据集和模型上进一步验证。
*   **开销分析:** HSplitLoRA 通过减少客户端可训练参数（对比 FT 减少数百倍）和仅传输 LoRA 适配器降低通信开销，计算负载通过模型分割有效分配，整体资源效率高。

## Further Thoughts

HSplitLoRA 动态调整模型分割点和 LoRA 秩的策略启发了我思考是否可以引入强化学习或资源预测模型，基于历史性能和实时资源状态进一步优化分割决策；此外，无噪声适配器聚合的矩阵拼接方法是否可以结合降维技术或数学变换，应用于其他分布式学习框架（如联邦学习），以解决异构参数聚合中的噪声问题，同时进一步压缩通信开销。