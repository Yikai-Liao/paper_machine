---
title: "HSplitLoRA: A Heterogeneous Split Parameter-Efficient Fine-Tuning Framework for Large Language Models"
pubDatetime: 2025-05-05T17:09:19+00:00
slug: "2025-05-hsplitlora-finetuning"
type: "arxiv"
id: "2505.02795"
score: 0.7670504264819312
author: "grok-3-latest"
authors: ["Zheng Lin", "Yuxin Zhang", "Zhe Chen", "Zihan Fang", "Xianhao Chen", "Praneeth Vepakomma", "Wei Ni", "Jun Luo", "Yue Gao"]
tags: ["LLM", "Parameter Efficient Tuning", "Split Learning", "Heterogeneous Computing", "Adapter Aggregation"]
institution: ["Fudan University", "University of Hong Kong", "Mohamed bin Zayed University of Artificial Intelligence", "Massachusetts Institute of Technology", "Data61, CSIRO", "University of New South Wales", "Nanyang Technological University"]
description: "HSplitLoRA 提出了一种异构参数高效微调框架，通过分割学习和 LoRA 技术结合，动态调整资源配置和无噪声适配器聚合，显著提升了大型语言模型在资源受限异构环境下的微调效率和性能。"
---

> **Summary:** HSplitLoRA 提出了一种异构参数高效微调框架，通过分割学习和 LoRA 技术结合，动态调整资源配置和无噪声适配器聚合，显著提升了大型语言模型在资源受限异构环境下的微调效率和性能。 

> **Keywords:** LLM, Parameter Efficient Tuning, Split Learning, Heterogeneous Computing, Adapter Aggregation

**Authors:** Zheng Lin, Yuxin Zhang, Zhe Chen, Zihan Fang, Xianhao Chen, Praneeth Vepakomma, Wei Ni, Jun Luo, Yue Gao

**Institution(s):** Fudan University, University of Hong Kong, Mohamed bin Zayed University of Artificial Intelligence, Massachusetts Institute of Technology, Data61, CSIRO, University of New South Wales, Nanyang Technological University


## Problem Background

大型语言模型（LLM）因参数量巨大，在资源受限的客户端设备上进行全参数微调不可行，尤其在隐私敏感场景下数据无法共享；联邦学习（FL）虽能保护隐私，但计算成本高且难以适应异构设备（不同设备计算能力差异大），设备不可用性问题和异构适配器聚合的复杂性进一步加剧挑战；论文提出 HSplitLoRA 框架，旨在通过分割学习（Split Learning, SL）和参数高效微调（PEFT）结合，解决资源受限环境下的高效微调问题。

## Method

* **核心思想**：通过分割学习将 LLM 分为客户端和服务器端部分，结合低秩适配（LoRA）技术减少参数更新量，并针对异构设备设计动态调整策略和无噪声聚合机制，实现高效微调。
* **重要权重识别（Important Weight Identification, IWI）**：提出资源归一化的梯度-权重乘积（RNGWP）指标，综合权重大小、梯度敏感性和计算成本，动态评估每个可训练权重的训练贡献；通过加权平均当前重要性和历史重要性（平衡参数随训练阶段和重要性波动调整），优先对重要权重配置 LoRA 适配器，提升资源利用效率。
* **自适应秩与模型分割配置（Adaptive Rank and Model Splitting Configuration, ARMSC）**：根据客户端设备的计算预算，动态调整 LoRA 适配器的分解秩（rank）和模型分割点（split point）；基于权重重要性指标，优先为重要权重分配较高秩，同时通过全局权重重要性计算选择最优分割点，平衡客户端和服务器端计算负载；采用自适应调整策略，根据重要性变化触发分割点更新，避免频繁调整导致的不稳定。
* **无噪声适配器聚合（Noise-free Adapter Aggregation, NAA）**：针对异构设备生成的 LoRA 适配器（不同分解秩），通过矩阵拼接（concatenation）而非直接平均的方式聚合低秩分解矩阵，确保数学一致性，避免传统聚合方法引入的噪声；聚合后增量更新直接融入预训练模型，下一轮重新初始化适配器矩阵，保持训练稳定性。

## Experiment

* **有效性**：HSplitLoRA 在 LLaMA-2-7B 和 GPT-2-L 模型上（E2E 数据集）显著优于基准方法（Full-parameter Fine-Tuning, Centralized LoRA, Heterogeneous Federated LoRA, SplitLoRA），在异构场景下 PPL 降低 0.06-0.47（LLaMA-2-7B），收敛时间缩短约 1.3-1.7 倍；在同构场景下同样表现出最快收敛速度和最高精度。
* **合理性**：实验设置涵盖同构和异构两种场景，客户端 GPU 内存限制均匀分布（5-20GB for LLaMA-2-7B），通过多种指标（BLEU, NIST, METEOR, ROUGE-L, CIDEr）验证性能；消融实验分别验证了 IWI、ARMSC 和 NAA 的贡献，设计较为全面。
* **局限性**：实验未充分探讨通信开销（激活值和梯度传输）对性能的影响，客户端数量（5 个）较少，未体现大规模分布式场景的挑战；此外，实验主要聚焦 NLG 任务，泛化性需进一步验证。

## Further Thoughts

HSplitLoRA 的自适应资源配置思路启发我们是否可以通过机器学习预测设备负载，进一步优化秩和分割点的动态调整；无噪声聚合的矩阵拼接方法是否可通过加权机制（基于设备贡献）提升效果，或推广至其他 PEFT 技术如 Adapter；此外，分割学习在隐私保护和通信成本间存在权衡，是否可结合差分隐私或加密技术，在提升安全性的同时优化通信效率？