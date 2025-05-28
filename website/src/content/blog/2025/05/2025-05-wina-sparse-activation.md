---
title: "WINA: Weight Informed Neuron Activation for Accelerating Large Language Model Inference"
pubDatetime: 2025-05-26T02:37:32+00:00
slug: "2025-05-wina-sparse-activation"
type: "arxiv"
id: "2505.19427"
score: 0.7727673458680153
author: "grok-3-latest"
authors: ["Sihan Chen", "Dan Zhao", "Jongwoo Ko", "Colby Banbury", "Huiping Zhuang", "Luming Liang", "Tianyi Chen"]
tags: ["LLM", "Sparse Activation", "Inference Efficiency", "Weight Matrix", "Neuron Selection"]
institution: ["Microsoft", "Renmin University of China", "New York University", "South China University of Technology"]
description: "WINA 提出了一种训练无关的稀疏激活框架，通过联合隐藏状态和权重矩阵信息优化神经元选择，显著提升大型语言模型推理效率和性能。"
---

> **Summary:** WINA 提出了一种训练无关的稀疏激活框架，通过联合隐藏状态和权重矩阵信息优化神经元选择，显著提升大型语言模型推理效率和性能。 

> **Keywords:** LLM, Sparse Activation, Inference Efficiency, Weight Matrix, Neuron Selection

**Authors:** Sihan Chen, Dan Zhao, Jongwoo Ko, Colby Banbury, Huiping Zhuang, Luming Liang, Tianyi Chen

**Institution(s):** Microsoft, Renmin University of China, New York University, South China University of Technology


## Problem Background

大型语言模型（LLMs）在推理过程中面临高计算成本的挑战，尤其在资源受限环境中效率低下。
现有训练无关的稀疏激活方法多基于隐藏状态大小选择激活神经元，导致近似误差较高，影响推理精度。
WINA 旨在通过结合隐藏状态和权重矩阵信息，提出一种新的训练无关稀疏激活框架，降低误差并加速推理。

## Method

*   **核心思想:** 提出 WINA（Weight Informed Neuron Activation），一种训练无关的稀疏激活框架，通过联合考虑隐藏状态大小和权重矩阵的列向 *ℓ*₂ 范数，选择最具影响力的神经元进行激活，以减少近似误差并提升推理效率。
*   **具体实现:** 
    *   在每一层中，计算隐藏状态值与对应权重矩阵列 *ℓ*₂ 范数的乘积，作为神经元影响力的综合指标。
    *   根据该指标选择 Top-K 个神经元激活，其余置为零，形成稀疏子网络。
    *   提供理论分析，证明在列正交性假设下，WINA 的近似误差界比现有方法更紧。
    *   为适应实际 LLM，引入基于奇异值分解（SVD）的张量变换协议，通过构造正交矩阵调整权重矩阵，保持模型输出不变的同时满足理论假设。
*   **关键优势:** 不需要重新训练模型，具有即插即用的特性；通过权重信息优化激活选择，减少误差传播；支持层级特定的稀疏比例分配，进一步提升性能。

## Experiment

*   **有效性:** 实验在多个 LLM（如 Qwen-2.5-7B、Llama-2-7B、Llama-3-8B、Phi-4-14B）上进行，WINA 在相同稀疏度下平均性能优于 TEAL 和 CATS，尤其在高稀疏度（如 65%）时优势明显，例如在 Qwen-2.5-7B 上比 TEAL 高 2.94%。
*   **计算效率:** WINA 显著降低计算开销，GFLOPs 减少高达 60%-63.7%，展现了推理加速的潜力。
*   **实验设置合理性:** 实验覆盖多种模型、任务和稀疏度（25%-65%），采用层级特定稀疏比例优化性能分配，设计全面；但未与所有现有方法比较，仅聚焦 TEAL 和 CATS，存在一定局限性。

## Further Thoughts

WINA 将权重矩阵的重要性纳入稀疏激活决策的思路具有普适性，启发我们在其他深度学习模型（如视觉模型）中探索结构化信息优化稀疏化策略；其基于 SVD 的张量变换协议提示了一种通用模型调整方法，可在不改变输出的前提下优化其他性能指标；此外，是否可以通过少量微调动态调整稀疏策略以适应不同任务需求，也是一个值得探索的方向。