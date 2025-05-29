---
title: "WINA: Weight Informed Neuron Activation for Accelerating Large Language Model Inference"
pubDatetime: 2025-05-26T02:37:32+00:00
slug: "2025-05-wina-sparse-activation"
type: "arxiv"
id: "2505.19427"
score: 0.7727673458680153
author: "grok-3-latest"
authors: ["Sihan Chen", "Dan Zhao", "Jongwoo Ko", "Colby Banbury", "Huiping Zhuang", "Luming Liang", "Tianyi Chen"]
tags: ["LLM", "Sparse Activation", "Inference Acceleration", "Weight Matrix", "Neuron Selection"]
institution: ["Microsoft", "Renmin University of China", "New York University", "South China University of Technology"]
description: "WINA 提出了一种无训练稀疏激活框架，通过联合隐藏状态大小和权重矩阵列向 *ℓ*₂ 范数选择激活神经元，在保持大型语言模型性能的同时显著降低推理计算成本。"
---

> **Summary:** WINA 提出了一种无训练稀疏激活框架，通过联合隐藏状态大小和权重矩阵列向 *ℓ*₂ 范数选择激活神经元，在保持大型语言模型性能的同时显著降低推理计算成本。 

> **Keywords:** LLM, Sparse Activation, Inference Acceleration, Weight Matrix, Neuron Selection

**Authors:** Sihan Chen, Dan Zhao, Jongwoo Ko, Colby Banbury, Huiping Zhuang, Luming Liang, Tianyi Chen

**Institution(s):** Microsoft, Renmin University of China, New York University, South China University of Technology


## Problem Background

大型语言模型（LLMs）在推理阶段面临高计算成本的挑战，现有无训练稀疏激活方法仅基于隐藏状态大小选择激活神经元，忽略权重矩阵的影响，导致近似误差较高和推理精度下降。
WINA 旨在通过结合隐藏状态大小和权重矩阵的列向 *ℓ*₂ 范数，提出一种更精确的稀疏激活策略，以在保持模型性能的同时显著降低计算开销。

## Method

*   **核心思想:** WINA（Weight Informed Neuron Activation）是一种无训练的稀疏激活框架，通过联合考虑隐藏状态的大小和权重矩阵的列向 *ℓ*₂ 范数，选择对后续层影响最大的神经元进行激活，从而减少计算量并控制近似误差。
*   **具体实现:** 
    *   在推理时，对于每一层，计算隐藏状态向量中每个元素与对应权重矩阵列向 *ℓ*₂ 范数的乘积。
    *   基于乘积值，选择前 K 个最大的神经元激活（top-K 策略），其余置为零，形成稀疏子网络。
    *   支持层级特定的稀疏率分配，通过贪婪算法优化每层稀疏率以满足全局稀疏目标，而非统一稀疏率。
*   **理论支持:** 论文证明 WINA 在单层和多层网络中均能获得比现有方法（如 TEAL）更紧的近似误差界，基于权重矩阵列向正交性和单调激活函数的假设。
*   **实践优化:** 针对实际 LLM 权重矩阵不满足列向正交性的问题，WINA 采用基于奇异值分解（SVD）的张量变换协议，将权重矩阵转换为列向正交形式，并通过计算不变性调整（如调整自注意力层和 MLP 层的投影矩阵及残差连接）确保模型输出不变。
*   **优势:** 不需要重新训练或修改模型参数，即插即用，适用于现成的 LLM 模型，同时提供理论保证和实践效果。

## Experiment

*   **有效性:** WINA 在多个 LLM 模型（如 Qwen-2.5-7B、Llama-2-7B、Llama-3-8B、Phi-4-14B）和基准数据集（如 PIQA、MMLU、GSM8K）上，显著优于现有方法 TEAL 和 CATS。例如，在 Qwen-2.5-7B 上，65% 稀疏率时，WINA 平均性能比 TEAL 高 2.94%，比 TEAL-transform 高 1.41%，尤其在高稀疏率下优势明显。
*   **计算效率:** WINA 显著降低计算量，在 65% 稀疏率下，Qwen-2.5-7B 的 FLOPs 减少 60.0%，Llama-2-7B 减少 63.7%，Phi-4-14B 减少 62.7%，表明其在资源受限环境下的实用性。
*   **实验设置合理性:** 实验覆盖不同规模模型和多种任务类型，采用 top-K 策略避免分布偏移问题，并引入 TEAL-transform 基线控制变换影响，设置全面且公平。
*   **局限性:** 实验仅与 TEAL 和 CATS 比较，未覆盖所有稀疏激活方法，且资源限制（两块 A100 GPU）可能影响更大规模模型的测试。

## Further Thoughts

WINA 结合权重矩阵和隐藏状态的思路可扩展至模型剪枝或量化领域，探索动态调整权重统计特性以优化稀疏激活；其通过 SVD 变换弥合理论与实践差距的策略对其他优化方法有借鉴意义，未来可研究更高效的变换算法；此外，是否可以根据输入数据特性或任务难度动态调整稀疏率，形成自适应稀疏策略，或与 MoE 架构结合以提升性能-效率平衡。