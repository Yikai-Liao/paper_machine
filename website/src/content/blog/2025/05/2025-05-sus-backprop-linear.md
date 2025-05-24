---
title: "SUS backprop: linear backpropagation algorithm for long inputs in transformers"
pubDatetime: 2025-05-21T04:00:38+00:00
slug: "2025-05-sus-backprop-linear"
type: "arxiv"
id: "2505.15080"
score: 0.5206288524020284
author: "grok-3-latest"
authors: ["Sergey Pankov", "Georges Harik"]
tags: ["Transformer", "Attention Mechanism", "Backpropagation", "Sparsity", "Long Sequence"]
institution: ["Harik Shazeer Labs", "Notbad AI Inc"]
description: "本文提出 SUS Backprop 算法，通过随机稀疏化注意力梯度流，将 Transformer 反向传播复杂度从 O(n²) 降为 O(nc)，并在长序列上实现显著计算节省，同时保持梯度方差增加极小。"
---

> **Summary:** 本文提出 SUS Backprop 算法，通过随机稀疏化注意力梯度流，将 Transformer 反向传播复杂度从 O(n²) 降为 O(nc)，并在长序列上实现显著计算节省，同时保持梯度方差增加极小。 

> **Keywords:** Transformer, Attention Mechanism, Backpropagation, Sparsity, Long Sequence

**Authors:** Sergey Pankov, Georges Harik

**Institution(s):** Harik Shazeer Labs, Notbad AI Inc


## Problem Background

Transformer 架构在处理长序列时，注意力机制的计算复杂度为 O(n²)，其中 n 是序列长度，这导致训练和推理长序列数据（如长文档或时间序列）时计算成本极高，尤其在反向传播阶段需要处理所有 token 之间的交互；
作者注意到注意力权重中大部分值非常小（即 token 间交互高度稀疏），提出通过裁剪梯度流来降低计算复杂度的可能性，旨在解决如何在不显著增加梯度方差的情况下，将反向传播复杂度从二次方降至线性。

## Method

*   **核心思想**：提出 SUS Backprop（Sparse Unbiased Stochastic Backpropagation），一种稀疏无偏随机反向传播算法，通过随机裁剪注意力权重矩阵中的大部分梯度流，仅保留每个 token 一定数量的注意力交互（由参数 c 控制），从而将反向传播复杂度从 O(n²) 降为 O(nc)。
*   **理论基础**：基于无偏梯度估计原理，通过在反向传播中引入随机掩码（Stochastic Mask）对注意力权重进行稀疏化处理，并通过上调权重（Upweighting）确保梯度估计的无偏性，避免系统性偏差影响训练。
*   **具体实现**：在注意力机制中，定义概率规则 q_ij = min{c * W_ij, 1}，其中 W_ij 是注意力权重，c 是注意力保留参数（Attention Retention Parameter），控制每个 token 保留的交互数量上限；
在反向传播时，仅计算被保留的权重对应的梯度，同时存储稀疏矩阵以减少内存和计算开销。
*   **辅助分析**：提出一个玩具模型（k-weight model）来理论分析稀疏性与梯度方差之间的权衡关系，为参数 c 的选择提供理论指导。
*   **关键优势**：不修改前向传播的注意力计算，仅在反向传播时引入稀疏性，避免影响模型输出，同时适用于现有 Transformer 架构，无需大幅调整模型设计。

## Experiment

*   **注意力分布验证**：在多个 Transformer 模型（OPT 系列和 Mistral-7B）上分析注意力权重分布，发现大多数 token 仅与少量其他 token 强相关，注意力权重高度稀疏，为 SUS Backprop 的稀疏化假设提供了实证支持。
*   **稀疏性-方差权衡**：在 OPT-125m 模型上测试不同 c 值和序列长度 n，发现当 c=20-30，n=2000 时，梯度方差增加（ρ）仅约 1%，且随着 n 增加，ρ 进一步减小，表明方法在长序列上优势显著。
*   **合理性与局限**：实验覆盖多种模型和序列长度，设置较为全面，验证了方法的计算复杂度降低效果；但缺乏实际训练效果验证（如完整预训练或微调结果），且当前实现依赖 PyTorch 稀疏矩阵模块，存在性能开销，需未来优化。
*   **计算开销**：反向传播复杂度从 O(n²) 降为 O(nc)，内存需求也从 O(n²) 降为 O(nc)，但存储稀疏矩阵可能引入额外开销，尤其当 c 较大时。

## Further Thoughts

SUS Backprop 利用注意力权重的天然稀疏性优化反向传播，启发我们是否可以将类似思路推广到其他深度学习架构（如卷积网络或图神经网络），通过根据特征重要性裁剪梯度流来降低计算成本；
此外，是否可以结合线性注意力机制（如 Performer 或 Linformer），在前向和反向传播上同时优化，形成双重效率提升；
最后，这种无偏随机梯度估计方法是否能启发硬件设计，开发针对稀疏梯度计算的专用加速器，进一步提升长序列任务的训练效率？