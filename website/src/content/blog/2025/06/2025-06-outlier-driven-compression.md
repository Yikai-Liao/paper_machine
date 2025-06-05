---
title: "Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition"
pubDatetime: 2025-06-02T09:15:13+00:00
slug: "2025-06-outlier-driven-compression"
type: "arxiv"
id: "2506.02077"
score: 0.7007584763143766
author: "grok-3-latest"
authors: ["Yoonjun Cho", "Soeun Kim", "Dongjae Jeon", "Kyelim Lee", "Beomsoo Lee", "Albert No"]
tags: ["LLM", "Quantization", "Low-Rank Decomposition", "Compression", "Initialization"]
institution: ["Yonsei University", "Hongik University"]
description: "本文提出 Outlier-Driven Low-Rank Initialization (ODLRI)，通过为低秩矩阵分配捕捉激活异常值相关权重的角色，显著提升低比特量化下大型语言模型的压缩性能和稳定性。"
---

> **Summary:** 本文提出 Outlier-Driven Low-Rank Initialization (ODLRI)，通过为低秩矩阵分配捕捉激活异常值相关权重的角色，显著提升低比特量化下大型语言模型的压缩性能和稳定性。 

> **Keywords:** LLM, Quantization, Low-Rank Decomposition, Compression, Initialization

**Authors:** Yoonjun Cho, Soeun Kim, Dongjae Jeon, Kyelim Lee, Beomsoo Lee, Albert No

**Institution(s):** Yonsei University, Hongik University


## Problem Background

大型语言模型（LLMs）因其规模庞大，在资源受限设备上的部署面临挑战，权重矩阵分解为量化矩阵（Q）和低秩矩阵（LR）的联合优化方法被用于压缩模型。
然而，现有方法在迭代优化中往往偏向某一组件，导致分解次优，无法充分发挥两者的优势，特别是在低比特量化场景下，激活值异常（outliers）会显著放大量化误差。

## Method

*   **核心思想:** 通过初始化策略为低秩矩阵（LR）和量化矩阵（Q）分配明确角色，让 LR 专门捕捉与激活值异常相关的权重，减轻量化过程中的误差，提升分解质量。
*   **具体实现:** 提出 Outlier-Driven Low-Rank Initialization (ODLRI) 方法：
    *   **异常值识别:** 利用激活矩阵的 Hessian 矩阵对角线，识别出与激活异常值相关的高方差通道（top-k channels）。
    *   **角色分配:** 通过受限激活协方差矩阵（H_o），优先将异常值敏感的权重分配给 LR 进行低秩分解，而 Q 则处理剩余的更均匀分布的权重。
    *   **初始化过程:** 对 Hessian 子矩阵进行选择性白化（selective whitening）以提高数值稳定性，随后通过截断的奇异值分解（SVD）初始化 LR 组件。
    *   **集成方式:** ODLRI 被集成到 CALDERA 框架中，仅替换低秩初始化策略，不改变后续迭代优化流程。
*   **关键优势:** 充分利用 LR 的高表示能力（通过两个低比特矩阵乘积实现更高精度），让 Q 处理更平滑的残差权重，从而在低比特场景下减少激活感知误差。

## Experiment

*   **有效性:** ODLRI 在多个模型（如 Llama2 系列、Llama3-8B、Mistral-7B）上显著降低困惑度（perplexity）和激活感知误差，并在零样本任务（如 PiQA、RTE）上提升准确率，例如在 Llama2-7B 上，rank=256 时 WikiText-2 困惑度从 6.47 降至 6.33。
*   **优越性:** 相比 CALDERA 的零初始化和低秩优先初始化，ODLRI 一致性降低量化尺度（quantization scale），使权重分布更适合低比特量化；在极低秩（如 r=16、32）下仍保持性能优势，显示出鲁棒性。
*   **实验设置合理性:** 实验覆盖多种模型架构和量化配置（2-Bit Q 搭配 4-Bit 或 16-Bit LR），在不同 rank 下测试，使用预计算 Hessian 和 RedPajama 数据集校准，评估指标包括多个 NLP 基准测试，设置全面。
*   **局限性:** 未涉及激活值或 KV 缓存量化，计算开销较高（如 Llama2-70B 量化需 48 GPU 小时）。

## Further Thoughts

ODLRI 的分角色初始化策略启发我们可以在其他混合压缩技术中探索类似思路，例如将稀疏矩阵优先分配给高频权重模式；此外，是否可以通过动态调整异常值选择标准（top-k）或引入多阶段角色调整，进一步优化分解效果，适应不同模型层级或任务特性？