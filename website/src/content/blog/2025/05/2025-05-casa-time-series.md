---
title: "CASA: CNN Autoencoder-based Score Attention for Efficient Multivariate Long-term Time-series Forecasting"
pubDatetime: 2025-05-04T06:46:21+00:00
slug: "2025-05-casa-time-series"
type: "arxiv"
id: "2505.02011"
score: 0.2758672069698342
author: "grok-3-latest"
authors: ["Minhyuk Lee", "HyeKyung Yoon", "MyungJoo Kang"]
tags: ["Time Series Forecasting", "Attention Mechanism", "CNN Autoencoder", "Multivariate Analysis", "Computational Efficiency"]
institution: ["Seoul National University"]
description: "本文提出 CNN 自动编码器分数注意力机制（CASA），通过线性复杂度的设计替代传统自注意力，显著提升多变量长期时间序列预测性能并降低计算资源需求。"
---

> **Summary:** 本文提出 CNN 自动编码器分数注意力机制（CASA），通过线性复杂度的设计替代传统自注意力，显著提升多变量长期时间序列预测性能并降低计算资源需求。 

> **Keywords:** Time Series Forecasting, Attention Mechanism, CNN Autoencoder, Multivariate Analysis, Computational Efficiency

**Authors:** Minhyuk Lee, HyeKyung Yoon, MyungJoo Kang

**Institution(s):** Seoul National University


## Problem Background

多变量长期时间序列预测（LTSF）在天气预测、交通分析等领域至关重要，但现有基于 Transformer 的模型面临高时间复杂度、计算资源需求大、训练饱和以及跨维度交互捕捉不足的问题，尤其是在多变量场景下，传统自注意力机制未能有效考虑变量间相关性，导致预测性能受限。

## Method

*   **核心思想:** 提出 CNN 自动编码器分数注意力机制（CASA），作为传统自注意力机制的替代，通过一维 CNN 自动编码器近似计算注意力分数，捕捉变量间的跨维度相关性，同时降低计算复杂度。
*   **具体实现:** 
    *   将每个变量视为一个通道，采用一维卷积操作替代传统自注意力中的仿射变换，确保变量间依赖性在计算注意力分数时被考虑。
    *   使用倒瓶颈自动编码器结构（Inverted Bottleneck Autoencoder），先将低维特征嵌入高维空间以增强表达能力，再压缩回低维以保留关键跨变量信息。
    *   最终通过 softmax 处理分数并与值（Value）进行元素级乘法，生成注意力输出。
*   **架构整合:** CASA 模块嵌入到 Transformer 编码器中，仅替换注意力机制，保持整体架构不变，同时支持不同的分词技术（如点式、通道式、块式）。
*   **复杂度优势:** 理论上，CASA 的计算复杂度随变量数量（N）、输入长度（L）和预测长度（H）线性增长（O(NL + NH)），显著低于传统自注意力的二次复杂度（O(NL^2 + NH)）。

## Experiment

*   **有效性:** CASA 在 8 个真实世界数据集上的实验中，在 54/64 个指标中排名第一，平均指标中 14/16 排名最高，显著优于 iTransformer、PatchTST 和 SOFTS 等基线模型，尤其在大型数据集（如 Traffic 和 Weather）上表现出色。
*   **效率提升:** 相比 Transformer 变体，CASA 内存使用减少高达 77.7%，推理速度提升 44.0%，验证了其线性复杂度的理论优势。
*   **适应性:** 通过与不同分词技术的 Transformer 模型集成，CASA 在 40/42 个结果中提升了性能，证明其模型无关性（model-agnostic）的特性。
*   **实验设置合理性:** 实验覆盖了多个领域的数据集，变量数量从 7 到 862，预测长度从 96 到 720，设置较为全面，验证了模型在不同条件下的鲁棒性；但未充分探讨极端长序列或极高维度数据下的表现，可能存在局限性。

## Further Thoughts

CASA 的 CNN 自动编码器设计启发了我思考是否可以将其他信号处理技术（如小波变换）引入时间序列预测以增强特征提取；此外，是否可以通过自适应调整 CNN 结构或引入外部知识图谱，进一步优化模型在小规模数据集或强噪声环境下的表现；最后，CASA 的线性复杂度策略是否可推广到其他高复杂度模型（如图神经网络），以提升计算效率。