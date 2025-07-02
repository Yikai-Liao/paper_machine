---
title: "Accurate Parameter-Efficient Test-Time Adaptation for Time Series Forecasting"
pubDatetime: 2025-06-29T23:09:35+00:00
slug: "2025-06-petsa-time-series-adaptation"
type: "arxiv"
id: "2506.23424"
score: 0.5280934016264075
author: "grok-3-latest"
authors: ["Heitor R. Medeiros", "Hossein Sharifi-Noghabi", "Gabriel L. Oliveira", "Saghar Irandoust"]
tags: ["Time Series Forecasting", "Test-Time Adaptation", "Parameter Efficiency", "Dynamic Calibration", "Loss Optimization"]
institution: ["Borealis AI, Montreal, Canada", "Dept. of Systems Engineering, ETS Montreal, Canada"]
description: "本文提出 PETSA 框架，通过轻量级校准模块和多损失优化策略，实现参数高效的测试时适应，显著提升时间序列预测模型在非平稳环境下的性能。"
---

> **Summary:** 本文提出 PETSA 框架，通过轻量级校准模块和多损失优化策略，实现参数高效的测试时适应，显著提升时间序列预测模型在非平稳环境下的性能。 

> **Keywords:** Time Series Forecasting, Test-Time Adaptation, Parameter Efficiency, Dynamic Calibration, Loss Optimization

**Authors:** Heitor R. Medeiros, Hossein Sharifi-Noghabi, Gabriel L. Oliveira, Saghar Irandoust

**Institution(s):** Borealis AI, Montreal, Canada, Dept. of Systems Engineering, ETS Montreal, Canada


## Problem Background

时间序列预测（Time Series Forecasting, TSF）在天气预测、交通监控和金融建模等领域至关重要，但现实世界中的时间序列数据往往是非平稳的（non-stationary），数据分布随时间发生变化（如季节性或领域偏移），导致预训练模型性能下降。
传统的测试时适应（Test-Time Adaptation, TTA）方法通过在推理时更新模型来应对分布偏移，但通常需要访问源数据或更新整个模型，导致计算和内存开销高，且测试时信息有限使得适应效果不稳定。

## Method

*   **核心思想:** 提出参数高效时间序列适应框架（PETSA），在测试时通过轻量级校准模块动态调整输入和输出特征，而不修改预训练模型的核心参数，以应对分布偏移。
*   **具体实现:** 
    *   **校准模块:** 在输入和输出端引入基于低秩适配器（low-rank adapters）和动态门控机制（dynamic gating）的校准模块。低秩适配器通过分解权重矩阵（如 W = A·B，其中 A 和 B 为低秩矩阵）减少参数量；动态门控通过可学习参数 α 和 tanh 函数对输入进行条件化调整，确保校准适应具体数据特征。
    *   **适应过程:** 在测试时，仅更新校准模块参数，利用部分和延迟的真实标签（ground truth）进行在线调整，预训练预测模型保持冻结状态。
    *   **损失函数设计:** 提出一种组合损失函数，包括：
        1. **Huber 损失:** 增强对异常值的鲁棒性，通过超参数 δ 控制对大误差的敏感度。
        2. **频域损失:** 通过快速傅里叶变换（FFT）对齐预测和真实值的频谱，保留时间序列的周期性模式。
        3. **分块结构损失:** 捕捉局部相关性、均值和方差，促进预测结果的结构对齐。
    *   **优化策略:** 损失函数分为部分标签损失（partial loss）和延迟全标签损失（delayed full loss），通过加权组合（如超参数 β 控制频域损失权重）优化校准模块。
*   **关键优势:** 参数高效（仅更新少量参数）、适应性强（通过多损失函数弥补有限适应能力），适用于多种预测模型架构。

## Experiment

*   **有效性:** 在多个时间序列基准数据集（如 ETTh1、ETTm1、Exchange、Weather）上，PETSA 显著提升了预测性能（以 MSE 指标衡量），在所有预测窗口长度（96 到 720）上均优于或与基线方法 TAFAS 相当，尤其在长窗口预测中表现突出。
*   **参数效率:** PETSA 的参数量远低于 TAFAS，例如在窗口 720 时，参数量减少了 33.6 倍，内存占用仅为 TAFAS 的几分之一，展现了极高的效率。
*   **实验设置合理性:** 实验覆盖了 Transformer、线性模型和 MLP 等多种预测模型架构，数据集选择具有代表性，充分验证了方法在不同非平稳场景下的适应能力；然而，未详细分析数据偏移的具体特征（如偏移类型或幅度），可能限制普适性评估。
*   **局限性:** 消融实验对损失组件的独立贡献分析不足，部分超参数（如频域损失权重 β）的选择对性能影响较大，需进一步调优。

## Further Thoughts

PETSA 的多损失函数设计（尤其是频域损失和分块结构损失）为处理时间序列的周期性和局部模式提供了新思路，可扩展至其他序列建模任务（如自然语言处理中的周期性数据分析）；此外，低秩适配器与动态门控的结合为参数高效适应提供了通用框架，未来可应用于在线学习或实时推荐系统；一个潜在改进方向是通过元学习或强化学习动态调整损失函数权重，以自适应地优化适应效果，而非依赖固定超参数。