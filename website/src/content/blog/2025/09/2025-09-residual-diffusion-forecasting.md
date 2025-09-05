---
title: "RDIT: Residual-based Diffusion Implicit Models for Probabilistic Time Series Forecasting"
pubDatetime: 2025-09-02T14:06:29+00:00
slug: "2025-09-residual-diffusion-forecasting"
type: "arxiv"
id: "2509.02341"
score: 0.39910453621528064
author: "grok-3-latest"
authors: ["Chih-Yu Lai", "Yu-Chien Ning", "Duane S. Boning"]
tags: ["Time Series Forecasting", "Diffusion Models", "Residual Modeling", "Uncertainty Quantification", "Distribution Matching"]
institution: ["Massachusetts Institute of Technology", "Harvard University"]
description: "RDIT通过解耦点预测与残差扩散建模，结合DDIM高效推理和分布匹配算法，在概率时间序列预测中显著提升了预测精度和覆盖率。"
---

> **Summary:** RDIT通过解耦点预测与残差扩散建模，结合DDIM高效推理和分布匹配算法，在概率时间序列预测中显著提升了预测精度和覆盖率。 

> **Keywords:** Time Series Forecasting, Diffusion Models, Residual Modeling, Uncertainty Quantification, Distribution Matching

**Authors:** Chih-Yu Lai, Yu-Chien Ning, Duane S. Boning

**Institution(s):** Massachusetts Institute of Technology, Harvard University


## Problem Background

概率时间序列预测（Probabilistic Time Series Forecasting, PTSF）在金融、医疗等领域至关重要，但现有方法在分布建模上存在不足，难以捕捉非线性关系和变量间依赖，同时训练目标（如MAE）与评估指标（如CRPS）的不匹配导致预测分布可能过于自信或分散不足。
论文指出，简单的点估计器加上基于训练误差的零均值高斯分布已能达到较好性能，提出新方法需同时在点预测和不确定性建模上超越这一强基线。

## Method

*   **核心框架：两阶段解耦设计**：
    *   **第一阶段 - 点估计**：采用现有的时间序列预测模型（如TimeFilter）生成条件中位数的点预测（即预测未来值的中间值），通过最小化MAE优化预测精度。
    *   **第二阶段 - 残差分布建模**：针对点预测与真实值之间的残差，使用基于扩散的模型（Denoising Diffusion Implicit Models, DDIM）建模其分布，残差被规范化（除以训练误差标准差）以确保扩散过程的起点和终点统计一致性，提升训练稳定性。
*   **网络架构**：残差建模中引入双向Mamba网络，通过前向和反向处理捕捉时间序列中的双向依赖关系，增强模型对时间模式的理解，同时在去噪输出中加入原始噪声比例以增加归纳偏置。
*   **分布匹配优化**：
    *   **Error-aware Expansion (EAE)**：基于理论推导，计算最小化CRPS的理想标准差，通过残差幅度的估计动态调整预测分布的方差，避免MAE训练目标导致的过自信问题。
    *   **Coverage Optimization (CO)**：在验证集上校准预测区间，通过二分搜索确定各分位数区间的扩展因子，确保预测覆盖率与目标一致，解决模型偏置或过拟合导致的分布失配。
*   **高效推理**：采用DDIM框架，通过非马尔可夫推理过程减少扩散步骤（从1000步减至约10步），在生成多样性与计算效率间取得平衡，适合实时应用场景。

## Experiment

*   **有效性**：RDIT在8个多变量数据集上（涵盖交通、天气、电力等多个领域），针对不同预测长度（24到720），在CRPS和PICP距离指标上几乎全面优于10个基线方法（包括点估计模型如TimeFilter、扩散模型如D3U、预训练模型如Chronos），表明其在概率预测中的强大适应性。
*   **点预测表现**：在MAE和MSE等点指标上，RDIT同样表现出色，优于大多数概率预测方法，证明其未因关注不确定性建模而牺牲点预测精度。
*   **实验设置合理性**：实验设计全面，数据集选择覆盖多种领域和时间粒度，预测长度从短期到长期，基线方法多样化，确保结果的普适性；消融研究验证了各组件（如DDIM、EAE、CO）的增益效果。
*   **局限性与开销**：部分数据集上CRPS改进幅度较小（<1%），表明训练误差标准差已较好捕捉不确定性；过度去噪可能导致过拟合，需分布匹配进一步优化；推理开销主要来自DDIM的去噪步骤，但通过减少步数已显著降低计算负担。

## Further Thoughts

RDIT的解耦思想（点预测与不确定性建模分离）可推广至其他领域，如图像生成中分离内容生成与噪声建模，或自然语言处理中分离语义预测与风格多样性；此外，EAE和CO的分布匹配策略或许适用于任意概率建模任务，通过后处理优化评估指标；DDIM的高效推理机制对实时性要求高的场景（如金融预测）有借鉴意义，值得在其他生成模型中探索类似加速策略。