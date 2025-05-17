---
title: "Does Scaling Law Apply in Time Series Forecasting?"
pubDatetime: 2025-05-15T11:04:39+00:00
slug: "2025-05-alinear-forecasting"
type: "arxiv"
id: "2505.10172"
score: 0.6838660662025718
author: "grok-3-latest"
authors: ["Zeyan Li", "Libing Chen", "Yin Tang"]
tags: ["Time Series Forecasting", "Lightweight Model", "Adaptive Decomposition", "Frequency Decay", "Parameter Efficiency"]
institution: ["Jinan University"]
description: "本文提出 ALinear 模型，通过自适应分解和频率衰减机制，以极低参数量（低于 1%）在时间序列预测中超越最先进模型，挑战了规模扩展定律的适用性。"
---

> **Summary:** 本文提出 ALinear 模型，通过自适应分解和频率衰减机制，以极低参数量（低于 1%）在时间序列预测中超越最先进模型，挑战了规模扩展定律的适用性。 

> **Keywords:** Time Series Forecasting, Lightweight Model, Adaptive Decomposition, Frequency Decay, Parameter Efficiency

**Authors:** Zeyan Li, Libing Chen, Yin Tang

**Institution(s):** Jinan University


## Problem Background

时间序列预测领域近年来模型规模急剧扩展，参数量从几十兆到几千兆（如 TimesNet），性能提升却与资源消耗不成正比，带来训练成本高、推理延迟和环境负担等问题；此外，现有模型未能解决‘预测 horizon 困境’，即从短期到超长期预测时性能显著下降，忽视了不同预测长度下趋势和季节性成分重要性的变化。本文旨在挑战‘规模扩展定律’在时间序列预测中的适用性，探索是否能通过轻量化模型实现高性能预测并解决 horizon 困境。

## Method

* **核心思想**：设计一个超轻量级模型 ALinear，参数量仅为 k 级（低于现有模型 1%），通过自适应机制根据预测长度动态调整对时间序列成分的处理策略，兼顾性能与效率。
* **Horizon-Aware Adaptive Decomposition**：提出基于预测长度的自适应分解框架，使用可学习参数控制的移动平均窗口大小（通过公式 α(H) = min(max(k1 + k2·H, w_min), w_max) 实现），动态分离趋势和季节性成分，确保短期预测关注细节，长期预测强调稳定性。
* **Component-Specific Projections**：对趋势和季节性成分分别应用独立的线性投影（通过权重矩阵和偏置向量），捕捉各自独特的数学特性，避免复杂架构如注意力机制带来的参数冗余，同时保持表达能力。
* **Horizon-Dependent Recombination**：引入基于预测长度的重组机制，通过 sigmoid 函数和可学习参数动态平衡趋势与季节性预测的权重（β_T(H) 和 β_S(H)），并采用渐进式频率衰减策略（通过公式 S_hat_decay(t) = S_hat(t)·exp(-λ·t)），随预测长度增加削弱高频成分影响，解决长期预测中的噪声问题。
* **效率设计**：模型参数复杂度为 O(HT)，时间复杂度为 O(HT)，空间复杂度为 O(HT)，适合资源受限环境。

## Experiment

* **性能表现**：在七个基准数据集（ETT 系列、Exchange Rate、Traffic、Weather）上，ALinear 在 71.4% 的场景中（MSE 和 MAE 指标）超越十个最先进模型，尤其在超长期预测（960 步）中 MSE 平均提升 23.7%，如在 ETTm1 数据集上比 Informer 降低 78.3% 误差，验证了自适应设计的有效性。
* **参数效率**：ALinear 参数量低于对比模型的 1%，通过提出的参数归一化性能（PNP）指标（PNP-MSE 为 105，PNP-MAE 为 68.9）证明其效率最高，适合资源受限场景。
* **实验设置合理性**：实验覆盖短期到超长期预测（48 到 960 步），数据集多样，包含不同领域和时间粒度；消融实验验证了自适应分解和频率衰减等组件的贡献；模型对超参数不敏感，增强了实用性；但实验仅限于单变量预测，未涉及多变量场景。
* **结论**：实验设计全面，性能提升显著，特别是在长期预测和效率权衡上表现出色。

## Further Thoughts

论文启发我们重新思考时间序列预测中模型复杂性与性能的关系，提示未来研究可以聚焦于任务驱动的自适应设计，例如根据数据特性（如金融或气象数据的周期性差异）定制分解策略；此外，PNP 指标为效率评估提供了新思路，可扩展到其他领域以平衡性能与资源消耗；另一个方向是将 ALinear 的轻量化设计与预训练模型结合，通过迁移学习提升泛化能力，同时保持低参数量。